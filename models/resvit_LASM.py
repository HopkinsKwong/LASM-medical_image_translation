import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import torch.nn.functional as F

def compute_moments(x):
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    skewness = ((x - mean) ** 3).mean(dim=(2, 3), keepdim=True) / (var.sqrt() + 1e-5) ** 3
    kurtosis = ((x - mean) ** 4).mean(dim=(2, 3), keepdim=True) / (var + 1e-5) ** 2
    return mean, var, skewness, kurtosis


def high_order_FSM(x, y, alpha, eps=1e-5):
    # Compute moments for x and y
    x_mean, x_var, x_skew, x_kurt = compute_moments(x)
    y_mean, y_var, y_skew, y_kurt = compute_moments(y)

    # Normalize x using its own moments
    x_norm = (x - x_mean) / torch.sqrt(x_var + eps)

    # Apply higher-order matching
    x_fsm = x_norm * torch.sqrt(y_var + eps) + y_mean
    x_fsm += (y_skew - x_skew) * (x_norm ** 3) * torch.sqrt(y_var + eps)
    x_fsm += (y_kurt - x_kurt) * (x_norm ** 4) * torch.sqrt(y_var + eps)

    # Mix the original and matched features
    x_mix = alpha * x + (1 - alpha) * x_fsm

    return x_mix

def blockwise_FSM(x, y, alpha, block_size=64, eps=1e-5):
    # Get the size of the input feature map
    n, c, h, w = x.size()

    # Ensure the height and width are divisible by block_size
    assert h % block_size == 0 and w % block_size == 0, "Image dimensions must be divisible by block size"

    # Split the input and reference feature maps into blocks
    x_blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    y_blocks = y.unfold(2, block_size, block_size).unfold(3, block_size, block_size)

    # Reshape blocks to apply FSM
    n_blocks, c_blocks, n_h, n_w, _, _ = x_blocks.size()
    x_blocks = x_blocks.contiguous().view(n_blocks, c_blocks, -1, block_size, block_size)
    y_blocks = y_blocks.contiguous().view(n_blocks, c_blocks, -1, block_size, block_size)

    # Apply FSM on each block
    x_fsm_blocks = []
    for i in range(n_h):
        for j in range(n_w):
            x_block = x_blocks[:, :, i * n_w + j, :, :]
            y_block = y_blocks[:, :, i * n_w + j, :, :]
            x_fsm_block = high_order_FSM(x_block, y_block, alpha, eps)
            x_fsm_blocks.append(x_fsm_block)

    # Reshape back to the original feature map shape
    x_fsm_blocks = torch.stack(x_fsm_blocks, dim=2)
    x_fsm_blocks = x_fsm_blocks.view(n, c, n_h, n_w, block_size, block_size)
    x_fsm = x_fsm_blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(n, c, h, w)

    return x_fsm


def discriminator(img, netD, use_lasm=False, device='cuda'):
    x = img  # Assuming NxHxWxC
    indices = torch.randperm(x.size(0)).to(device)
    alpha = torch.rand(1).to(device)

    for layer in netD.children():
        # print(f"Processing layer: {type(layer)}")  # 打印当前层的类型
        # before_shape = x.shape
        x = layer(x)
        # after_shape = x.shape
        # print(f"Before shape: {before_shape}, After shape: {after_shape}")
        if isinstance(layer, torch.nn.Conv2d) and use_lasm:  # Check if the layer is convolutional
            y = x[indices]  # Shuffled batch
            x = blockwise_FSM(x, y, alpha)

    return x  # Assuming Nx1 for the output


class LASM_ResViT_model(BaseModel):
    def name(self):
        return 'LASM'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG,opt.vit_name,opt.fineSize,opt.pre_trained_path, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet)


        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,opt.vit_name,opt.fineSize,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B= self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A)
            self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Use lasm
        pred_fake_lasm = discriminator(fake_AB.detach(), self.netD, use_lasm=True, device=self.device)
        pred_real_lasm = discriminator(real_AB, self.netD, use_lasm=True, device=self.device)
        self.loss_D_lasm = F.mse_loss(pred_real, pred_real_lasm) + F.mse_loss(pred_fake, pred_fake_lasm)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D += self.loss_D_lasm * 1.0

        self.loss_D.backward()

        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*1
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())

                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
