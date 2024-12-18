# LASM-medical_image_translation
The code for this project is the complete source code for the paper "**Localized Adaptive Style Mixing for Feature Statistics Manipulation in Medical Image Translation with Limited Data**." The LASM method mentioned in the paper helps achieve high-quality medical modality images with limited data. (The paper is currently under peer review, and in accordance with journal requirements, the original paper can be requested by contacting the author via email.)


## Data Preparation
We utilize the publicly available [BraTS 2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) and [SynthRAD2023](https://zenodo.org/records/7260705) datasets for our experiments. 

## Model training and inference
The complete source code is located in the `src` folder. To use this project, follow the steps below:
```python
# Example

## Train
python train.py --dataroot ./dataset/T2_FLAIR_830/T2_FLAIR --name T2_FLAIR_830_Mixup_BtoA --gpu_ids 0 --model mixup_resvit_one --which_model_netG res_cnn --which_direction BtoA --lambda_A 100 --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 1 --loadSize 256 --fineSize 256 --niter 50 --niter_decay 50 --save_epoch_freq 100 --checkpoints_dir checkpoints/mixup_resvit --display_id 0 --lr 0.0002 --batchSize 16 --nThreads 50

## Inference
python test.py --dataroot ./dataset/T2_FLAIR_830/T2_FLAIR  --name T2_FLAIR_830_Mixup_AtoB --gpu_ids 0 --model mixup_resvit_one --which_model_netG res_cnn --which_direction AtoB --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 30000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/mixup_resvit --checkpoints_dir checkpoints/mixup_resvit --which_epoch latest
```

## :relaxed: Easy to Use！！！ 
* The code for "Localized Adaptive Style Mixing (LASM)" is mainly found in `models/resvit_LASM.py`. The code is as follows:
```python
import torch

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

......
# Use lasm
pred_fake_lasm = discriminator(fake_AB.detach(), self.netD, use_lasm=True, device=self.device)
pred_real_lasm = discriminator(real_AB, self.netD, use_lasm=True, device=self.device)
loss_D_lasm = F.mse_loss(pred_real, pred_real_lasm) + F.mse_loss(pred_fake, pred_fake_lasm)

self.loss_D += self.loss_D_lasm
......
```

## Citation :heart_eyes: :fire:
If you find this repo useful for your research, please consider citing the paper. Although the paper has not been published yet, we kindly ask that you revisit this project code when you publish your work to check for any updates regarding the paper.

## Acknowledgment :sparkles:
This code is based on implementations by ResViT ([https://github.com/icon-lab/ResViT](https://github.com/icon-lab/ResViT)).
