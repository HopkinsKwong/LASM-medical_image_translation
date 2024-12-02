import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import torch

def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

# 加载检查点
def load_checkpoint(model, optimizer_G, optimizer_D, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.netG.load_state_dict(checkpoint['netG_state_dict'])
        model.netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        epoch = checkpoint['epoch']
        total_steps = checkpoint['total_steps']
        print(f"恢复训练，开始于 epoch {epoch}, steps {total_steps}")
        return epoch, total_steps
    else:
        print("没有找到检查点，开始新训练。")
        return 1, 0

# 保存检查点
def save_checkpoint(model, optimizer_G, optimizer_D, epoch, total_steps, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'total_steps': total_steps,
        'netG_state_dict': model.netG.state_dict(),
        'netD_state_dict': model.netD.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, checkpoint_path)
    print(f"保存模型检查点：{checkpoint_path}")

if __name__ == '__main__':
    opt = TrainOptions().parse()
    #Training data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()
    #validation data
    opt.phase='val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#Validation images = %d' % dataset_size_val)
    if opt.model=='cycle_gan':
        L1_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])      
        psnr_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])            
    else:
        L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])      
        psnr_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])

    model = create_model(opt)
    visualizer = Visualizer(opt)
    optimizer_G = model.optimizer_G  # 生成器优化器
    optimizer_D = model.optimizer_D  # 判别器优化器
    total_steps = 0

    # 加载检查点（如果有）
    checkpoint_path = os.path.join(save_dir, 'do_not_stop_checkpoint.pth')
    epoch_start, total_steps = load_checkpoint(model, optimizer_G, optimizer_D, checkpoint_path)

    # Record the start time of the training
    global_start_time = time.time()

    for epoch in range(epoch_start, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        #Training step
        opt.phase='train'
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                if opt.dataset_mode=='aligned_mat':
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                elif  opt.dataset_mode=='unaligned_mat':   
                    temp_visuals=model.get_current_visuals()
                    temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
                    temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    temp_visuals['fake_A']=temp_visuals['fake_A'][:,:,0:3]
                    temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    temp_visuals['rec_A']=temp_visuals['rec_A'][:,:,0:3]
                    temp_visuals['rec_B']=temp_visuals['rec_B'][:,:,0:3]
                    if opt.lambda_identity>0:
                      temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,0:3]
                      temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,0:3]                    
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                else:
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                    

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        #Validaiton step
        if epoch % opt.save_epoch_freq == 0 and epoch != 0:
            logger = open(os.path.join(save_dir, 'log.txt'), 'a')
            print(opt.dataset_mode)
            opt.phase='val'
            for i, data_val in enumerate(dataset_val):
#        		    
                model.set_input(data_val)
#        		    
                model.test()
#        		    
                fake_im=model.fake_B.cpu().data.numpy()
#        		    
                real_im=model.real_B.cpu().data.numpy() 
#        		    
                real_im=real_im*0.5+0.5
#        		    
                fake_im=fake_im*0.5+0.5
                if real_im.max() <= 0:
                    continue
                L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()
                psnr_avg[epoch-1,i]=psnr(fake_im/fake_im.max(),real_im/real_im.max())
#                  
#                 
            l1_avg_loss = np.mean(L1_avg[epoch-1])
#                
            mean_psnr = np.mean(psnr_avg[epoch-1])
#                
            std_psnr = np.std(psnr_avg[epoch-1])
#                
            print_log(logger,'Epoch %3d   l1_avg_loss: %.5f   mean_psnr: %.3f  std_psnr:%.3f ' % \
            (epoch, l1_avg_loss, mean_psnr,std_psnr))
#                
            print_log(logger,'')
            logger.close()

            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
#        		    
            model.save('latest')
#        		   
            model.save(epoch)

        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer_G, optimizer_D, epoch, total_steps, checkpoint_path)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    # Record the end time of the training
    global_end_time = time.time()
    global_training_time = global_end_time - global_start_time
    print('Total training time: {:.2f} seconds'.format(global_training_time))