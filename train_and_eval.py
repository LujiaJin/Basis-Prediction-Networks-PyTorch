import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import argparse
import time
from model import *
from data_provider import DataSet
from utils.train_utils import *
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms


def train_initialization(args, model, optimizer, scheduler, checkpoint_dir):
    if args.checkpoint != '':
        try:
            checkpoint = load_checkpoint(checkpoint_dir, args.checkpoint)
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            best_psnr = checkpoint['best_psnr']
            best_ssim = checkpoint['best_ssim']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print(
                '=> loaded checkpoint for training: '
                '(epoch {}, global_step {})'.format(
                    start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            best_psnr = -np.inf
            best_ssim = -np.inf
            print('=> no checkpoint file to be loaded for training.')
    else:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        best_psnr = -np.inf
        best_ssim = -np.inf
        print('=> training from scratch.')

    return model, start_epoch, global_step, best_loss, best_psnr, best_ssim, \
           optimizer, scheduler


def train(config, args):
    # fix the random seeds to make the training process reproducible
    setup_seed(42)

    device = torch.device('cuda' if args.cuda else 'cpu')
    color = config['color']
    batch_size = config['batch_size']
    burst_length = config['burst_length']
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    lr_decay = config['lr_decay']
    n_epoch = config['num_epochs']
    print('Configs:', config)

    # checkpoint path
    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # logs path
    logs_dir = config['logs_dir']
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_writer = SummaryWriter(logs_dir)

    # dataset and dataloader
    data_set = DataSet(config, args.eval)
    train_size = int(0.999 * len(data_set))
    val_size = int(0.001 * len(data_set))
    abandon_size = len(data_set) - train_size - val_size
    trainData, valData, _ = torch.utils.data.random_split(
        data_set, [train_size, val_size, abandon_size])
    print('Datasize for train: {}; Datasize for val: {}'.format(
        trainData.__len__(), valData.__len__()))

    data_loaders = {
        'train': DataLoader(trainData, batch_size=batch_size, shuffle=True,
                            num_workers=args.num_workers),
        'val': DataLoader(valData, batch_size=1, shuffle=True,
                          num_workers=args.num_workers)}

    # model here
    model = BPN(color=color, burst_length=burst_length,
                blind_est=config['blind_est'],
                kernel_size=config['kernel_size'],
                basis_size=config['basis_size'], upMode=config['upmode']).to(
        device)

    # loss function here
    loss_func = LossFunc(
        coeff_basic=1.0,
        coeff_anneal=1.0,
        gradient_L1=True,
        alpha=config['alpha'],
        beta=config['beta']
    )

    # Optimizer here
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    optimizer.zero_grad()

    # learning rate scheduler here
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               factor=lr_decay, patience=1)

    # training initialization
    model, start_epoch, global_step, best_loss, best_psnr, best_ssim, \
    optimizer, scheduler = train_initialization(
        args, model, optimizer, scheduler, checkpoint_dir)
    if args.mGPU:
        model = nn.DataParallel(model)

    average_loss = MovingAverage(config['save_freq'])
    average_psnr = MovingAverage(config['save_freq'])
    average_ssim = MovingAverage(config['save_freq'])

    loss_val = 0.  # variable declaration for lr scheduling

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        print('=' * 20,
              'lr={}'.format([param['lr'] for param in optimizer.param_groups]),
              '=' * 20)
        for step, (burst_noise, gt) in enumerate(data_loaders['train']):
            t1 = time.time()
            model.train()
            burst_noise = burst_noise.to(device)
            gt = gt.to(device)

            # forward
            pred_burst, pred = model(torch.flatten(burst_noise, 1, 2),
                                     burst_noise)

            # calculate loss_train
            loss_basic_train, loss_anneal_train = loss_func(pred_burst, pred,
                                                            gt, global_step)
            loss_train = loss_basic_train + loss_anneal_train

            # calculate PSNR and SSIM for train
            psnr_train = calculate_psnr(pred, gt)
            ssim_train = calculate_ssim(pred, gt)

            # validation
            loss_basic_val, loss_anneal_val, psnr_val, ssim_val = evaluate(
                model, device, global_step, data_loaders['val'], loss_func)
            loss_val = loss_basic_val + loss_anneal_val

            # add scalars to tensorboardX
            log_writer.add_scalars('LOSS', {'loss_train': loss_train,
                                            'loss_val': loss_val},
                                   global_step)
            log_writer.add_scalars('PSNR', {'psnr_train': psnr_train,
                                            'psnr_val': psnr_val},
                                   global_step)
            log_writer.add_scalars('SSIM', {'ssim_train': ssim_train,
                                            'ssim_val': ssim_val},
                                   global_step)
            log_writer.add_scalars('LOSS_basic',
                                   {'loss_basic_train': loss_basic_train,
                                    'loss_basic_val': loss_basic_val},
                                   global_step)
            log_writer.add_scalars('LOSS_anneal',
                                   {'loss_anneal_train': loss_anneal_train,
                                    'loss_anneal_val': loss_anneal_val},
                                   global_step)
            # print
            print(
                '{:-4d}\t| epoch {:2d}\t| step {:4d}\t| time:{:.2f}s\n '
                'loss_train: {:.4f}\t| loss_val: {:.4f}\n '
                'loss_basic_train: {:.4f}   | loss_basic_val: {:.4f}\t| '
                'loss_anneal_train: {:.4f}\t| loss_anneal_val: {:.4f}\n '
                'PSNR_train: {:.2f}dB\t| PSNR_val: {:.2f}dB\t| '
                'SSIM_train: {:.4f}\t| SSIM_val: {:.4f}'.format(
                    global_step, epoch, step, time.time() - t1,
                    loss_train, loss_val, loss_basic_train, loss_basic_val,
                    loss_anneal_train, loss_anneal_val, psnr_train,
                    psnr_val, ssim_train, ssim_val))

            # backward
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # update the average loss, average psnr, and average ssim
            average_loss.update(loss_val)
            average_psnr.update(psnr_val)
            average_ssim.update(ssim_val)

            # global_step
            global_step += 1

            if global_step % config['save_freq'] == 0:
                is_best_loss = True if average_loss.get_value() < best_loss \
                    else False
                is_best_psnr = True if average_psnr.get_value() > best_psnr \
                    else False
                is_best_ssim = True if average_ssim.get_value() > best_ssim \
                    else False
                if is_best_loss and is_best_psnr and is_best_ssim:
                    is_best = True
                    best_loss = average_loss.get_value()
                    best_psnr = average_psnr.get_value()
                    best_ssim = average_ssim.get_value()
                else:
                    is_best = False

                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.module.state_dict(),
                    'best_loss': best_loss,
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict()
                }
                save_checkpoint(
                    save_dict, is_best, checkpoint_dir, global_step,
                    max_keep=config['ckpt_to_keep']
                )

        print('Epoch {} is finished, time elapsed {:.2f} seconds.'.format(
            epoch, time.time() - epoch_start_time))
        # decay the learning rate
        lr_cur = [param['lr'] for param in optimizer.param_groups]
        if lr_cur[0] > 5e-6:
            scheduler.step(loss_val)
        else:
            for param in optimizer.param_groups:
                param['lr'] = 5e-6
    log_writer.close()


def evaluate(model, device, global_step, data_loader, loss_func):
    model.eval()
    loss_basic_val, loss_anneal_val, psnr_val, ssim_val = 0., 0., 0., 0.
    batch_num = 0
    for step, (burst_noise, gt) in enumerate(data_loader):
        burst_noise = burst_noise.to(device)
        gt = gt.to(device)
        pred_burst, pred = model(torch.flatten(burst_noise, 1, 2), burst_noise)
        # calculate Loss
        loss_basic, loss_anneal = loss_func(pred_burst, pred, gt, global_step)
        loss_basic_val += loss_basic.item()
        loss_anneal_val += loss_anneal.item()
        # calculate PSNR
        psnr_val += calculate_psnr(pred, gt)
        # calculate SSIM
        ssim_val += calculate_ssim(pred, gt)
        batch_num += 1

    return loss_basic_val / batch_num, loss_anneal_val / batch_num, \
           psnr_val / batch_num, ssim_val / batch_num


def test(config, args):
    # fix the random seeds to make the testing process reproducible
    setup_seed(42)

    device = torch.device('cuda' if args.cuda else 'cpu')

    print('Testing Process......')
    print('Configs:', config)

    # the path for loading checkpoint
    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir) or len(
            os.listdir(checkpoint_dir)) == 0:
        print(
            'There is no any checkpoint file in path:{}'.format(checkpoint_dir))

    # the path for saving eval images
    eval_dir = config['eval_dir']
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    files = os.listdir(eval_dir)
    for f in files:
        # Clean the results of the last evaluation！！！
        os.remove(os.path.join(eval_dir,f))

    # dataset and dataloader
    data_set = DataSet(config, args.eval)
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # model here
    model = BPN(color=config['color'], burst_length=config['burst_length'],
                blind_est=config['blind_est'],
                kernel_size=config['kernel_size'],
                basis_size=config['basis_size'], upMode=config['upmode']).to(
        device)

    # load trained model
    ckpt = load_checkpoint(checkpoint_dir, args.checkpoint)
    model.load_state_dict(ckpt['state_dict'])
    print('The BPN model has been loaded from epoch {}, n_iter {}.'.format(
        ckpt['epoch'], ckpt['global_iter']))
    if args.mGPU:
        model = nn.DataParallel(model)
    # switch the eval mode
    model.eval()

    trans = transforms.ToPILImage()

    with torch.no_grad():
        psnr = []
        ssim = []
        rmse = []
        pearsonr = []
        test_num = 0
        for i, (burst_noise, gt) in enumerate(data_loader):
            burst_noise = burst_noise.to(device)
            gt = gt.to(device)

            pred_burst, pred = model(torch.flatten(burst_noise, 1, 2),
                                     burst_noise)

            psnr_t = calculate_psnr(pred, gt)
            ssim_t = calculate_ssim(pred, gt)
            rmse_t = calculate_rmse(pred, gt)
            pearsonr_t = calculate_pearsonr(torch.flatten(pred, 1, -1),
                                            torch.flatten(gt, 1, -1))
            psnr_noisy = calculate_psnr(burst_noise[:, 0, ...], gt)
            ssim_noisy = calculate_ssim(burst_noise[:, 0, ...], gt)
            rmse_noisy = calculate_rmse(burst_noise[:, 0, ...], gt)
            pearsonr_noisy = calculate_pearsonr(
                torch.flatten(burst_noise[:, 0, ...], 1, -1),
                torch.flatten(gt, 1, -1))
            psnr.append(psnr_t)
            ssim.append(ssim_t)
            rmse.append(rmse_t)
            pearsonr.append(pearsonr_t)

            pred = torch.clamp(pred, 0.0, 1.0)
            noise = torch.clamp(burst_noise[:, 0, ...], 0.0, 1.0)

            if args.cuda:
                pred = pred.cpu()
                gt = gt.cpu()
                noise = noise.cpu()

            trans(noise.squeeze()).save(os.path.join(eval_dir,
                                                     '{}_noisy_{:.4f}dB_{:.4f}_'
                                                     '{:.4f}_{:.4f}.png'.format(
                                                         i, psnr_noisy,
                                                         ssim_noisy, rmse_noisy,
                                                         pearsonr_noisy)),
                                        quality=100)
            trans(pred.squeeze()).save(os.path.join(eval_dir,
                                                    '{}_pred_{:.4f}dB_{:.4f}_'
                                                    '{:.4f}_{:.4f}.png'.format(
                                                        i, psnr_t, ssim_t,
                                                        rmse_t, pearsonr_t)),
                                       quality=100)
            trans(gt.squeeze()).save(
                os.path.join(eval_dir, '{}_gt.png'.format(i)), quality=100)

            print(
                '{}-th image is OK, with PSNR: {:.4f}dB, SSIM: {:.4f}, RMSE: '
                '{:.4f}, Pearson-R: {:.4f}'.format(i, psnr_t, ssim_t, rmse_t,
                                                   pearsonr_t))
            test_num += 1

        psnr_mean, psnr_std = np.mean(psnr), np.std(psnr)
        ssim_mean, ssim_std = np.mean(ssim), np.std(ssim)
        rmse_mean, rmse_std = np.mean(rmse), np.std(rmse)
        pearsonr_mean, pearsonr_std = np.mean(pearsonr), np.std(pearsonr)
        print(
            'All {} images are OK, average PSNR: {:.4f} ± {:.4f}dB, SSIM: '
            '{:.4f} ± {:.4f}, RMSE: {:.4f} ± {:.4f}, Pearson-R: {:.4f} ± '
            '{:.4f}'.format(test_num, psnr_mean, psnr_std, ssim_mean, ssim_std,
                            rmse_mean, rmse_std, pearsonr_mean, pearsonr_std))
        print(
            '({:.4f}±{:.4f}dB)-({:.4f}±{:.4f})-({:.4f}±{:.4f})-'
            '({:.4f}±{:.4f})'.format(psnr_mean, psnr_std, ssim_mean, ssim_std,
                                     rmse_mean, rmse_std, pearsonr_mean,
                                     pearsonr_std))


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(
        description='parameters for training or testing')
    parser.add_argument('--config_file', dest='config_file',
                        default='configs/AWGN_RGB.conf', type=str,
                        help='path to config file')
    parser.add_argument('--config_spec', dest='config_spec',
                        default='configs/configspec.conf', type=str,
                        help='path to config spec file')
    parser.add_argument('--num_workers', '-nw', dest='num_workers', default=4,
                        type=int, help='number of workers in data loader')
    parser.add_argument('--num_threads', '-nt', dest='num_threads', default=8,
                        type=int, help='number of threads in data loader')
    parser.add_argument('--cuda', '-c', dest='cuda', action='store_true',
                        help='whether to train on the GPU')
    parser.add_argument('--mGPU', '-m', dest='mGPU', action='store_true',
                        help='whether to train on multiple GPUs')
    parser.add_argument('--eval', dest='eval', action='store_true',
                        help='whether to work on the evaluation mode')
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', type=str,
                        default='',
                        help='the checkpoint to resume for train or val')
    args = parser.parse_args()

    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    #
    config = read_config(args.config_file, args.config_spec)
    if args.eval:
        test(config, args)
    else:
        train(config, args)
