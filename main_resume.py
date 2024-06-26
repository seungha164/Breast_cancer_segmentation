import argparse
import os
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms, Normalize
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from albumentations import RandomRotate90, Resize, Flip
from src import losses
from src.dataset import Dataset
from src.utils import str2bool
from src.network.CMUNet import CMUNet
import solver
import json
from src.network.modeling import Model_R2C
from torch.utils.tensorboard import SummaryWriter

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='CMUnet',
                        help='model name')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    # model
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES)
    # dataset
    parser.add_argument('--dataset_json')           #!
    parser.add_argument('--dataset', default='BUSI',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=100, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--num_workers', default=4, type=int)
    config = parser.parse_args()
    return config

def loading_DaLoader(ds_root, json_file, config):
    with open(f"{ds_root}/{json_file}", 'r') as f:
        data = json.load(f)
    trains = {'images': data['train_images'], 'labels': data['train_labels']}
    valids = {'images': data['valid_images'], 'labels': data['valid_labels']}
    
    train_transform = Compose([
        RandomRotate90(),
        Flip(),     #transforms.Flip(),
        Normalize(),
    ])
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    
    train_dataset = Dataset(
        ids=trains,
        root = './inputs',
        mode = 'with_boundary' if config['name'] == 'CMUnet_R2C' else 'base',
        num_classes=config['num_classes'],
        transform=train_transform
        )
    
    val_dataset = Dataset(
        ids=valids,
        root = './inputs',
        mode = 'with_boundary' if config['name'] == 'CMUnet_R2C' else 'base',
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    return train_loader, val_loader   # {'train' : , 'valid' : valids}
    
def main():
    writer = SummaryWriter()
    config = vars(parse_args())
    #! hyperparameter ---------------------------------------------
    # name = CMUnet(Base), CMUnet_R2C(Boundary Distance map -> Mask)
    #* CMUnet => BCEDiceLoss    | CMUnet_R2C => BoundarywithMaskLoss
    config['name']          = 'CMUnet_R2C'
    config['dataset_json']  = 'BU_ST_UD_QA.json'
    config['loss']          = 'BoundarywithMaskLoss'
    
    config['save_root']     = f"{config['name']}/" + config['dataset_json'].replace('split_', '').replace('.json', '')
    #! ------------------------------------------------------------
    os.makedirs('checkpoint/%s' % config['save_root'], exist_ok=True)

    # print('-' * 20)
    # for key in config:
    #     print('%s: %s' % (key, config[key]))
    # print('-' * 20)

    with open('checkpoint/%s/config.yml' % config['save_root'], 'w') as f:
        yaml.dump(config, f)

    #* ===== define loss function (criterion) =====
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()
    cudnn.benchmark = True

    #* ===== create model =====
    if config['name'] == 'CMUnet':
        model = CMUNet(img_ch=3, output_ch=1, l=7, k=7)
    elif config['name'] == 'CMUnet_R2C':
        model = Model_R2C(img_ch=3, output_ch=1, l=7, k=7)
        model.load_state_dict(torch.load('checkpoint/%s/model.pth' % config['save_root']))
    model = model.cuda()
    #* =======================
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    #* ===== Data loading code =====
    train_loader, val_loader = loading_DaLoader('./configs', config['dataset_json'], config)
    
    #* ============
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])
    best_iou = 0
    trigger = 0
    config['epochs'] = config['epochs'] + 300
    for epoch in range(300, config['epochs'] + 300):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log   = solver.train(train_loader, model, criterion, optimizer, modelName = config['name'], writer=writer, epoch=epoch, totalepoch=config['epochs'])                       # train for one epoch
        val_log     = solver.validate(val_loader, model, criterion, modelName = config['name'], writer=writer, epoch=epoch, totalepoch=config['epochs'])    # evaluate on validation set

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_ACC %.4f'
            % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['SE'],
               val_log['PC'], val_log['F1'], val_log['SP'], val_log['ACC']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('checkpoint/%s/log.csv' % config['save_root'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'checkpoint/%s/model.pth' % config['save_root'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        torch.cuda.empty_cache()

    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
