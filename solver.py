from collections import OrderedDict
import torch
from src.metrics import iou_score
from src.utils import AverageMeter
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

id2task = {
    0:'BUSI',
    1:'STU',
    2:'TestSet',
    3:'UDIAT',
    4:'QAMEBI'
}

def train(train_loader, model, criterion, optimizer, modelName, writer = None, epoch=0, totalepoch=0):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    model.train()

    for dics, ids in train_loader:
        input, target = dics['image'], dics['mask']
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        
        if modelName == 'CMUnet':
            loss = criterion(output, target)
            iou, dice, _, _, _, _, _ = iou_score(output, target)
        elif modelName == 'CMUnet_R2C':
            target_boundary_distance_map = dics['boundary_distance_map'].cuda()
            loss = criterion(output, target, target_boundary_distance_map, ids['task_id'], t=epoch, N=totalepoch)
            iou, dice, _, _, _, _, _ = iou_score(output['mask'], target)
        if writer != None:
            writer.add_scalar('Loss/Train', loss, epoch)
            writer.add_scalar('iou/Train', iou, epoch)
            writer.add_scalar('dice/Train', dice, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)
                        ])


def validate(val_loader, model, criterion, modelName, writer = None, saveRoot = './result', save=False, epoch=0, totalepoch=0):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter(),
                   'SE':AverageMeter(),
                   'PC':AverageMeter(),
                   'F1':AverageMeter(),
                   'SP':AverageMeter(),
                   'ACC':AverageMeter()
                   }

    # switch to evaluate mode
    model.eval()
    if save:
        import os
        save_scores = []  #!
        os.makedirs(f'{saveRoot}/predictions/', exist_ok=True)
    with torch.no_grad():
        for dics, ids in val_loader:
            input, target = dics['image'], dics['mask']
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            if modelName == 'CMUnet':
                loss = criterion(output, target)
                iou, dice, _, _, _, _, _ = iou_score(output, target)
            elif modelName == 'CMUnet_R2C':
                target_boundary_distance_map = dics['boundary_distance_map'].cuda()
                loss = criterion(output, target, target_boundary_distance_map, ids['task_id'], t=epoch, N=totalepoch)
                iou, dice, _, _, _, _, _ = iou_score(output['mask'], target)
            
            iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)
            if writer != None:
                writer.add_scalar('Loss/Val', loss, epoch)
                writer.add_scalar('iou/Val', iou, epoch)
                writer.add_scalar('dice/Val', dice, epoch)
            if save:
                if modelName == 'CMUnet':
                    cv2.imwrite(f'{saveRoot}/predictions/{id2task[ids["task_id"].item()]}_{ids["img_id"][0]}.png', (torch.sigmoid(output).data.cpu().numpy() > 0.5)[0, 0].astype(np.float32) * 255)
                    save_scores.append([ids['img_id'][0], iou, dice, SE, PC, F1, SP, ACC])  #!
                elif modelName == 'CMUnet_distancemap':
                    plt.axis('off')
                    plt.imshow((output['dis']).data.cpu().numpy()[0, 0], cmap='bwr')
                    plt.savefig(f'{saveRoot}/predictions/{id2task[ids["task_id"].item()]}_{ids["img_id"][0]}_b.png', bbox_inches='tight')
                    #cv2.imwrite(f'{saveRoot}/predictions/{id2task[name["task_id"].item()]}_{name["img_id"][0]}_b.png', (output['dis']).data.cpu().numpy()[0, 0].astype(np.float32) * 255)
                    cv2.imwrite(f'{saveRoot}/predictions/{id2task[ids["task_id"].item()]}_{ids["img_id"][0]}.png', (torch.sigmoid(output['mask']).data.cpu().numpy() > 0.5)[0, 0].astype(np.float32) * 255)
                    save_scores.append([ids['img_id'][0], iou, dice, SE, PC, F1, SP, ACC])  #!
                
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            avg_meters['PC'].update(PC, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))
            avg_meters['SP'].update(SP, input.size(0))
            avg_meters['ACC'].update(ACC, input.size(0))

    if save:
        df = pd.DataFrame(save_scores, columns=['ID', 'iou', 'dice', 'SE', 'PE', 'F1', 'SP', 'ACC'])
        df.to_csv(f'{saveRoot}/result.csv')
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('SE', avg_meters['SE'].avg),
                        ('PC', avg_meters['PC'].avg),
                        ('F1', avg_meters['F1'].avg),
                        ('SP', avg_meters['SP'].avg),
                        ('ACC', avg_meters['ACC'].avg)
                        ])



def detect(val_loader, model, criterion, saveRoot = './result', save=False, only_predict=True):

    # switch to evaluate mode
    model.eval()
    if save:
        import os
        
        os.makedirs(f'{saveRoot}/predictions/', exist_ok=True)
    with torch.no_grad():
        for input, name in tqdm(val_loader):
            input = input.cuda()
            output = model(input)
            if save:
                cv2.imwrite(f'{saveRoot}/predictions/{name["img_id"][0]}.png', (torch.sigmoid(output).data.cpu().numpy() > 0.5)[0, 0].astype(np.float32) * 255)
                
