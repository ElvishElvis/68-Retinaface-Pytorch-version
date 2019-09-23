import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from dataloader import TrainDataset, collater, Resizer, PadToSquare,Color,Rotate,RandomErasing,RandomFlip, ValDataset
from torch.utils.data import Dataset, DataLoader, random_split
from terminaltables import AsciiTable, DoubleTable, SingleTable
# from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import torch.distributed as dist
import eval_widerface
import torchvision
import model
import os
from torch.utils.data.distributed import DistributedSampler
import torchvision_model


def get_args():
    parser = argparse.ArgumentParser(description="Train program for retinaface.")
    parser.add_argument('--data_path', type=str,default='./widerface' ,help='Path for dataset,default WIDERFACE')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=121, help='Max training epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset or not')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--verbose', type=int, default=20, help='Log verbose')
    parser.add_argument('--save_step', type=int, default=10, help='Save every save_step epochs')
    parser.add_argument('--eval_step', type=int, default=10, help='Evaluate every eval_step epochs')
    parser.add_argument('--save_path', type=str, default='./out', help='Model save path')
    parser.add_argument('--training', help='the training mode or not ( True for Training, False for eval', type=bool, default=True)
    args = parser.parse_args()
    print(args)

    return args


def main():
    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_path = os.path.join(args.save_path,'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    data_path = args.data_path
    # dataset_train = TrainDataset(train_path,transform=transforms.Compose([RandomCroper(),()]))
    dataset_train = TrainDataset('./widerface/train/label.txt',transform=transforms.Compose([RandomErasing(),RandomFlip(),Rotate(),Color(),Resizer(),PadToSquare()]))
    # dataset_train = TrainDataset('./widerface/train/label.txt',transform=transforms.Compose([Resizer(),PadToSquare()]))
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=args.batch, collate_fn=collater,shuffle=True)
    # dataset_val = ValDataset(val_path,transform=transforms.Compose([RandomCroper()]))
    dataset_val = TrainDataset('./widerface/train/label.txt',transform=transforms.Compose([Resizer(640),PadToSquare()]))
    dataloader_val = DataLoader(dataset_val, num_workers=8, batch_size=args.batch, collate_fn=collater)
    
    total_batch = len(dataloader_train)

       

    # Create torchvision model
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    retinaface = torchvision_model.create_retinaface(return_layers)
    retinaface_ = retinaface.cuda()
    retinaface = torch.nn.DataParallel(retinaface_).cuda()
    retinaface.training = True
    base_lr=1e-7

    # pre_train = torch.load('network.torch')
    # cur=retinaface.state_dict()
    # for k, v in cur.items():
    #     if k[12:] in pre_train:
    #         print(k[12:])
    #         cur[k]=pre_train[k[12:]]
    # retinaface.load_state_dict(cur)
    retinaface.load_state_dict(torch.load("/versa/elvishelvis/RetinaYang/out/stage_5_68_full_model_epoch_121.pt"))
    lr=base_lr
    # optimizer=torch.optim.Adam(retinaface.parameters(),lr=lr)
    # fix encoder
    for name, value in retinaface.named_parameters():
        if 'Landmark' in name:
            value.requires_grad = False
    lr_cos = lambda n: 0.5 * (1 + np.cos((n) / (args.epochs) * np.pi)) * base_lr
    params = filter(lambda p: p.requires_grad==True, retinaface.parameters())
    body=filter(lambda p: p.requires_grad==False, retinaface.parameters())
    optimizer = torch.optim.Adam([
                {'params': body, 'lr': lr*3},
                {'params': params, 'lr': lr}]
                )
    #evaluation the current model
    if (args.training==False):
        print("not pretrain")
        recall, precision, landmakr,miss= eval_widerface.evaluate(dataloader_val,retinaface)
        print('Recall:',recall)
        print('Precision:',precision)
        print("landmark: ",str(landmakr))
        print("miss: "+ str(miss))
        return 
    ##
    print('Start to train.')

    epoch_loss = []
    iteration = 0
    retinaface=retinaface.cuda()
    for epoch in range(args.epochs):
        lr=lr_cos(epoch)
        
        retinaface.train()

        # Training
        for iter_num,data in enumerate(dataloader_train):
            optimizer.zero_grad()
            classification_loss, bbox_regression_loss,ldm_regression_loss = retinaface([data['img'].cuda().float(), data['annot']])
            classification_loss = classification_loss.mean()
            bbox_regression_loss = bbox_regression_loss.mean()
            ldm_regression_loss = ldm_regression_loss.mean()

            # loss = classification_loss + 1.0 * bbox_regression_loss + 0.5 * ldm_regression_loss
            loss = classification_loss + 0.15*bbox_regression_loss + 0.25*ldm_regression_loss

            loss.backward()
            optimizer.step()
            
            if iter_num % args.verbose == 0:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, iter_num, total_batch)
                table_data = [
                    ['loss name','value'],
                    ['total_loss',str(loss.item())],
                    ['classification',str(classification_loss.item())],
                    ['bbox',str(bbox_regression_loss.item())],
                    ['landmarks',str(ldm_regression_loss.item())]
                    ]
                table = AsciiTable(table_data)
                log_str +=table.table
                print(log_str)
                iteration +=1
                

        # Eval
        if epoch % args.eval_step == 0:
            with open("aaa.txt", 'a') as f:
                f.write('-------- RetinaFace Pytorch --------'+'\n')
                f.write ('Evaluating epoch {}'.format(epoch)+'\n')
                f.write('total_loss:'+str(loss.item())+'\n')
                f.write('classification'+str(classification_loss.item())+'\n')
                f.write('bbox'+str(bbox_regression_loss.item())+'\n')
                f.write('landmarks'+str(ldm_regression_loss.item())+'\n')

                f.close()
            print('-------- RetinaFace Pytorch --------')
            print ('Evaluating epoch {}'.format(epoch))
            recall, precision, landmakr,miss= eval_widerface.evaluate(dataloader_val,retinaface)
            print('Recall:',recall)
            print('Precision:',precision)
            print("landmark: ",str(landmakr))
            print("miss: "+ str(miss))
            
            with open("aaa.txt", 'a') as f:
                f.write('-------- RetinaFace Pytorch --------(not pretrain)'+'\n')
                f.write ('Evaluating epoch {}'.format(epoch)+'\n')
                f.write('Recall:'+str(recall)+'\n')
                f.write('Precision:'+str(precision)+'\n')
                f.write("landmark: "+str(landmakr)+'\n')
                f.write("miss: "+ str(miss)+'\n')
                f.close()
        # Save model
        if (epoch) % args.save_step == 0:
            torch.save(retinaface.state_dict(), args.save_path + '/stage_5_68_full_model_epoch_{}.pt'.format(epoch + 1))

    # writer.close()


if __name__=='__main__':
    main()
