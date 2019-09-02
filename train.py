import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from dataloader import TrainDataset, collater, RandomFlip, Resizer,RandomErasing, Rotate, Color
from torch.utils.data import Dataset, DataLoader,random_split
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
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=9, help='Max training epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset or not')
    parser.add_argument('--img_size', type=int, default=320, help='Input image size')
    parser.add_argument('--verbose', type=int, default=20, help='Log verbose')
    parser.add_argument('--save_step', type=int, default=2, help='Save every save_step epochs')
    parser.add_argument('--eval_step', type=int, default=8, help='Evaluate every eval_step epochs')
    parser.add_argument('--save_path', type=str, default='./out', help='Model save path')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
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

    # # writer = SummaryWriter(log_dir=log_path)

    dataset_train = TrainDataset(transform=transforms.Compose([Rotate(),Resizer(),Color()]))
    len_train_set = int(len(dataset_train) * 0.7)
    len_val_set   = len(dataset_train) - len_train_set

    train_set, val_set = random_split(dataset_train, [len_train_set, len_val_set])
    dataloader_train = DataLoader(train_set, num_workers=8, batch_size=args.batch, collate_fn=collater,shuffle=True)
    dataloader_val = DataLoader(val_set, num_workers=8, batch_size=args.batch, collate_fn=collater)
    
    total_batch = len(dataloader_train)

    # Create torchvision model
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    retinaface = torchvision_model.create_retinaface(return_layers)
    retinaface = retinaface.cuda()
    base_lr=1e-4
    lr = base_lr
    optimizer = optim.Adam(retinaface.parameters(), lr=lr)
    retinaface = torch.nn.DataParallel(retinaface).cuda()
    retinaface.training = True
    # print(type(retinaface))
    # retinaface.load_state_dict(torch.load("./pretrained.torch"))
    retinaface.load_state_dict(torch.load("./out/pretrain11111113.pt"))
    
    
    # ####
    # print("not pretrain")
    # recall, precision, landmakr,miss= eval_widerface.evaluate(dataloader_val,retinaface)
    # print('Recall:',recall)
    # print('Precision:',precision)
    # print("landmark: ",str(landmakr))
    # print("miss: "+ str(miss))
    # sdfsdfsdf

    # ###




    lr_cos = lambda n: 0.5 * (1 + np.cos((n) / (args.epochs) * np.pi)) * base_lr
    # optimizer = optim.SGD(retinaface.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,60], gamma=0.1)
    print('Start to train.')

    epoch_loss = []
    iteration = 0

    for epoch in range(args.epochs):
        lr=lr_cos(epoch)
        print("Current lr is {}".format(lr))
        retinaface.train()
        #print('Current learning rate:',scheduler.get_lr()[0])
        # retinaface.module.freeze_bn()
        # retinaface.module.freeze_first_layer()

        # Training
        for iter_num,data in enumerate(dataloader_train):
            optimizer.zero_grad()
            classification_loss, bbox_regression_loss,ldm_regression_loss = retinaface([data['img'].cuda().float(), data['annot']])
            classification_loss = classification_loss.mean()
            bbox_regression_loss = bbox_regression_loss.mean()
            ldm_regression_loss = ldm_regression_loss.mean()

            # loss = classification_loss + 0.25 * bbox_regression_loss + 0.07 * ldm_regression_loss
            loss = classification_loss+0.1*ldm_regression_loss
            # loss = classification_loss + bbox_regression_loss + ldm_regression_loss

            loss.backward()
            optimizer.step()
            #epoch_loss.append(loss.item())
            
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
                #table = SingleTable(table_data)
                #table = DoubleTable(table_data)
                log_str +=table.table
                print(log_str)
                # write the log to tensorboard
                # writer.add_scalar('losses:',loss.item(),iteration*args.verbose)
                # writer.add_scalar('class losses:',classification_loss.item(),iteration*args.verbose)
                # writer.add_scalar('box losses:',bbox_regression_loss.item(),iteration*args.verbose)
                # writer.add_scalar('landmark losses:',ldm_regression_loss.item(),iteration*args.verbose)
                iteration +=1
        
        #scheduler.step()
        #scheduler.step(np.mean(epoch_loss))	

        # Eval
        if epoch % args.eval_step == 0:
            print('-------- RetinaFace Pytorch --------')
            print ('Evaluating epoch {}'.format(epoch))
            recall, precision, landmakr,miss= eval_widerface.evaluate(dataloader_val,retinaface)
            print('Recall:',recall)
            print('Precision:',precision)
            print("landmark: ",str(landmakr))
            print("miss: "+ str(miss))

            # writer.add_scalar('Recall:', recall, epoch*args.eval_step)
            # writer.add_scalar('Precision:', precision, epoch*args.eval_step)
            with open("aaa.txt", 'a') as f:
                f.write('-------- RetinaFace Pytorch --------(not pretrain)'+'\n')
                f.write ('Evaluating epoch {}'.format(epoch)+'\n')
                f.write('Recall:'+str(recall)+'\n')
                f.write('Precision:'+str(precision)+'\n')
                f.write("landmark: "+str(landmakr)+'\n')
                f.write("miss: "+ str(miss)+'\n')
                f.close()

        # Save model
        if (epoch + 1) % args.save_step == 0:
            torch.save(retinaface.state_dict(), args.save_path + '/mnas_epoch__ori{}.pt'.format(epoch + 1+5+1112222111))

    # writer.close()


if __name__=='__main__':
    main()