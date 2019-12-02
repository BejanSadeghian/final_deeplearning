import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np

from PIL import Image
from torchvision import transforms

from utils import load_data, load_vision_data, VisionData
from vision_model.vision_model import Vision, save_vision_model

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Vision(normalize=True, inference=True).to(device)

    # loss = torch.nn.MSELoss()
    # loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1]))
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_data = load_vision_data(args.train_path, batch_size=args.batch_size)
    valid_data = load_vision_data(args.valid_path, batch_size=args.batch_size)

    #Sample image
    train_dataset = VisionData(args.train_path)
    sample_image0 = train_dataset[394]
    sample_image1 = train_dataset[423]
    sample_image2 = train_dataset[473]


    #Valid images
    valid_dataset = VisionData(args.valid_path)
    sample_valid_image0 = valid_dataset[394]
    sample_valid_image1 = valid_dataset[423]
    sample_valid_image2 = valid_dataset[473]


    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)
        valid_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'valid_{}'.format(args.log_suffix)), flush_secs=1)

    global_step = 0
    for e in range(args.epochs):
        model.train(True)
        for bix, batch in enumerate(train_data):
            print('\rEpoch: {}; Batch: {}'.format(e,bix), end='\r')
            images = batch[0].to(device)
            labels = batch[1].to(device) 

            pred = model(images)
            # all_predictions.append(pred.cpu().detach().numpy())
            # print(labels.shape, pred.shape)
            l = loss(torch.sigmoid(pred.cpu()), labels.cpu())
            # print(l)
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1
            
        model.eval()
        im = sample_image0[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original0',sample_image0[0].cpu(), global_step=e)
        # train_logger.add_image('Heatmap0',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid0',torch.sigmoid(heatmap.cpu()), global_step=e)

        im = sample_image1[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original1',sample_image1[0].cpu(), global_step=e)
        # train_logger.add_image('Heatmap1',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid1',torch.sigmoid(heatmap.cpu()), global_step=e)

        im = sample_image2[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original2',sample_image2[0].cpu(), global_step=e)
        # train_logger.add_image('Heatmap2',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid2v',torch.sigmoid(heatmap.cpu()), global_step=e)

        ##Valid images
        im = valid_image0
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original1_valid',im.cpu(), global_step=e)
        # train_logger.add_image('Heatmap1_valid',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid1_valid',torch.sigmoid(heatmap.cpu()), global_step=e)
        
        im = valid_image1
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original2_valid',im.cpu(), global_step=e)
        # train_logger.add_image('Heatmap2_valid',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid2_valid',torch.sigmoid(heatmap.cpu()), global_step=e)
        
        im = valid_image2
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original3_valid',im.cpu(), global_step=e)
        # train_logger.add_image('Heatmap3_valid',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid3_valid',torch.sigmoid(heatmap.cpu()), global_step=e)

        save_vision_model(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train(args)