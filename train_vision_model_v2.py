import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np

from PIL import Image
from torchvision import transforms

from utils import load_data, load_vision_data, VisionData, to_multi_channel
from vision_model.vision_model import Vision, save_vision_model

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Vision(normalize=True, inference=True).to(device)

    # loss = torch.nn.MSELoss()
    # loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1]))
    loss = torch.nn.CrossEntropyLoss(torch.tensor([1,10,1000], dtype=torch.float))
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

    #Logger
    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)
        valid_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'valid_{}'.format(args.log_suffix)), flush_secs=1)

    # e = 0
    # im = sample_valid_image2[0].unsqueeze(0)
    # heatmap = model(im.to(device))
    # heatmap = heatmap.squeeze(0)
    # x = sample_valid_image2[1]
    # print(x.shape)
    # print(np.unique(x.detach().numpy()))
    # for d in x:
    #     print(np.unique(x.detach().numpy()))
    # train_logger.add_image('Original_1',sample_valid_image2[0].cpu(), global_step=e)
    # train_logger.add_image('Actual_1',to_multi_channel(sample_valid_image2[1]), global_step=e)
    # x = to_multi_channel(heatmap)
    # train_logger.add_image('Pred_1',x, global_step=e)

    global_step = 0
    for e in range(args.epochs):
        model.train(True)
        for bix, batch in enumerate(train_data):
            # if bix > 10:
            #      break
            
            print('Epoch: {}; Batch: {}'.format(e,bix))
            images = batch[0].to(device)
            labels = batch[1].to(device) 

            pred = model(images)
            # all_predictions.append(pred.cpu().detach().numpy())
            # print(labels.shape, pred.shape)
            l = loss(pred.cpu(), labels.cpu())
            # print(l)
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1
            
        model.eval()
        model.train(False)
        ##Train Images
        im = sample_image0[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original_1',sample_image0[0].cpu(), global_step=e)
        train_logger.add_image('Actual_1',to_multi_channel(sample_image0[1]), global_step=e)
        x = to_multi_channel(heatmap)
        train_logger.add_image('Pred_1',x, global_step=e)

        im = sample_image1[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original_2',sample_image1[0].cpu(), global_step=e)
        train_logger.add_image('Actual_2',to_multi_channel(sample_image1[1]), global_step=e)
        x = to_multi_channel(heatmap)
        train_logger.add_image('Pred_2',x, global_step=e)

        im = sample_image2[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original_3',sample_image2[0].cpu(), global_step=e)
        train_logger.add_image('Actual_3',to_multi_channel(sample_image2[1]), global_step=e)
        x = to_multi_channel(heatmap)
        train_logger.add_image('Pred_3',x, global_step=e)

        ##Valid images
        im = sample_valid_image0[0]
        heatmap = model(im.unsqueeze(0).to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original_1',im.cpu(), global_step=e)
        x = to_multi_channel(heatmap)
        train_logger.add_image('Pred_1',x, global_step=e)
        
        im = sample_valid_image1[0]
        heatmap = model(im.unsqueeze(0).to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original_2',im.cpu(), global_step=e)
        x = to_multi_channel(heatmap)
        train_logger.add_image('Pred_2',x, global_step=e)
        
        im = sample_valid_image2[0]
        heatmap = model(im.unsqueeze(0).to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original_3',im.cpu(), global_step=e)
        x = to_multi_channel(heatmap)
        train_logger.add_image('Pred_3',x, global_step=e)

        save_vision_model(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()
    train(args)