import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np

from PIL import Image
from torchvision import transforms

from utils import load_data, load_vision_data, VisionData
from agent0.vision_model import Vision, save_model

def getRMSE(list_preds, list_targets, idx):
    predicted = np.array([x[idx] for x in list_preds])
    targets = np.array([x[idx] for x in list_targets])
    return np.sqrt(((predicted - targets)**2).mean())

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Vision(normalize=True, inference=True).to(device)

    # loss = torch.nn.MSELoss()
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_data = load_vision_data(args.train_path, batch_size=args.batch_size)
    # valid_data = load_data(args.valid_path, batch_size=args.batch_size)

    #Sample image
    train_dataset = VisionData(args.train_path)
    sample_image0 = train_dataset[394]
    sample_image1 = train_dataset[423]
    sample_image2 = train_dataset[473]


    #Valid images
    image_to_tensor = transforms.ToTensor()
    image_target_size = (130,100)
    img = Image.open(os.path.join(args.train_path, 'player02_00195.png'))
    # img = img.resize(image_target_size)
    valid_image0 = image_to_tensor(img)
    # print(valid_image0.shape)

    img = Image.open(os.path.join(args.train_path, 'player02_00254.png'))
    # img = img.resize(image_target_size)
    valid_image1 = image_to_tensor(img)

    img = Image.open('player50b.png') #player02_00627
    # img = img.resize(image_target_size)
    valid_image2 = image_to_tensor(img)

    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)
        # valid_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'valid_{}'.format(args.log_suffix)), flush_secs=1)

    global_step = 0
    for e in range(args.epochs):
        model.train()
        print('Epoch:',e)
        # all_targets = []
        # all_predictions = []
        for batch in train_data:
            images = batch[0].to(device)
            labels = batch[2].to(device) #Image at location 2
            # print('batch', images.shape)
            # all_targets.append(batch[2].cpu().numpy())

            pred = model(images)
            # all_predictions.append(pred.cpu().detach().numpy())
            # print(labels.shape, pred.shape)
            l = loss(pred.cpu(), labels.cpu().float())
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1
        # train_logger.add_scalar('RMSE_steer', getRMSE(all_predictions, all_targets, 0),global_step=e)
        # train_logger.add_scalar('RMSE_acceleration', getRMSE(all_predictions, all_targets, 1),global_step=e)
        # train_logger.add_scalar('RMSE_brake', getRMSE(all_predictions, all_targets, 2),global_step=e)
        model.eval()
        sample = sample_image0[0].to(device)
        im = sample_image0[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original0',sample_image0[0].cpu(), global_step=e)
        train_logger.add_image('Heatmap0',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid0',torch.sigmoid(heatmap.cpu()), global_step=e)
        train_logger.add_image('Actual0',sample_image0[2].cpu(), global_step=e)

        sample = sample_image1[0].to(device)
        im = sample_image1[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original1',sample_image1[0].cpu(), global_step=e)
        train_logger.add_image('Heatmap1',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid1',torch.sigmoid(heatmap.cpu()), global_step=e)
        train_logger.add_image('Actual1',sample_image1[2].cpu(), global_step=e)

        sample = sample_image2[0].to(device)
        im = sample_image2[0].unsqueeze(0)
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original',sample_image2[0].cpu(), global_step=e)
        train_logger.add_image('Heatmap',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid',torch.sigmoid(heatmap.cpu()), global_step=e)
        train_logger.add_image('Actual',sample_image2[2].cpu(), global_step=e)

        ##Valid images
        im = valid_image0
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original1_valid',im.cpu(), global_step=e)
        train_logger.add_image('Heatmap1_valid',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid1_valid',torch.sigmoid(heatmap.cpu()), global_step=e)
        
        im = valid_image1
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original2_valid',im.cpu(), global_step=e)
        train_logger.add_image('Heatmap2_valid',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid2_valid',torch.sigmoid(heatmap.cpu()), global_step=e)
        
        im = valid_image2
        heatmap = model(im.to(device))
        heatmap = heatmap.squeeze(0)
        train_logger.add_image('Original3_valid',im.cpu(), global_step=e)
        train_logger.add_image('Heatmap3_valid',heatmap.cpu(), global_step=e)
        train_logger.add_image('Heatmap_Sigmoid3_valid',torch.sigmoid(heatmap.cpu()), global_step=e)

        save_model(model)

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