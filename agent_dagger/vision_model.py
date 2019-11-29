import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class Vision(torch.nn.Module):
    
    class conv_block(torch.nn.Module):
        def __init__(self, channel_in, channel_out, stride=2, kernel_size=3, dilation=2):
            super().__init__()
            self.c1 = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=kernel_size//2)
            self.b1 = nn.BatchNorm2d(channel_out)
            self.c2 = nn.Conv2d(channel_out, channel_out, kernel_size=kernel_size, dilation=dilation, stride=1, padding=kernel_size//2)
            self.b2 = nn.BatchNorm2d(channel_out)
            
            self.downsample = None
            if channel_in != channel_out or stride != 1:
                self.downsample = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride, dilation=dilation)
            
        def forward(self, x):
            self.activation = F.relu(self.b2(self.c2(self.b1(self.c1(x))))) #consider adding relus between
            identity = x
            if self.downsample != None:
                identity = self.downsample(identity)
            return self.activation + identity
            
    class upconv_block(torch.nn.Module):
        
        def __init__(self, channel_in, channel_out, stride=2, kernel_size=3, dilation=2):    
            super().__init__()
            self.upsample = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, output_padding=1)
        
        def forward(self, x, output_pad=False):
            return F.relu(self.upsample(x))
            
    def __init__(self, layers=[32,32,64,64,128,128], normalize=True, inference=True):
        super().__init__()

        """
        Your code here
        """
        self.normalize = normalize
        self.inference = inference
        self.mean = torch.tensor([8.9478, 8.9478, 8.9478], dtype=torch.float) 
        self.std = torch.tensor([47.0021, 42.1596, 39.2562], dtype=torch.float)

        c = 3        
        self.network = torch.nn.ModuleList()
        for l in layers:
            kernel_size = 7 if c == 3 else 3
            stride = 1 if c == 3 else 2
            self.network.append(self.conv_block(c, l, stride, kernel_size, 1))
            c = l
        
        self.upnetwork = torch.nn.ModuleList()
        self.upnetwork.append(self.upconv_block(c, layers[-2]))
        c = layers[-2]
        for l in reversed(layers[:-2]):
            self.upnetwork.append(self.upconv_block(c * 2, l, 2, 3, 1)) # x2 input because of skip
            c = l
        self.classifier = torch.nn.Conv2d(c, 3, kernel_size=1)
        # self.classifier = torch.nn.Linear(c, 3)
        

    def forward(self, x):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        ##Add preprocessing
        if self.inference:
            # print('inference')
            device = x.device
            x = x.squeeze()
            if len(x.shape) == 4:
                images = []
                for i in x:
                    img = transforms.functional.to_pil_image(i.cpu())
                    x = transforms.functional.to_tensor(transforms.Resize((100,130))(img))
                    images.append(x[None].to(device))
                x = torch.cat(images)
            else:
                img = transforms.functional.to_pil_image(x.cpu())
                x = transforms.functional.to_tensor(transforms.Resize((100,130))(img))
                x = x[None].to(device)
        pass_through_x = x.clone()
        if self.normalize:
            x = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device) 
        
        activations = []
        for i, layer in enumerate(self.network):
            z = layer(x)
            activations.append(z)
            x = z
        z = self.upnetwork[0](x)
        for i, layer in enumerate(self.upnetwork[1:]):
            x = torch.cat([z[:,:, :activations[-2-i].size(2), :activations[-2-i].size(3)], activations[-2-i]], dim=1)
            z = layer(x)
        return (self.classifier(z), pass_through_x)
        # return self.classifier(z.mean([2,3]))

    def detect(self, heatmap):
        """
           Your code here.
           Implement object detection here.
           @heatmap: 1 x H x W heatmap
           @return: List of detections [(class_id, score, cx, cy), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """

        def extract_peak(heatmap, max_pool_ks=7, min_score=0.4, max_det=100):
            """
            Your code here.
            Extract local maxima (peaks) in a 2d heatmap.
            @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
            @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
            @min_score: Only return peaks greater than min_score
            @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                        heatmap value at the peak. Return no more than max_det peaks per image
            """
            
            H,W = heatmap.size()
            max_map = F.max_pool2d(heatmap[None,None], kernel_size=max_pool_ks, stride=1, padding=max_pool_ks//2)
            mask = (heatmap >= max_map) & (heatmap > min_score)
            
            mask.squeeze_(0).squeeze_(0).size()
            local_maxima = heatmap[mask]
            
            top_k = torch.topk(local_maxima, min(len(local_maxima),max_det), sorted=True)
            indices = (mask == True).nonzero()
            
            response = []
            for i in range(len(top_k.values)):
                response.append((top_k.values[i].item(), indices[top_k.indices[i]][1].item(), (indices[top_k.indices[i]][0].item())))

            return response

        heatmap.squeeze_(0)

        
        penultimate_res = extract_peak(heatmap, max_pool_ks=15) 
        ultimate_res = [(None,s,x,y) for s,x,y in penultimate_res]
        
        return ultimate_res

    def find_puck(self, heatmap, sigmoid=True, min_val=0.2, max_step=30, step_size=2):
        """
           Your code here. (extra credit)
           Implement object detection here.
           @image: 3 x H x W image
           @return: List of detections [(class_id, score cx, cy, w/2, h/2), ...],
                    return no more than 100 detections per image
           Hint: Use extract_peak here
        """
        heatmap.squeeze_(0)
        heatmap = heatmap[0].squeeze(0)
        if sigmoid:
            heatmap = torch.sigmoid(heatmap)
        
        ultimate_res = []
        centers = self.detect(heatmap)
        
        dim_H, dim_W = heatmap.shape
        dim_H -= 1
        dim_W -= 1
        for c in centers:
            W = None #W/2
            H = None #H/2
            cx = c[2]
            cy = c[3]
            for step in range(1,max_step,step_size):
                if max(0,cy - step) < dim_H * (1/5) and H is None:
                    break
                left = heatmap[cy,max(0,cx - step)]
                right = heatmap[cy,min(dim_W,cx + step)]
                top = heatmap[min(dim_H,cy + step),cx]
                bottom = heatmap[max(0,cy - step),cx]
                if (left.cpu() < min_val or right.cpu() < min_val) and W is None:
                    W = step
                if (top.cpu() < min_val or bottom.cpu() < min_val) and H is None:
                    H = step
                if H is not None and W is not None:
                    break
            if H is not None and W is not None: #We ignore if we dont have a puck that meets our position and size criteria
                size = W * H
                res = (c[0], c[1], c[2], c[3], W, H, size)
                ultimate_res.append(res)
        result = np.ones((dim_H+1, dim_W+1)) * -1
        if len(ultimate_res) > 0:
            largest = sorted(ultimate_res, key=lambda x: x[-1], reverse=True)[0]
            cx = largest[2]
            cy = largest[3]
            H = largest[5]
            W = largest[4]
            result[int(cy - H):int(cy + H),int(cx - W):int(cx + W)] = 1
        return result

def save_model(model, name='vision'):
    from torch import save
    from os import path
    if isinstance(model, Vision):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '{}.th'.format(name)))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_vision_model(name='vision'):
    from torch import load
    from os import path
    r = Vision()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '{}.th'.format(name)), map_location='cpu'))
    return r

if __name__ == '__main__':
    model = Action()