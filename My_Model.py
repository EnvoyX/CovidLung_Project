'''
The homework of the class: image and video processing.

Faxin Zhou, May 06, 2023
'''


import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class COVIDNet(nn.Module):
    '''
    Covid/control classification model
    '''
    def __init__(self, num_classes = 2):
        super(COVIDNet, self).__init__()
        self.model = models.resnet18(pretrained = True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    
class WARP_Model(nn.Module):
    def __init__(self, WARP_Model_tmp1, WARP_Model_tmp2, device = 'mps'):
        super(WARP_Model, self).__init__()
        self.device = device
        self.WARP_Model_1 = WARP_Model_tmp1.to(self.device)
        self.WARP_Model_2 = WARP_Model_tmp2.to(self.device)
        
    def forward(self, x):
        # Apply the MLP to the feature vectors
        x_hat_vec_field = self.WARP_Model_1(x)
        ST = SpatialTransformer(size = np.array([224, 224])).to(self.device)
        x_hat = ST(x[:, 0, :, :][:, np.newaxis, :, :], x_hat_vec_field).repeat([1, 3, 1, 1])

        # Apply the GNN to the graph and feature vectors
        x_tilda_vec_field = self.WARP_Model_2(x_hat)
        x_tilda = SpatialTransformer(size = np.array([224, 224]))(x_hat[:, 0, :, :][:, np.newaxis, :, :], x_tilda_vec_field).repeat([1, 3, 1, 1])
        return x_hat, x_hat_vec_field, x_tilda, x_tilda_vec_field

 
class WARP_Model_tmp1(nn.Module):
    '''
    input: bs * 3 * 224 * 224 [batchsize, channels, height, width]
    '''
    def __init__(self, in_ch = 3, out_ch = 1):
        super(WARP_Model_tmp1, self).__init__()
        # Encoder
        self.input = Conv_bare(in_ch, 16)  # (bs, 16, 224, 224)
        self.down1 = Conv_dn(16, 32)       # (bs, 32, 112, 112)
        self.down2 = Conv_dn(32, 64)       # (bs, 64, 56, 56)
        self.down3 = Conv_dn(64, 16)       # (bs, 16, 28, 28)
        self.down4 = Conv_dn(16, 16)       # (bs, 16, 14, 14)
        # Decoder
        self.up_input = Conv_bare(16, 16)  # (bs, 16, 14, 14)
        self.up1_noconv = Conv_up(16, 16)  # (bs, 16, 28, 28)
        self.up2 = Conv_up(16, 64)         # (bs, 64, 56, 56)
        self.up3 = Conv_up(64, 32)         # (bs, 32, 112, 112)
        self.up4 = Conv_up(32, 16)         # (bs, 16, 224, 224)
        self.output = Conv_bare(16, out_ch) 
        # Middle layer
        self.in_Linear = nn.Linear(in_features = 16 * 14 * 14, out_features = 512) 
        self.out_Linear = nn.Linear(in_features = 512, out_features = 16 * 14 * 14) 
    
    def forward(self, x):
        # Encoder
        d0 = self.input(x.float())
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        # Middle layer
        batch_size = d4.shape[0]
        z1 = d4.reshape(batch_size, -1)
        z = self.in_Linear(z1)
        z = nn.ReLU()(z)
        z = self.out_Linear(z)
        z = z.reshape(batch_size, 16, 14, 14)
        # Decoder
        u4 = self.up_input(z) + d4
        u3 = self.up1_noconv(u4) + d3
        u2 = self.up2(u3) + d2
        u1 = self.up3(u2) + d1
        u0 = self.up4(u1) + d0
        output = self.output(u0)
        
        return output



class WARP_Model_tmp2(nn.Module):
    '''
    input: bs * 3 * 224 * 224 [batchsize, channels, height, width]
    '''
    def __init__(self, in_ch = 3, out_ch = 1):
        super(WARP_Model_tmp2, self).__init__()
        # Encoder
        self.input = Conv_bare(in_ch, 16)  # (bs, 16, 224, 224)
        self.down1 = Conv_dn(16, 32)       # (bs, 32, 112, 112)
        self.down2 = Conv_dn(32, 64)       # (bs, 64, 56, 56)
        self.down3 = Conv_dn(64, 16)       # (bs, 16, 28, 28)
        self.down4 = Conv_dn(16, 16)       # (bs, 16, 14, 14)
        # Decoder
        self.up_input = Conv_bare(16, 16)  # (bs, 16, 14, 14)
        self.up1_noconv = Conv_up(16, 16)  # (bs, 16, 28, 28)
        self.up2 = Conv_up(16, 64)         # (bs, 64, 56, 56)
        self.up3 = Conv_up(64, 32)         # (bs, 32, 112, 112)
        self.up4 = Conv_up(32, 16)         # (bs, 16, 224, 224)
        self.output = Conv_bare(16, out_ch) 
        # Middle layer
        self.in_Linear = nn.Linear(in_features = 16 * 14 * 14, out_features = 512) 
        self.out_Linear = nn.Linear(in_features = 512, out_features = 16 * 14 * 14) 
    
    def forward(self, x):
        # Encoder
        d0 = self.input(x.float())
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        # Middle layer
        batch_size = d4.shape[0]
        z1 = d4.reshape(batch_size, -1)
        z = self.in_Linear(z1)
        z = nn.ReLU()(z)
        z = self.out_Linear(z)
        z = z.reshape(batch_size, 16, 14, 14)
        # Decoder
        u4 = self.up_input(z) + d4
        u3 = self.up1_noconv(u4) + d3
        u2 = self.up2(u3) + d2
        u1 = self.up3(u2) + d1
        u0 = self.up4(u1) + d0
        output = self.output(u0)
        
        return output


class Conv_up(nn.Module):
    '''
    Conditional Convolution upsample module.
    '''
    def __init__(self, in_ch, out_ch, kernel_size = 3):
       super(Conv_up, self).__init__() 
       self.conv = nn.Conv2d(in_ch, 
                             out_ch, 
                             kernel_size = kernel_size,
                             padding = 'same')
       self.feed = nn.Sequential(nn.BatchNorm2d(out_ch),
                                 nn.LeakyReLU(0.1, inplace = True),
                                 nn.Upsample(scale_factor = 2))
        
    def forward(self, x):
        return self.feed(self.conv(x))



class Conv_dn(nn.Module):
    '''
    Conditional Convolution downsample module.
    '''
    def __init__(self, in_ch, out_ch, kernel_size = 3):
       super(Conv_dn, self).__init__() 
       self.conv = nn.Conv2d(in_ch, 
                             out_ch, 
                             kernel_size = kernel_size,
                             padding = 'same')
       self.feed = nn.Sequential(nn.BatchNorm2d(out_ch),
                                 nn.LeakyReLU(0.1, inplace = True),
                                 nn.MaxPool2d(2))
        
    def forward(self, x):
        return self.feed(self.conv(x))



class Conv_bare(nn.Module):
    '''
    Conditional Convolution module, only change the channel number.
    '''
    def __init__(self, in_ch, out_ch, kernel_size = 3):
       super(Conv_bare, self).__init__()
       self.conv = nn.Conv2d(in_ch, 
                             out_ch, 
                             kernel_size = kernel_size,
                             padding = 'same')
        
    def forward(self, x):
        return self.conv(x)



class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    
    reference: https://github.com/ZucksLiu/DeepInterpret
    """

    def __init__(self, size, device = 'mps', mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).to(device)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nn.functional.grid_sample(src, new_locs, align_corners = True, mode = self.mode)
    
    

class warp_loss(nn.Module):
    def __init__(self, thred = 5):
        super().__init__()
        self.thred = thred

    def shift_logit_loss(self, img_orig_logit, img_hat_logit):
        '''Shift logit loss
        orig: original logit value
        pred: simulated logit value
        '''
        diff = img_orig_logit - img_hat_logit
        diff[diff > self.thred] = self.thred
        return nn.ReLU()(diff).mean()
    
    def cycle_con_loss(self, img_orig, img_tilda):
        ''' Cycle-consistent loss
        '''
        loss = torch.nn.MSELoss(reduction = 'none')
        return loss(img_orig, img_tilda).mean()
    
    def cyc_grad_loss(self, warp, penalty = 'l1', weight_grad = None):
        ''' Warping gradient loss (for linear interpolation)
        orig: the vec_field
        pred: None
        '''
        dy = torch.abs(warp[:, :, 1 :, :] - warp[:, :, : -1, :]) 
        dx = torch.abs(warp[:, :, :, 1 :] - warp[:, :, :, : -1]) 

        if penalty == 'l2':
            dy, dx = dy * dy, dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if weight_grad is not None:
            grad *= weight_grad
        return grad
    
    def forward(self, X, X_hat, X_tilda, Y, X_logit, X_hat_logit, X_tilda_logit, Y_logit, img_hat_warp, img_tilda_warp):
        loss1 = (self.shift_logit_loss(X_logit, X_tilda_logit) + self.shift_logit_loss(X_hat_logit, Y_logit)) / 2
        loss2 = (self.cycle_con_loss(X, X_tilda) + self.cycle_con_loss(X_hat, Y)) / 2
        loss3 = (self.cyc_grad_loss(img_hat_warp) + self.cyc_grad_loss(img_tilda_warp)) / 2
        return  (loss1 + loss2 + loss3) / 3
















