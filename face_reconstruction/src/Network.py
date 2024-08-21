import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, length_of_embedding):
        super(Generator, self).__init__()
        # nz is the length of the z input vector, 
        # ngf relates to the size of the feature maps that are propagated through the generator,
        # and nc is the number of channels in the output image (set to 3 for RGB images).
        nz  = length_of_embedding #128
        nc  = 3
        
        #
        # input is Z, going into a convolution
        self.dconv1 = nn.ConvTranspose2d( length_of_embedding, 512, 4, 1, 0, bias=False)
        self.BN_dconv1 = nn.BatchNorm2d(512)
        self.conv1_1  = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv1_1 = nn.BatchNorm2d(512)
        self.conv1_2  = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv1_2 = nn.BatchNorm2d(512)
        self.conv1_3  = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv1_3 = nn.BatchNorm2d(512)
        
        # state size. (512) x 4 x 4
        self.dconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.BN_dconv2 = nn.BatchNorm2d(256)
        self.conv2_1  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv2_1 = nn.BatchNorm2d(256)
        self.conv2_2  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv2_2 = nn.BatchNorm2d(256)
        self.conv2_3  = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv2_3 = nn.BatchNorm2d(256)
        
        # state size. (256) x 8 x 8
        self.dconv3 = nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False)
        self.BN_dconv3 = nn.BatchNorm2d(128)
        self.conv3_1  = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv3_1 = nn.BatchNorm2d(128)
        self.conv3_2  = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv3_2 = nn.BatchNorm2d(128)
        self.conv3_3  = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv3_3 = nn.BatchNorm2d(128)
        
        # state size. (128) x 16 x 16
        self.dconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 2, bias=False)
        self.BN_dconv4 = nn.BatchNorm2d(64)
        self.conv4_1  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv4_1 = nn.BatchNorm2d(64)
        self.conv4_2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv4_2 = nn.BatchNorm2d(64)
        self.conv4_3  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv4_3 = nn.BatchNorm2d(64)
        
        # state size. (64) x 30 x 30
        self.dconv5 = nn.ConvTranspose2d(64, 32, 4, 2, 2, bias=False)
        self.BN_dconv5 = nn.BatchNorm2d(32)
        self.conv5_1  = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv5_1 = nn.BatchNorm2d(32)
        self.conv5_2  = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv5_2 = nn.BatchNorm2d(32)
        self.conv5_3  = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv5_3 = nn.BatchNorm2d(32)
        
        # state size. (32) x 58 x 58
        self.dconv6 = nn.ConvTranspose2d( 32, 16, 4, 2, 3, bias=False) #112
        # self.dconv6 = nn.ConvTranspose2d( 32, 16, 3, 3, 7, bias=False) #160
        self.BN_dconv6 = nn.BatchNorm2d(16)
        self.conv6_1  = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv6_1 = nn.BatchNorm2d(16)
        self.conv6_2  = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv6_2 = nn.BatchNorm2d(16)
        self.conv6_3  = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, bias=True, padding=1)
        self.BN_conv6_3 = nn.BatchNorm2d(16)
        # state size. (nc) x 116 x 116
        # state size. (nc) x 160 x 160
        
        self.conv7  = nn.Conv2d(in_channels=16, out_channels=nc, kernel_size=3, stride=1, bias=True, padding=1)        

    def forward(self, x):
        x_ = self.dconv1(x)
        x_ = self.BN_dconv1(x_)
        xd  = nn.ReLU(True)(x_)
        
        x_ = self.conv1_1(xd)
        x_ = self.BN_conv1_1(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv1_2(xc)
        x_ = self.BN_conv1_2(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv1_3(xc)
        x_ = self.BN_conv1_3(x_)
        xc = nn.ReLU(True)(x_) 
        
        xd = xd +xc
        
        # deconv 2
        x_ = self.dconv2(xd)
        x_ = self.BN_dconv2(x_)
        xd  = nn.ReLU(True)(x_)
        
        x_ = self.conv2_1(xd)
        x_ = self.BN_conv2_1(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv2_2(xc)
        x_ = self.BN_conv2_2(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv2_3(xc)
        x_ = self.BN_conv2_3(x_)
        xc = nn.ReLU(True)(x_) 
        
        xd = xd +xc
        
        
        # deconv 3
        x_ = self.dconv3(xd)
        x_ = self.BN_dconv3(x_)
        xd  = nn.ReLU(True)(x_)
        
        x_ = self.conv3_1(xd)
        x_ = self.BN_conv3_1(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv3_2(xc)
        x_ = self.BN_conv3_2(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv3_3(xc)
        x_ = self.BN_conv3_3(x_)
        xc = nn.ReLU(True)(x_) 
        
        xd = xd +xc
        
        
        # deconv 4
        x_ = self.dconv4(xd)
        x_ = self.BN_dconv4(x_)
        xd  = nn.ReLU(True)(x_)
        
        x_ = self.conv4_1(xd)
        x_ = self.BN_conv4_1(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv4_2(xc)
        x_ = self.BN_conv4_2(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv4_3(xc)
        x_ = self.BN_conv4_3(x_)
        xc = nn.ReLU(True)(x_) 
        
        xd = xd +xc
        
        
        # deconv 5
        x_ = self.dconv5(xd)
        x_ = self.BN_dconv5(x_)
        xd  = nn.ReLU(True)(x_)
        
        x_ = self.conv5_1(xd)
        x_ = self.BN_conv5_1(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv5_2(xc)
        x_ = self.BN_conv5_2(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv5_3(xc)
        x_ = self.BN_conv5_3(x_)
        xc = nn.ReLU(True)(x_) 
        
        xd = xd +xc
        
        
        # deconv 6
        x_ = self.dconv6(xd)
        x_ = self.BN_dconv6(x_)
        xd  = nn.ReLU(True)(x_)
        
        x_ = self.conv6_1(xd)
        x_ = self.BN_conv6_1(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv6_2(xc)
        x_ = self.BN_conv6_2(x_)
        xc = nn.ReLU(True)(x_) 

        x_ = self.conv6_3(xc)
        x_ = self.BN_conv6_3(x_)
        xc = nn.ReLU(True)(x_) 
        
        xd = xd +xc
        
        
        # deconv 7
        x_3 = self.conv7(xd)
        xd  = nn.Sigmoid()(x_3)
        
        return xd