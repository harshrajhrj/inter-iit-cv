import torch
import torch.nn as nn

# defining the Squeeze-and-Excitation (SE) block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #global average pooling to get channel-wise statistics
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), #reduce channel dimension
            nn.ReLU(inplace=True), # non-linearity
            nn.Linear(channel // reduction, channel, bias=False), #restore channel dimension
            nn.Sigmoid() #output weights between 0 and 1 for each channel
        )

    def forward(self, x):
        b, c, _, _ = x.size() #get batch size and channel count
        y = self.avg_pool(x).view(b, c) #apply global average pooling and flatten
        y = self.fc(y).view(b, c, 1, 1) #pass through fully connected layers and reshape for broadcasting
        return x * y.expand_as(x) #scale input by channel-wise weights

# defining the basic residual block with SE
class BasicBlock(nn.Module):
    expansion = 1 #expansion factor for output channels
    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) #first 3x3 conv
        self.bn1 = nn.BatchNorm2d(planes) #batch normalization after first conv
        self.relu = nn.ReLU(inplace=True) #ReLU activation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) #second 3x3 conv
        self.bn2 = nn.BatchNorm2d(planes) #batch normalization after second conv
        self.se = SEBlock(planes, reduction) #squeeze-and-Excitation block
        self.downsample = None #default: no downsampling
        if stride != 1 or in_planes != planes:
            #downsampling input if dimensions change
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x #save input for the skip connection
        out = self.relu(self.bn1(self.conv1(x))) #first conv, BN, and ReLU
        out = self.bn2(self.conv2(out)) #second conv and BN
        out = self.se(out) #apply SE block
        if self.downsample is not None:
            identity = self.downsample(x) #downsample input if needed
        out += identity #add skip connection
        out = self.relu(out) #final ReLU
        return out

# defining the overall SEResNet architecture
class SEResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SEResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # created four sequential layers of blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #global average pooling before the classifier
        self.fc = nn.Linear(512 * block.expansion, num_classes) #final fully connected layer for classification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) #first block may have stride, rest have stride 1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride)) #add block to layer
            self.in_planes = planes * block.expansion #updating in_planes for next block
        return nn.Sequential(*layers) #retured sequential module

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) #flattened for the classifier
        x = self.fc(x) #final classification layer
        return x

def seresnet18(num_classes=10):
    return SEResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)