import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction,PointNetSetAbstraction_PointMax


class get_model(nn.Module):
    def __init__(self,in_channel,num_class):
        super(get_model, self).__init__()
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=9, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, point,feature,label=None):
        B, _, _ = point.shape
        
        xyz=point.permute(0,2,1).contiguous()
        norm=feature.permute(0,2,1).contiguous()
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        if self.training:
            return x,0
        else:
            return x


class get_model_pointmax(nn.Module):
    def __init__(self,in_channel,num_class):
        super(get_model_pointmax, self).__init__()
        
        self.sa1 = PointNetSetAbstraction_PointMax(npoint=512, radius=0.2, nsample=32, in_channel=9, mlp=[64, 64, 128], group_all=False,pointmax=False)
        self.sa2 = PointNetSetAbstraction_PointMax(npoint=128, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False,pointmax=True)
        self.sa3 = PointNetSetAbstraction_PointMax(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True,pointmax=False)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, point,feature,label=None):
        B, _, _ = point.shape
        
        xyz=point.permute(0,2,1).contiguous()
        norm=feature.permute(0,2,1).contiguous()
        all_loss=0
        l1_xyz, l1_points,loss = self.sa1(xyz, norm,label)
        all_loss=all_loss+loss
        l2_xyz, l2_points,loss = self.sa2(l1_xyz, l1_points,label)
        all_loss=all_loss+loss
        l3_xyz, l3_points,loss= self.sa3(l2_xyz, l2_points,label)
        all_loss=all_loss+loss
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        if self.training:
            return x,all_loss
        else:
            return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


def make(cfg):
    return get_model(cfg.model.in_channel, cfg.model.out_channel)
def make_pointmax(cfg):
    return get_model_pointmax(cfg.model.in_channel, cfg.model.out_channel)