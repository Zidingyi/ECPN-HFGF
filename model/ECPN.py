import torch
import torch.nn as nn
import torch.nn.functional as F


class ECPN(nn.Module):
    def __init__(self, config):
        super(ECPN, self).__init__()
        self.alpha = config["alpha"]
        self.final_count = config["labels"][-1]
        """
        kernel size:
        channel 1: 5, 5, 15
        channel 2: 17, 5, 3
        channel 3: 15, 5, 5
        """
        self.conv1_1 = nn.Conv2d(1, 128, (5, 21), padding=(2, 0))
        self.conv2_1 = nn.Conv2d(1, 128, (17, 21), padding=(8, 0))
        self.conv3_1 = nn.Conv2d(1, 128, (15, 21), padding=(7, 0))

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        self.drop1_1 = nn.Dropout(0.1)
        self.drop1_2 = nn.Dropout(0.1)
        self.drop1_3 = nn.Dropout(0.1)

        self.conv1_2 = BasicBlock(128, 128, 5, 2)
        self.conv2_2 = BasicBlock(128, 128, 5, 2)
        self.conv3_2 = BasicBlock(128, 128, 5, 2)

        self.conv1_3 = BasicBlock(128, 128, 15, 7)
        self.conv2_3 = BasicBlock(128, 128, 3, 1)
        self.conv3_3 = BasicBlock(128, 128, 5, 2)

        self.conv = nn.Conv2d(128 * 3, 128 * 3, (1, 1))
        self.bn = nn.BatchNorm2d(128 * 3)

        self.dp = DP(9 + 1, 128 * 3)

        self.msml = MSML(config["labels"])

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=128 * 3, out_features=512)
        self.bn_fc1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.bn_fc2 = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=self.final_count)
        self.bn_fc3 = nn.BatchNorm1d(num_features=self.final_count)
        self.out_act = nn.Sigmoid() if self.final_count != 2 else nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # DeepTFactor
        x1 = self.conv1_1(x)
        x2 = self.conv2_1(x)
        x3 = self.conv3_1(x)

        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x3 = self.bn3(x3)

        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x3 = self.relu(x3)

        x1 = self.drop1_1(x1)
        x2 = self.drop1_2(x2)
        x3 = self.drop1_3(x3)

        x1 = self.conv1_2(x1)
        x2 = self.conv2_2(x2)
        x3 = self.conv3_2(x3)

        x1 = self.conv1_3(x1)
        x2 = self.conv2_3(x2)
        x3 = self.conv3_3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.relu(self.bn(self.conv(x)))

        # DP for global
        y_global = self.dp(x).view(-1, 128*3)

        y_global = self.relu(self.bn_fc1(self.fc1(y_global)))
        y_global = self.relu(self.bn_fc2(self.fc2(y_global)))
        y_global = self.out_act(self.bn_fc3(self.fc3(y_global)))

        # MSML for local
        y_local = self.msml(x)

        return {
            "global": y_global,
            "local": y_local,
            "scores": torch.cat([y_local[:, :-self.final_count], y_global * self.alpha + y_local[:, -self.final_count:] * (1-self.alpha)], dim=1),
            #"scores": y_global * self.alpha + y_local * (1-self.alpha)
        }


class MSML(nn.Module):
    def __init__(self, dims):
        super(MSML, self).__init__()

        self.conv1 = BasicBlock(128 * 3, 128 * 3, 5, 2)
        self.conv2 = BasicBlock(128 * 3, 128 * 3, 5, 2)
        self.conv3 = BasicBlock(128 * 3, 128 * 3, 5, 2)
        self.conv4 = BasicBlock(128 * 3, 128 * 3, 5, 2)

        self.pool = nn.MaxPool2d((1000, 1))

        self.hgm01 = nn.Linear(384, 256)
        self.hgm12 = HGM(256, 384)
        self.hgm23 = HGM(256 + 384, 384)
        self.hgm34 = HGM(256 + 384 + 384, 384)

        self.fc1_1 = nn.Linear(in_features=256, out_features=512)
        self.bn_fc1_1 = nn.BatchNorm1d(num_features=512)
        self.fc1_2 = nn.Linear(in_features=512, out_features=512)
        self.bn_fc1_2 = nn.BatchNorm1d(num_features=512)
        self.fc1_3 = nn.Linear(in_features=512, out_features=dims[0])
        self.bn_fc1_3 = nn.BatchNorm1d(num_features=dims[0])

        self.fc2_1 = nn.Linear(in_features=256 + 384, out_features=512)
        self.bn_fc2_1 = nn.BatchNorm1d(num_features=512)
        self.fc2_2 = nn.Linear(in_features=512, out_features=512)
        self.bn_fc2_2 = nn.BatchNorm1d(num_features=512)
        self.fc2_3 = nn.Linear(in_features=512, out_features=dims[1])
        self.bn_fc2_3 = nn.BatchNorm1d(num_features=dims[1])

        self.fc3_1 = nn.Linear(in_features=256 + 384 + 384, out_features=512)
        self.bn_fc3_1 = nn.BatchNorm1d(num_features=512)
        self.fc3_2 = nn.Linear(in_features=512, out_features=512)
        self.bn_fc3_2 = nn.BatchNorm1d(num_features=512)
        self.fc3_3 = nn.Linear(in_features=512, out_features=dims[2])
        self.bn_fc3_3 = nn.BatchNorm1d(num_features=dims[2])

        self.fc4_1 = nn.Linear(in_features=256 + 384 + 384 + 384, out_features=512)
        self.bn_fc4_1 = nn.BatchNorm1d(num_features=512)
        self.fc4_2 = nn.Linear(in_features=512, out_features=512)
        self.bn_fc4_2 = nn.BatchNorm1d(num_features=512)
        self.fc4_3 = nn.Linear(in_features=512, out_features=dims[3])
        self.bn_fc4_3 = nn.BatchNorm1d(num_features=dims[3])

        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):  # bs, 384, 979, 1
        x1 = self.pool(self.conv1(x)).view(-1, 384)
        x2 = self.pool(self.conv2(x)).view(-1, 384)
        x3 = self.pool(self.conv3(x)).view(-1, 384)
        x4 = self.pool(self.conv4(x)).view(-1, 384)

        f1 = self.hgm01(x1)  # 256

        y1 = self.relu(self.bn_fc1_1(self.fc1_1(f1)))
        y1 = self.relu(self.bn_fc1_2(self.fc1_2(y1)))
        y1 = self.out_act(self.bn_fc1_3(self.fc1_3(y1)))

        f2 = self.hgm12(f1, x2)  # 256 + 384

        y2 = self.relu(self.bn_fc2_1(self.fc2_1(f2)))
        y2 = self.relu(self.bn_fc2_2(self.fc2_2(y2)))
        y2 = self.out_act(self.bn_fc2_3(self.fc2_3(y2)))

        f3 = self.hgm23(f2, x3)

        y3 = self.relu(self.bn_fc3_1(self.fc3_1(f3)))
        y3 = self.relu(self.bn_fc3_2(self.fc3_2(y3)))
        y3 = self.out_act(self.bn_fc3_3(self.fc3_3(y3)))

        f4 = self.hgm34(f3, x4)

        y4 = self.relu(self.bn_fc4_1(self.fc4_1(f4)))
        y4 = self.relu(self.bn_fc4_2(self.fc4_2(y4)))
        y4 = self.out_act(self.bn_fc4_3(self.fc4_3(y4)))

        return torch.cat([y1, y2, y3, y4], dim=1)


class HGM(nn.Module):
    def __init__(self, channel_p, channel_c):
        super(HGM, self).__init__()
        self.fc_p = nn.Linear(channel_p, channel_p)
        self.fc_c = nn.Linear(channel_c, channel_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_parent, f_child):
        """
        f_parent: 上一层特征，池化后 bs, channel
        f_child: 本层特征，池化后 bs, channel
        """
        f = torch.mul(f_parent, self.sigmoid(self.fc_p(f_parent) + self.fc_c(f_child)))
        f = torch.cat([f, f_child], dim=1)
        return f


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel, pad):
        super(BasicBlock, self).__init__()
        self.conv0 = nn.Conv2d(inplanes, planes, (1, 1))
        self.bn0 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(planes, planes, (kernel, 1), padding=(pad, 0))
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.conv2 = nn.Conv2d(planes, inplanes, (1, 1))
        self.bn2 = nn.BatchNorm2d(planes)

        self.drop = nn.Dropout(0.1)

        self.relu = nn.ReLU()

    def forward(self, res):
        x = self.conv0(res)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + res
        x = self.relu(x)

        return x


class DP(nn.Module):
    def __init__(self, conv_block_repeat_num, dim):
        super(DP, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(conv_block_repeat_num):
            self.convs += [self.conv_block(dim)]
        self.pool = nn.MaxPool2d((3, 1), stride=2)
        self.pad = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        for j, conv_block in enumerate(self.convs):
            if j == 0:
                fx = conv_block(x)
                x = x + fx
            else:
                x = self.pad(x)
                x = self.pool(x)
                fx = conv_block(x)
                x = x + fx
        return x

    def conv_block(self, dim):
        return nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(dim, dim, (3, 1)),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(dim, dim, (3, 1)),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.label = config['labels']

    def forward(self, y_pred, y_truth):
        y_local = y_pred.get("local")
        y_global = y_pred.get("global")

        local_loss = F.binary_cross_entropy(y_local, y_truth)
        global_loss = F.binary_cross_entropy(y_global, y_truth[:, -self.label[-1]:])

        return local_loss + global_loss


if __name__ == "__main__":
    CONFIG = {
        "labels": [7, 70, 212, 2234],
        "alpha": 0.5,
        "checkpoint_path": None,
    }
    x = torch.rand([32, 1, 1000, 21])
    model = ECPN(CONFIG)
    y = model(x)
    loss_func = Loss(CONFIG)
    loss_value = loss_func(y, y["scores"])
    print(y["scores"].size())
    print(loss_value)
