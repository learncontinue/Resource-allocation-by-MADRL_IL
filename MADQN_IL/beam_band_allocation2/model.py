# -*- coding:utf-8 -*-
"""
作者：Admin
日期：2022年04月15日
"""
import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 720)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(720, 360)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(360, 128)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(128, n_actions)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        output = self.fc4(x)
        return output

class choice_Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(choice_Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 720)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(720, 1024)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(1024, 1440)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(1440, n_actions)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        output = self.fc4(x)
        return output

## 以上是DQN，下面是DDQN
class ActorNet(nn.Module):
    def __init__(self, state_dim, a_dim):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 100)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(100, 50)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(50, 30)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(30, a_dim)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, x):
        x = f.relu((self.fc1(x)))
        x = f.relu((self.fc2(x)))
        x = f.relu((self.fc3(x)))
        x = f.sigmoid(self.fc4(x))
        return x


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(s_dim+a_dim, 100)
        nn.init.normal_(self.fc1.weight, 0., 0.1)
        nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(100, 50)
        nn.init.normal_(self.fc2.weight, 0., 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(50, 30)
        nn.init.normal_(self.fc3.weight, 0., 0.1)
        nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(30, 1)
        nn.init.normal_(self.fc4.weight, 0., 0.1)
        nn.init.constant_(self.fc4.bias, 0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = f.relu((self.fc1(x)))
        x = f.relu((self.fc2(x)))
        x = f.relu((self.fc3(x)))
        x = self.fc4(x)
        return x
