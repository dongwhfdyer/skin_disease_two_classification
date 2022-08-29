import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingLR
import itertools

initial_lr = 0.1


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass


net_1 = model()

optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
scheduler_1 = ExponentialLR(optimizer_1, gamma=0.1)
scheduler_2 = MultiStepLR(optimizer_1, milestones=[10, 20, 30], gamma=0.1)
scheduler_3 = StepLR(optimizer_1, step_size=10, gamma=0.1)
scheduler_4 = LambdaLR(optimizer_1, lr_lambda=lambda epoch: initial_lr * (0.1 ** epoch))
scheduler_5 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.1, patience=10, verbose=True)
scheduler_6 = CosineAnnealingLR(optimizer_1, T_max=10, eta_min=0.001)



print("初始化的学习率：", optimizer_1.defaults['lr'])

for epoch in range(1, 300):
    optimizer_1.zero_grad()
    optimizer_1.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    scheduler_6.step()
