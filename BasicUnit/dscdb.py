from torch import nn

"""
    @description: DSCDB单元包含6层网络。其中2层卷积、2层实例归一化、2层LeakyReLU激活函数。
    @author: ZhouRenjie
    @Date: 2023/12/03
"""


class DSCDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSCDB, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        n = self.sequential(x)
        return n
