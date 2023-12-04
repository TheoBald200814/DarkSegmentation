from torch import nn
from BasicUnit import dscdb
import Tools.tensor_summation as t_sum
import Tools.upsample

"""
    @description: level three stream framework, which consists of four DSCDB and some basic processing stream.
    @author: ZhouRenjie
    @Date: 2023/12/03
"""


class LevelThreeStream(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(LevelThreeStream, self).__init__()

        self.dscdb1 = dscdb.DSCDB(in_channels, out_channels)
        self.dscdb2 = dscdb.DSCDB(in_channels, out_channels)
        self.dscdb3 = dscdb.DSCDB(in_channels, out_channels)
        self.dscdb4 = dscdb.DSCDB(in_channels, out_channels)
        self.sequential = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, 3),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(3)
        )

    def forward(self, x):

        # basic processing stream
        x = self.sequential(x)

        # first dscdb process
        x = self.dscdb1(x)

        # second dscdb process
        temp_x = self.dscdb2(x)

        # first skip connection
        x = t_sum.tensor_summation(temp_x, x)

        # third dscdb process
        temp_x = self.dscdb3(x)

        # second skip connection
        x = t_sum.tensor_summation(temp_x, x)

        # fourth dscdb process
        x = self.dscdb4(x)

        # upsample
        x = Tools.upsample.upsample(x, 2)

        return x

