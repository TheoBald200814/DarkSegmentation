from torch import nn
from BasicUnit import dscdb
import Frameworks.level_three_stream as level_three_stream
import Tools.tensor_concate as t_c
import Tools.tensor_summation as t_s
import Tools.upsample as upsample

"""
    @description: level two stream framework, which consists of four DSCDB and some basic processing stream.
    @author: ZhouRenjie
    @Date: 2023/12/04
"""


class LevelTwoStream(nn.Module):
    def __init__(self, level_three_in_channels=3, level_three_out_channels=3, level_two_in_channels=6, level_two_out_channels=6):
        super(LevelTwoStream, self).__init__()
        self.dscdb1 = dscdb.DSCDB(level_two_in_channels, level_two_out_channels)
        self.dscdb2 = dscdb.DSCDB(level_two_in_channels, level_two_out_channels)
        self.dscdb3 = dscdb.DSCDB(level_two_in_channels, level_two_out_channels)
        self.dscdb4 = dscdb.DSCDB(level_two_in_channels, level_two_out_channels)
        self.sequential = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, 3),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(3)
        )
        self.level_three_stream = level_three_stream.LevelThreeStream(level_three_in_channels, level_three_out_channels)

    def forward(self, x):
        x = self.sequential(x)
        level_three_stream_result = self.level_three_stream(x)
        x = t_c.tensor_concate(x, level_three_stream_result)
        x = self.dscdb1(x)
        temp = self.dscdb2(x)
        x = t_s.tensor_summation(x, temp)
        temp = self.dscdb3(x)
        x = t_s.tensor_summation(x, temp)
        x = self.dscdb4(x)
        result = t_c.tensor_concate(x, level_three_stream_result)
        result = upsample.upsample(result, 2)

        return result




