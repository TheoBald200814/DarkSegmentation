from torch import nn
import Frameworks.level_two_stream as level_two_stream
import BasicUnit.mdscdb as mdscdb
import Tools.tensor_concate as t_c
import Tools.tensor_summation as t_s


"""
    @description: level one stream framework, which consists of four MDSCDB and some basic processing stream.
    @author: ZhouRenjie
    @Date: 2023/12/04
"""


class LevelOneStream(nn.Module):
    def __init__(self, level_three_in_channels,
                 level_three_out_channels,
                 level_two_in_channels,
                 level_two_out_channels):
        super(LevelOneStream, self).__init__()
        self.mdscdb1 = mdscdb.MDSCDB(12, 12)
        self.mdscdb2 = mdscdb.MDSCDB(24, 24)
        self.mdscdb3 = mdscdb.MDSCDB(48, 48)
        self.mdscdb4 = mdscdb.MDSCDB(96, 96)
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.LeakyReLU()
        )
        self.level_two_stream = level_two_stream.LevelTwoStream(level_three_in_channels,
                                                                level_three_out_channels,
                                                                level_two_in_channels,
                                                                level_two_out_channels)
        self.post_processing = nn.Sequential(
            nn.Conv2d(204, 3, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sequential(x)
        x_backup = x
        level_two_stream_result = self.level_two_stream(x)
        x = t_c.tensor_concate(x, level_two_stream_result)
        temp = self.mdscdb1(x)
        x = t_s.tensor_summation(x, temp)
        temp = self.mdscdb2(x)
        x = t_s.tensor_summation(x, temp)
        temp = self.mdscdb3(x)
        x = t_s.tensor_summation(x, temp)
        x = self.mdscdb4(x)
        x = t_c.tensor_concate(x, level_two_stream_result)
        x = t_c.tensor_concate(x, x_backup)

        result = self.post_processing(x)
        return result


