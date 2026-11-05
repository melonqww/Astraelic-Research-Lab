"""Bijection from a uniformly drawn valid-window index to (file, offset).

Documents stay separate. A file of length L has L-context valid starts for
context inputs plus one shifted target. No validation-derived weighting.
"""
from bisect import bisect_right

class WindowIndex:
    def __init__(self, lengths, context):
        if type(context) is not int or context<=0 or not lengths:
            raise ValueError('positive context and nonempty lengths required')
        self.counts=[];self.ends=[];total=0
        for length in lengths:
            if type(length) is not int or length<=context:
                raise ValueError('each document must fit inputs plus next-token target')
            count=length-context;total+=count;self.counts.append(count);self.ends.append(total)
        self.total=total

    def locate(self, draw):
        if type(draw) is not int or not 0<=draw<self.total:
            raise ValueError('draw outside valid windows')
        file=bisect_right(self.ends,draw)
        return file,draw-(self.ends[file-1] if file else 0)
