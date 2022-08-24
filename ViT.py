import torch
from torch import nn 
from einops import rearrange,reduce,repeat

class Embedding(nn.Module):
    def __init__(self,height,width,in_channel,embed_size):
        super().__init__()
        self.height=height
        self.width=width
        self.in_channel=in_channel
        self.embed_size=embed_size
        self.patch=nn.Sequential(
            rearrange('b c (h 2) (w 2) -> b (h w) (2 2 c)'),
            nn.Linear(in_channel*4,embed_size)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, (height*width)/4 + 1, embed_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

    def forward(self,x):
        x=self.patch(x)
        b,n,_=x.shape
        cls_token=repeat(self.cls_token,'1 1 d -> b 1 d',b=b)
        x=torch.cat((cls_token,x),dim=1)
        x+=self.pos_embedding()


