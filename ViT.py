import torch
from torch import nn 
import torch.nn.functional as F
from einops import rearrange,reduce,repeat

class Embedding(nn.Module):
    def __init__(self,height,width,in_channel,embed_size):
        super(Embedding,self).__init__()
        self.height=height
        self.width=width
        self.in_channel=in_channel
        self.embed_size=embed_size
        self.projection =nn.Conv2d(in_channel, embed_size, kernel_size=2, stride=2)
        self.pos_embedding = nn.Parameter(torch.randn(1, (height*width)//4 + 1, embed_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

    def forward(self,x):
        b,c,h,w=x.shape
        x=self.projection(x)
        x=rearrange(x,'b e (h) (w) -> b (h w) e')
        b,n,_=x.shape
        cls_token=repeat(self.cls_token,'1 1 d -> b 1 d',b=b)
        x=torch.cat((cls_token,x),dim=1)
        x+=self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size,num_heads=8,dropout=0):
        super(MultiHeadAttention,self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self,x):
        queries=rearrange(self.queries(x),"b n (h d) -> b h n d", h=self.num_heads)
        keys=rearrange(self.keys(x),"b n (h d) -> b h n d", h=self.num_heads)
        values=rearrange(self.values(x),"b n (h d) -> b h n d", h=self.num_heads)
        energy=torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        att = F.softmax(energy, dim=-1) /(self.emb_size**0.5)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out+=x
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion = 4, drop_p= 0):
        super(FeedForwardBlock,self).__init__()
        self.block=nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size))

    def forward(self,x):
        out=self.block(x)
        out+=x
        return out



class TransformerEncoderBlock(nn.Module):
    def __init__(self,emb_size=768,drop_p=0,forward_expansion=4,forward_drop_p=0):
        super(TransformerEncoderBlock,self).__init__()
        self.block=nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size),
                nn.Dropout(drop_p),
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
    def forward(self,x):
        out=self.block(x)
        return out

class Classification(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super(Classification,self).__init__()
        self.block=nn.Sequential(
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
    def forward(self,x):
        out=reduce(x,'b n e -> b e', reduction='mean')
        out=self.block(x)
        return out

class ViT(nn.Module):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 1,
                n_classes: int = 1000):
        super(ViT,self).__init__()
        self.embedding=Embedding(224, 224, in_channels, emb_size)
        self.block=self.make_layer(depth)
        self.classification=Classification(emb_size, n_classes)

    def make_layer(self,depth):
        layers=[]
        for i in range(depth):
            layers.append(TransformerEncoderBlock())
        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.embedding(x)
        out=self.block(out)
        out=self.classification(out)
        return out
        