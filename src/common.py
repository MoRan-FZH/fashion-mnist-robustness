"""Self-contained data, network, augmentation and evaluation utilities."""
from pathlib import Path
import os,json,gzip,struct
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn

CLASSES = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

RESOURCES = [('train-images-idx3-ubyte.gz','8d4fb7e6c68d591d4c3dfef9ec88bf0d'),('train-labels-idx1-ubyte.gz','25c81989df183df01b3e8a0aad5dffbe'),('t10k-images-idx3-ubyte.gz','bef4ecab320f06d8554ea6380940ec79'),('t10k-labels-idx1-ubyte.gz','bb300cfdad3c16e7a12a480ee83cd310')]

def read_data():
    arrays=[]
    for name,_ in RESOURCES:
        b=gzip.decompress((ROOT/'data'/name).read_bytes())
        if 'images' in name:
            magic,n,h,w=struct.unpack('>IIII',b[:16]); assert magic==2051
            a=np.frombuffer(b,dtype=np.uint8,offset=16).copy().reshape(n,1,h,w)
        else:
            magic,n=struct.unpack('>II',b[:8]); assert magic==2049
            a=np.frombuffer(b,dtype=np.uint8,offset=8).copy()
        assert len(a)==n
        arrays.append(a)
    return arrays

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__(); blocks=[]; channels=1
        for width,drop in [(32,.05),(64,.10),(128,.15)]:
            blocks += [nn.Conv2d(channels,width,3,padding=1,bias=False),nn.BatchNorm2d(width),nn.ReLU(),nn.Conv2d(width,width,3,padding=1,bias=False),nn.BatchNorm2d(width),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout2d(drop)]; channels=width
        self.features=nn.Sequential(*blocks)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128*3*3,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,10))
    def forward(self,x): return self.head(self.features(x))

def augment(x,generator):
    n=len(x); x=torch.where(torch.rand(n,1,1,1,device='cuda',generator=generator)<.5,x.flip(-1),x)
    padded=nn.functional.pad(x,(2,2,2,2),value=0)
    shift=torch.randint(0,5,(n,2),device='cuda',generator=generator)
    row=torch.arange(28,device='cuda')[None,:,None]+shift[:,0,None,None]
    col=torch.arange(28,device='cuda')[None,None,:]+shift[:,1,None,None]
    return padded[torch.arange(n,device='cuda')[:,None,None],0,row,col].unsqueeze(1)

@torch.inference_mode()
def logits_for(model,x,batch=256):
    model.eval(); parts=[]
    for start in range(0,len(x),batch):
        with torch.autocast('cuda',dtype=torch.float16): z=model(x[start:start+batch])
        parts.append(z.float())
    return torch.cat(parts)

def metrics(logits,y,probabilities=False):
    pred=logits.argmax(1); cm=torch.bincount(y*10+pred,minlength=100).reshape(10,10).cpu().numpy(); tp=np.diag(cm)
    precision=tp/np.maximum(cm.sum(0),1); recall=tp/np.maximum(cm.sum(1),1); f1=2*tp/np.maximum(cm.sum(0)+cm.sum(1),1)
    loss=nn.functional.nll_loss(logits.clamp_min(1e-12).log(),y).item() if probabilities else nn.functional.cross_entropy(logits,y).item()
    return {'loss':loss,'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean()),'confusion':cm.tolist(),'precision':precision.tolist(),'recall':recall.tolist(),'f1':f1.tolist()}

def make_model(config="deep_aug"):
    return DeepCNN()
def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
