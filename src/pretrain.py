"""Recreate the warm-start checkpoint when full retraining is requested.
No previous-stage weights are required. Completed robust models need no pretraining.
"""
from pathlib import Path
import random,time,json
import numpy as np
import torch
from common import ROOT,make_model,augment,logits_for,metrics,read_data,dump
from torch import nn
OPT=ROOT/'work'/'pretraining'
def train_run(config,seed,raw_train,train_y,val_x,val_y,normalization,epochs=50):
    tag=f'{config}_seed{seed}'; resultfile=OPT/'results'/f'{tag}.json'; ckptpath=OPT/'models'/f'{tag}.pt'
    if resultfile.exists() and ckptpath.exists(): return json.loads(resultfile.read_text(encoding='utf-8'))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=make_model(config).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00001)
    scaler=torch.amp.GradScaler('cuda'); ordergen=torch.Generator(device='cuda').manual_seed(seed+10000); auggen=torch.Generator(device='cuda').manual_seed(seed+20000)
    best_acc=-1.; best_loss=float('inf'); history=[]; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin=time.perf_counter()
    print('START',tag,flush=True)
    for epoch in range(1,epochs+1):
        model.train(); order=torch.randperm(len(raw_train),device='cuda',generator=ordergen); loss_sum=torch.zeros((),device='cuda'); correct=torch.zeros((),device='cuda'); started=time.perf_counter(); lr=optimizer.param_groups[0]['lr']
        for start in range(0,len(raw_train),256):
            ix=order[start:start+256]; image=raw_train[ix]
            if config=='deep_aug': image=augment(image,auggen)
            image=(image-normalization['mean'])/normalization['std']; optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(image); loss=nn.functional.cross_entropy(logits,train_y[ix])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); loss_sum+=loss.detach()*len(ix); correct+=(logits.argmax(1)==train_y[ix]).sum()
        validation=metrics(logits_for(model,val_x),val_y); scheduler.step(); torch.cuda.synchronize()
        row={'epoch':epoch,'lr':lr,'train_loss':loss_sum.item()/len(raw_train),'train_accuracy':correct.item()/len(raw_train),'val_accuracy':validation['accuracy'],'val_loss':validation['loss'],'seconds':time.perf_counter()-started}; history.append(row)
        if (validation['accuracy'],-validation['loss'])>(best_acc,-best_loss):
            best_acc=validation['accuracy']; best_loss=validation['loss']; best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'config':config,'seed':seed,'epoch':epoch,'normalization':normalization,'classes':CLASSES,'inference_precision':'autocast float16'},ckptpath)
        print(tag,f'epoch={epoch}/{epochs} train={row["train_loss"]:.4f} val={row["val_accuracy"]:.4f} val_loss={row["val_loss"]:.4f} lr={lr:.6f} sec={row["seconds"]:.2f}',flush=True)
    torch.cuda.synchronize(); seconds=time.perf_counter()-begin
    result={'config':config,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'validation_accuracy':best_acc,'validation_loss':best_loss,'history':history,'train_seconds':seconds,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'test_evaluated':False}
    dump(resultfile,result); print('DONE',tag,'validation',best_acc,'seconds',seconds,flush=True); del model,optimizer,scaler; torch.cuda.empty_cache(); return result

def ensure_warm_start(seed,normalization):
    path=OPT/'models'/f'deep_aug_seed{seed}.pt'
    if path.exists(): return path
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    for folder in ['models','results']: (OPT/folder).mkdir(parents=True,exist_ok=True)
    images,labels,_,_=read_data(); split=np.load(ROOT/'data'/'split_indices.npz'); ti,vi=split['train'],split['validation']
    raw=torch.from_numpy(images[ti]).to('cuda',torch.float32)/255
    targets=torch.from_numpy(labels[ti].astype(np.int64)).cuda()
    val=(torch.from_numpy(images[vi]).to('cuda',torch.float32)/255-normalization['mean'])/normalization['std']
    val_y=torch.from_numpy(labels[vi].astype(np.int64)).cuda()
    train_run('deep_aug',seed,raw,targets,val,val_y,normalization,epochs=50)
    return path
