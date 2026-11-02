"""Noise-aware fine-tuning, dual-condition validation, locked three-seed test.
Original data splits and official test corruption remain unchanged.
"""
from pathlib import Path
import os,sys,json,time,random,hashlib
ROB=Path(__file__).resolve().parents[1]; ROOT=ROB
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); os.environ.setdefault('TORCH_HOME',str(ROB/'work'/'torch'))

import numpy as np
import torch
from torch import nn
from common import read_data,CLASSES,DeepCNN,augment,logits_for,metrics
from pretrain import ensure_warm_start

def dump(path,data): Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def selection_key(clean,noisy):
    a,b=clean['accuracy'],noisy['accuracy']
    return (int(a>=.925),min(a,b),(a+b)/2,-(clean['loss']+noisy['loss'])/2)
def make_noisy(raw,sigma,seed):
    noise=torch.from_numpy(np.random.default_rng(seed).normal(0,sigma,raw.shape).astype(np.float32)).cuda()
    return (raw+noise).clamp(0,1)
def train_noise(raw,config,generator):
    n=len(raw); mask=torch.rand(n,1,1,1,device='cuda',generator=generator)<.5
    if config=='noise_fixed': sigma=torch.full((n,1,1,1),.15,device='cuda')
    else: sigma=.05+.15*torch.rand(n,1,1,1,device='cuda',generator=generator)
    noise=torch.randn(raw.shape,device='cuda',generator=generator)*sigma
    return torch.where(mask,(raw+noise).clamp(0,1),raw)

def train_run(config,seed,raw_train,train_y,val_x,noisy_val_x,val_y,norm,epochs=25):
    tag=f'{config}_seed{seed}'; rp=ROB/'results'/f'{tag}.json'; cp=ROB/'models'/f'{tag}.pt'
    if rp.exists() and cp.exists(): return json.loads(rp.read_text(encoding='utf-8'))
    initial=ensure_warm_start(seed,norm)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    initial_state=torch.load(initial,weights_only=True)
    model=DeepCNN().cuda(); model.load_state_dict(initial_state['state_dict']); optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00001); scaler=torch.amp.GradScaler('cuda')
    ordergen=torch.Generator(device='cuda').manual_seed(seed+110000); auggen=torch.Generator(device='cuda').manual_seed(seed+120000); noisegen=torch.Generator(device='cuda').manual_seed(seed+130000)
    key=(-1,-1,-1,-float('inf')); history=[]; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin=time.perf_counter(); print('START',tag,flush=True)
    for epoch in range(1,epochs+1):
        model.train(); order=torch.randperm(len(raw_train),device='cuda',generator=ordergen); loss_sum=torch.zeros((),device='cuda'); correct=torch.zeros((),device='cuda'); start=time.perf_counter(); lr=optimizer.param_groups[0]['lr']
        for start_index in range(0,len(raw_train),256):
            ix=order[start_index:start_index+256]; images=train_noise(augment(raw_train[ix],auggen),config,noisegen); images=(images-norm['mean'])/norm['std']; optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): logits=model(images); loss=nn.functional.cross_entropy(logits,train_y[ix])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); loss_sum+=loss.detach()*len(ix); correct+=(logits.argmax(1)==train_y[ix]).sum()
        clean=metrics(logits_for(model,val_x),val_y); noisy=metrics(logits_for(model,noisy_val_x),val_y); current=selection_key(clean,noisy); scheduler.step(); torch.cuda.synchronize()
        row={'epoch':epoch,'lr':lr,'train_loss':loss_sum.item()/len(raw_train),'train_accuracy':correct.item()/len(raw_train),'clean_val_accuracy':clean['accuracy'],'clean_val_loss':clean['loss'],'noisy_val_accuracy':noisy['accuracy'],'noisy_val_loss':noisy['loss'],'selection_key':list(current),'seconds':time.perf_counter()-start}; history.append(row)
        if current>key:
            key=current; best_epoch=epoch; best_clean=clean; best_noisy=noisy
            torch.save({'state_dict':model.state_dict(),'config':config,'seed':seed,'epoch':epoch,'normalization':norm,'classes':CLASSES,'inference_precision':'CUDA autocast float16','warm_start':str(initial.relative_to(ROOT))},cp)
        print(tag,f'epoch={epoch}/{epochs} clean_val={clean["accuracy"]:.4f} noisy_val={noisy["accuracy"]:.4f} loss={row["train_loss"]:.4f} sec={row["seconds"]:.2f}',flush=True)
    torch.cuda.synchronize(); seconds=time.perf_counter()-begin
    record={'config':config,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'validation':best_clean,'noisy_validation':best_noisy,'selection_key':list(key),'history':history,'finetune_seconds':seconds,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'warm_start_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'test_evaluated':False}; dump(rp,record); print('DONE',tag,'clean_val',best_clean['accuracy'],'noisy_val',best_noisy['accuracy'],flush=True); del model,optimizer,scaler; torch.cuda.empty_cache(); return record

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required; CPU fallback disabled.')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True)
    images,labels,_,_=read_data(); split_path=ROOT/'data'/'split_indices.npz'; split=np.load(split_path); ti,vi=split['train'],split['validation']; assert len(ti)==54000 and len(vi)==6000 and not np.intersect1d(ti,vi).size
    norm=json.loads((ROOT/'results'/'protocol.json').read_text(encoding='utf-8'))['normalization']; raw_train=torch.from_numpy(images[ti]).to('cuda',torch.float32)/255; train_y=torch.from_numpy(labels[ti].astype(np.int64)).cuda(); raw_val=torch.from_numpy(images[vi]).to('cuda',torch.float32)/255
    val_x=(raw_val-norm['mean'])/norm['std']; noisy_val_x=(make_noisy(raw_val,.15,71717)-norm['mean'])/norm['std']; val_y=torch.from_numpy(labels[vi].astype(np.int64)).cuda(); del raw_val
    reference=json.loads((ROB/'results'/'baseline_reference.json').read_text(encoding='utf-8')); baselines=reference['runs']; reference_curve={float(row['sigma']):dict(zip([42,2026,9122],row['baseline_cnn'])) for row in reference['curve']}; target_clean=float(np.mean([r['test']['accuracy'] for r in baselines])); target_noisy=float(np.mean([r['noisy_test']['accuracy'] for r in baselines]))
    protocol={'student_id':'2026439122','name':'范哲豪','started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'search_configs':['noise_fixed','noise_range'],'search_seed':42,'final_seeds':[42,2026,9122],'finetune_epochs':25,'pretraining_epochs':50,'batch_size':256,'optimizer':'AdamW','lr':.0003,'weight_decay':.0001,'scheduler':'CosineAnnealingLR T_max=25 eta_min=1e-5','normalization':norm,'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'train_size':54000,'validation_size':6000,'test_size':10000,'noise_probability':.5,'fixed_sigma':.15,'range_sigma':[.05,.20],'validation_noise_seed':71717,'test_noise_seed':12345,'primary_sigma':.15,'test_sigmas':[0.,.05,.10,.15,.20,.25],'selection':'Prefer clean validation accuracy >=0.925, then maximize min(clean,noisy) validation accuracy; then mean accuracy; then minimize mean CE. Earliest exact tie.','precision':'CUDA autocast float16 + GradScaler; FP32 parameters','warm_start':'Previous deep_aug checkpoints of matching seeds; no new test-driven warm-start selection','geometry':'Training only: horizontal flip p=0.5 and integer translations dx,dy in [-2,2]','ensemble':'Fixed equal mean softmax across seeds 42,2026,9122, no TTA, no clean/noisy oracle','target_clean':target_clean,'target_noisy':target_noisy,'test_policy':'Validation-only current search; lock before official test. Official test was viewed in previous stages, so this is iterative benchmark development, not a pristine holdout.'}
    pp=ROB/'results'/'protocol.json'
    if pp.exists():
        prior=json.loads(pp.read_text(encoding='utf-8')); assert all(prior[k]==protocol[k] for k in ['search_configs','final_seeds','finetune_epochs','noise_probability','fixed_sigma','range_sigma','split_sha256'])
    else: dump(pp,protocol)
    search=[train_run(c,42,raw_train,train_y,val_x,noisy_val_x,val_y,norm) for c in protocol['search_configs']]; dump(ROB/'results'/'search_results.json',search); winner=max(search,key=lambda r:tuple(r['selection_key']))['config']; lock={'config':winner,'locked_at':time.strftime('%Y-%m-%d %H:%M:%S'),'selection':'dual-condition validation only','candidate_selection_keys':{r['config']:r['selection_key'] for r in search},'test_used_for_current_search':False}; lockpath=ROB/'results'/'selection_lock.json'
    if lockpath.exists(): assert json.loads(lockpath.read_text(encoding='utf-8'))['config']==winner
    else: dump(lockpath,lock)
    print('CONFIGURATION LOCKED',winner,flush=True); selected=[next(r for r in search if r['config']==winner)]
    for seed in [2026,9122]: selected.append(train_run(winner,seed,raw_train,train_y,val_x,noisy_val_x,val_y,norm))
    del raw_train,train_y,val_x,noisy_val_x,val_y; torch.cuda.empty_cache()
    _,_,images,labels=read_data(); raw=torch.from_numpy(images).to('cuda',torch.float32)/255; test_y=torch.from_numpy(labels.astype(np.int64)).cuda(); curves=[]; all_selected={r['seed']:r for r in selected}
    ensemble_metrics={}; ensemble_probs={}
    for sigma in protocol['test_sigmas']:
        raw_condition=raw if sigma==0 else make_noisy(raw,sigma,12345); x=(raw_condition-norm['mean'])/norm['std']; base_scores=[]; selected_scores=[]; probabilities=[]
        for seed in [42,2026,9122]:
            base_scores.append({'accuracy':float(reference_curve[sigma][seed])})
            cp=ROB/'models'/f'{winner}_seed{seed}.pt'; ckpt=torch.load(cp,weights_only=True); model=DeepCNN().cuda(); model.load_state_dict(ckpt['state_dict']); logits=logits_for(model,x); result=metrics(logits,test_y); selected_scores.append(result); probabilities.append(logits.softmax(1))
            if sigma in [0.,.15]:
                suffix='clean' if sigma==0 else 'noisy'; np.savez_compressed(ROB/'results'/f'{winner}_seed{seed}_{suffix}_predictions.npz',logits=logits.cpu().numpy(),labels=labels,predictions=logits.argmax(1).cpu().numpy()); all_selected[seed]['test' if sigma==0 else 'noisy_test']=result
            del model; torch.cuda.empty_cache()
        average=torch.stack(probabilities).mean(0); ens=metrics(average,test_y,probabilities=True); ensemble_metrics[str(sigma)]=ens
        curves.append({'sigma':sigma,'baseline_cnn':[r['accuracy'] for r in base_scores],'robust_single':[r['accuracy'] for r in selected_scores],'ensemble_accuracy':ens['accuracy'],'ensemble_macro_f1':ens['macro_f1']})
        if sigma in [0.,.15]:
            suffix='clean' if sigma==0 else 'noisy'; np.savez_compressed(ROB/'results'/f'ensemble_{suffix}_predictions.npz',probabilities=average.cpu().numpy(),labels=labels,predictions=average.argmax(1).cpu().numpy())
        print('TEST sigma',sigma,'baseline',float(np.mean(curves[-1]['baseline_cnn'])),'robust',float(np.mean(curves[-1]['robust_single'])),'ensemble',ens['accuracy'],flush=True)
        del x,probabilities,average
    final=list(all_selected.values())
    for r in final: r['test_evaluated']=True; dump(ROB/'results'/f'{winner}_seed{r["seed"]}.json',r)
    ensemble={'config':winner,'seeds':[42,2026,9122],'method':'equal mean softmax, no TTA or condition oracle','test':ensemble_metrics['0.0'],'noisy_test':ensemble_metrics['0.15'],'parameters_total':sum(r['parameters'] for r in final)}
    for r in final:
        r['acceptance']={'clean_not_below_original':r['test']['accuracy']>=target_clean,'noisy_above_original_noisy':r['noisy_test']['accuracy']>target_noisy,'noisy_above_original_clean':r['noisy_test']['accuracy']>target_clean}
    acceptance={'target_clean':target_clean,'target_noisy':target_noisy,'single_mean_clean':float(np.mean([r['test']['accuracy'] for r in final])),'single_mean_noisy':float(np.mean([r['noisy_test']['accuracy'] for r in final])),'ensemble_clean':ensemble['test']['accuracy'],'ensemble_noisy':ensemble['noisy_test']['accuracy'],'all_three_seeds_pass':all(r['acceptance']['clean_not_below_original'] and r['acceptance']['noisy_above_original_noisy'] for r in final),'ensemble_pass':ensemble['test']['accuracy']>=target_clean and ensemble['noisy_test']['accuracy']>target_noisy,'all_three_noisy_above_original_clean':all(r['acceptance']['noisy_above_original_clean'] for r in final)}
    dump(ROB/'results'/'final_results.json',final); dump(ROB/'results'/'ensemble.json',ensemble); dump(ROB/'results'/'noise_curve.json',curves); dump(ROB/'results'/'acceptance.json',acceptance)
    print('COMPLETE robustness',json.dumps(acceptance),flush=True)
if __name__=='__main__': main()
