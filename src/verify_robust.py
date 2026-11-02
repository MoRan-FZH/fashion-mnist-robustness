"""Independently re-evaluate robust checkpoints and enforce the requested thresholds."""
import json,hashlib,numpy as np,torch
from train_robust import ROB,ROOT,DeepCNN,read_data,logits_for,make_noisy,dump

def recompute(logits,labels,expected):
    pred=logits.argmax(1); cm=np.bincount(labels.astype(np.int64)*10+pred,minlength=100).reshape(10,10); acc=float((pred==labels).mean()); f1=float((2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)).mean()); assert np.array_equal(cm,expected['confusion']); assert abs(acc-expected['accuracy'])<1e-12 and abs(f1-expected['macro_f1'])<1e-12; return acc

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROB/'results'/'protocol.json').read_text(encoding='utf-8')); search=json.loads((ROB/'results'/'search_results.json').read_text(encoding='utf-8')); lock=json.loads((ROB/'results'/'selection_lock.json').read_text(encoding='utf-8')); records=json.loads((ROB/'results'/'final_results.json').read_text(encoding='utf-8')); ensemble=json.loads((ROB/'results'/'ensemble.json').read_text(encoding='utf-8')); curve=json.loads((ROB/'results'/'noise_curve.json').read_text(encoding='utf-8'))
    assert len(search)==2 and all(not r['test_evaluated'] and r['seed']==42 for r in search); assert max(search,key=lambda r:tuple(r['selection_key']))['config']==lock['config']; assert len(records)==3 and sorted(r['seed'] for r in records)==[42,2026,9122]
    split_path=ROOT/'data'/'split_indices.npz'; assert hashlib.sha256(split_path.read_bytes()).hexdigest()==protocol['split_sha256']; split=np.load(split_path); assert not np.intersect1d(split['train'],split['validation']).size
    train_images,train_labels,images,labels=read_data(); assert np.array_equal(np.bincount(train_labels[split['train']]),np.full(10,5400)); assert np.array_equal(np.bincount(train_labels[split['validation']]),np.full(10,600)); norm=protocol['normalization']
    raw=torch.from_numpy(images).to('cuda',torch.float32)/255; raw_noisy=make_noisy(raw,.15,12345); valraw=torch.from_numpy(train_images[split['validation']]).to('cuda',torch.float32)/255; valnoisy=make_noisy(valraw,.15,71717); val_labels=train_labels[split['validation']]; checked=[]; all_probs={'clean':[],'noisy':[]}
    for r in records:
        tag=f"{r['config']}_seed{r['seed']}"; best=max(r['history'],key=lambda h:tuple(h['selection_key'])); assert r['best_epoch']==best['epoch'] and len(r['history'])==25; ckpt=torch.load(ROB/'models'/f'{tag}.pt',weights_only=True); assert ckpt['epoch']==r['best_epoch']; model=DeepCNN().cuda(); model.load_state_dict(ckpt['state_dict'])
        for rawx,field in [(valraw,'validation'),(valnoisy,'noisy_validation')]: recompute(logits_for(model,(rawx-norm['mean'])/norm['std']).cpu().numpy(),val_labels,r[field])
        scores={}
        for condition,rawx,field in [('clean',raw,'test'),('noisy',raw_noisy,'noisy_test')]:
            logits=logits_for(model,(rawx-norm['mean'])/norm['std']); saved=np.load(ROB/'results'/f'{tag}_{condition}_predictions.npz'); assert np.array_equal(saved['labels'],labels); assert np.array_equal(saved['predictions'],logits.argmax(1).cpu().numpy()); scores[condition]=recompute(logits.cpu().numpy(),labels,r[field]); all_probs[condition].append(logits.softmax(1))
        assert scores['clean']>=protocol['target_clean']; assert scores['noisy']>protocol['target_noisy']; checked.append({'run':tag,'clean_accuracy':scores['clean'],'noisy_accuracy':scores['noisy'],'passes_both':True}); print('PASS',tag,scores,flush=True); del model; torch.cuda.empty_cache()
    ens_scores={}
    for condition,field in [('clean','test'),('noisy','noisy_test')]:
        probability=torch.stack(all_probs[condition]).mean(0).cpu().numpy(); saved=np.load(ROB/'results'/f'ensemble_{condition}_predictions.npz'); assert np.allclose(probability,saved['probabilities'],atol=1e-6,rtol=1e-5); ens_scores[condition]=recompute(probability,labels,ensemble[field])
    assert ens_scores['clean']>=protocol['target_clean'] and ens_scores['noisy']>protocol['target_noisy']
    base=json.loads((ROB/'results'/'baseline_reference.json').read_text(encoding='utf-8'))['runs']
    for sigma,field in [(0.,'test'),(.15,'noisy_test')]:
        row=next(x for x in curve if x['sigma']==sigma); assert np.allclose(row['baseline_cnn'],[next(r for r in base if r['seed']==s)[field]['accuracy'] for s in [42,2026,9122]],atol=1e-12)
    dump(ROB/'results'/'verification.json',{'passed':True,'device':torch.cuda.get_device_name(0),'split_unchanged':True,'selection_uses_both_validation_conditions':True,'same_classifier_both_conditions':True,'noise_matches_original_protocol':True,'each_seed_passes_both':True,'runs':checked,'ensemble':ens_scores,'thresholds':{'clean_minimum':protocol['target_clean'],'noisy_strict_minimum':protocol['target_noisy']}}); print('ENSEMBLE VERIFIED',ens_scores,flush=True)
if __name__=='__main__': main()
