"""One classifier for clean/noisy images; no condition-specific routing."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from train_robust import ROB,ROOT,DeepCNN,read_data,logits_for,CLASSES

def main():
    p=argparse.ArgumentParser(); p.add_argument('--image',type=Path); p.add_argument('--test-index',type=int,default=0); p.add_argument('--sigma',type=float,default=0.); p.add_argument('--noise-seed',type=int,default=12345); p.add_argument('--single-seed',type=int,choices=[42,2026,9122]); p.add_argument('--output',type=Path,default=ROB/'results'/'inference_example.json'); args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU fallback disabled')
    if args.sigma<0: raise ValueError('sigma must be nonnegative')
    if args.image:
        raw=np.array(Image.open(args.image).convert('L').resize((28,28)),dtype=np.float32)[None,None]/255; label=None
        if args.sigma: raw=np.clip(raw+np.random.default_rng(args.noise_seed).normal(0,args.sigma,raw.shape).astype(np.float32),0,1)
    else:
        if not 0<=args.test_index<10000: raise ValueError('test-index must be in [0,9999]')
        _,_,images,labels=read_data(); raw=images[args.test_index:args.test_index+1].astype(np.float32)/255; label=CLASSES[int(labels[args.test_index])]
        if args.sigma:
            noise=np.random.default_rng(args.noise_seed).normal(0,args.sigma,images.shape).astype(np.float32)[args.test_index:args.test_index+1]; raw=np.clip(raw+noise,0,1)
    config=json.loads((ROB/'results'/'selection_lock.json').read_text(encoding='utf-8'))['config']; seeds=[args.single_seed] if args.single_seed is not None else [42,2026,9122]; probabilities=[]
    for seed in seeds:
        ckpt=torch.load(ROB/'models'/f'{config}_seed{seed}.pt',weights_only=True); model=DeepCNN().cuda(); model.load_state_dict(ckpt['state_dict']); norm=ckpt['normalization']; x=(torch.from_numpy(raw).cuda()-norm['mean'])/norm['std']; probabilities.append(logits_for(model,x).softmax(1)); del model
    prob=torch.stack(probabilities).mean(0)[0].cpu().numpy(); order=np.argsort(-prob)[:3]; result={'device':torch.cuda.get_device_name(0),'config':config,'seeds':seeds,'added_sigma':args.sigma,'true_label':label,'predicted':CLASSES[int(order[0])],'top3':[{'class':CLASSES[int(k)],'score':float(prob[k])} for k in order],'note':'Same model and preprocessing for clean/noisy input. Scores are uncalibrated. External images must resemble Fashion-MNIST.'}; args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
