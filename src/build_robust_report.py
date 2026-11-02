"""Build a six-page course report from verified robust-training measurements."""
from pathlib import Path
import os,sys,json,csv,gzip,re,statistics
ROB=Path(__file__).resolve().parents[1]; ROOT=ROB; OPT=ROB
os.environ['MPLCONFIGDIR']=str(ROB/'work'/'mplconfig'); sys.path.insert(0,str(ROOT/'work'/'packages'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from reportlab.lib.pagesizes import A4

runs=json.loads((ROB/'results'/'final_results.json').read_text(encoding='utf-8')); search=json.loads((ROB/'results'/'search_results.json').read_text(encoding='utf-8')); ensemble=json.loads((ROB/'results'/'ensemble.json').read_text(encoding='utf-8')); protocol=json.loads((ROB/'results'/'protocol.json').read_text(encoding='utf-8')); curve=json.loads((ROB/'results'/'noise_curve.json').read_text(encoding='utf-8')); verification=json.loads((ROB/'results'/'verification.json').read_text(encoding='utf-8')); acceptance=json.loads((ROB/'results'/'acceptance.json').read_text(encoding='utf-8')); lock=json.loads((ROB/'results'/'selection_lock.json').read_text(encoding='utf-8')); baseline=json.loads((ROB/'results'/'baseline_reference.json').read_text(encoding='utf-8'))['runs']; assert verification['passed'] and acceptance['all_three_seeds_pass'] and acceptance['ensemble_pass']
FIG=ROB/'report'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
def mean(rs,metric='accuracy',section='test'): return float(np.mean([r[section][metric] for r in rs]))
def stat(rs,metric='accuracy',section='test'):
    v=np.array([r[section][metric] for r in rs])*100; return f'{v.mean():.2f} ± {v.std(ddof=1):.2f}'
clean_mean=mean(runs); noise_mean=mean(runs,section='noisy_test'); old_clean=mean(baseline); old_noisy=mean(baseline,section='noisy_test'); ens_clean=ensemble['test']['accuracy']; ens_noisy=ensemble['noisy_test']['accuracy']
labels=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']; clabels=['T恤／上衣','裤子','套头衫','连衣裙','外套','凉鞋','衬衫','运动鞋','包','短靴']; names={'noise_fixed':'Fixed sigma 0.15','noise_range':'Uniform sigma 0.05-0.20'}; cnames={'noise_fixed':'固定强度混合训练','noise_range':'随机强度混合训练'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
def save(fig,name): fig.savefig(FIG/name,bbox_inches='tight',facecolor='white'); plt.close(fig)
train_x=np.frombuffer(gzip.decompress((ROOT/'data'/'train-images-idx3-ubyte.gz').read_bytes()),dtype=np.uint8,offset=16).reshape(-1,28,28)
train_y=np.frombuffer(gzip.decompress((ROOT/'data'/'train-labels-idx1-ubyte.gz').read_bytes()),dtype=np.uint8,offset=8)
fig,axes=plt.subplots(1,10,figsize=(11,1.4))
for k,ax in enumerate(axes): ax.imshow(train_x[np.flatnonzero(train_y==k)[0]],cmap='gray'); ax.set_title(labels[k],fontsize=8); ax.axis('off')
save(fig,'samples.png')
fig,axes=plt.subplots(1,2,figsize=(10,3.1))
for r,col in zip(search,['#3274A1','#E1812C']):
    epochs=[h['epoch'] for h in r['history']]
    for ax,key in zip(axes,['clean_val_accuracy','noisy_val_accuracy']): ax.plot(epochs,[h[key]*100 for h in r['history']],color=col,label=names[r['config']]); ax.set_xlabel('Fine-tuning epoch'); ax.set_ylabel('Validation accuracy (%)'); ax.grid(alpha=.2)
axes[0].set_title('Clean validation'); axes[1].set_title('Noisy validation: sigma=0.15'); axes[1].legend(fontsize=8); fig.tight_layout(); save(fig,'dual_validation.png')
fig,ax=plt.subplots(figsize=(9.5,3.6)); sigmas=[r['sigma'] for r in curve]
for key,name,col in [('baseline_cnn','Original CNN mean','#777777'),('robust_single','Robust single mean','#3274A1')]:
    y=np.array([r[key] for r in curve])*100; avg=y.mean(1); sd=y.std(1,ddof=1); ax.plot(sigmas,avg,marker='o',color=col,label=name); ax.fill_between(sigmas,avg-sd,avg+sd,color=col,alpha=.13)
ax.plot(sigmas,[r['ensemble_accuracy']*100 for r in curve],marker='s',color='#E1812C',label='Robust 3-model ensemble'); ax.axvline(.15,color='black',linestyle='--',linewidth=.8,label='Primary condition'); ax.set_xlabel('Gaussian noise standard deviation (pixel range 0-1)'); ax.set_ylabel('Test accuracy (%)'); ax.set_ylim(0,100); ax.set_xticks(sigmas); ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); save(fig,'noise_strength_curve.png')
raw=np.frombuffer(gzip.decompress((ROOT/'data'/'t10k-images-idx3-ubyte.gz').read_bytes()),dtype=np.uint8,offset=16).reshape(-1,28,28).astype(np.float32)/255; noisy=np.clip(raw+np.random.default_rng(12345).normal(0,.15,raw.shape).astype(np.float32),0,1)
pred=np.load(ROB/'results'/'ensemble_noisy_predictions.npz'); y=pred['labels']; yp=pred['predictions']; probability=pred['probabilities']; wrong=np.flatnonzero(y!=yp)[:8]
fig,axes=plt.subplots(2,4,figsize=(9,3.1))
for idx,ax in zip(wrong,axes.flat): ax.imshow(noisy[idx],cmap='gray',vmin=0,vmax=1); ax.set_title(f'#{idx} True: {labels[y[idx]]}\nPred: {labels[yp[idx]]} ({probability[idx,yp[idx]]:.2f})',fontsize=8); ax.axis('off')
fig.tight_layout(); save(fig,'noisy_errors.png')
cm=np.array(ensemble['noisy_test']['confusion']); fig,ax=plt.subplots(figsize=(8,5.5)); im=ax.imshow(cm/10,cmap='Blues',vmin=0,vmax=100); ax.set_xticks(range(10),labels,rotation=35,ha='right'); ax.set_yticks(range(10),labels); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
for i in range(10):
    for j in range(10):
        if cm[i,j]>0: ax.text(j,i,f'{cm[i,j]/10:.1f}',ha='center',va='center',fontsize=8,color='white' if cm[i,j]>550 else 'black')
fig.colorbar(im,ax=ax,label='Row percentage (%)',fraction=.04); fig.tight_layout(); save(fig,'noisy_confusion.png')
summary={'baseline_clean':old_clean,'baseline_noisy':old_noisy,'robust_single_clean_mean':clean_mean,'robust_single_clean_sample_std':float(np.std([r['test']['accuracy'] for r in runs],ddof=1)),'robust_single_noisy_mean':noise_mean,'robust_single_noisy_sample_std':float(np.std([r['noisy_test']['accuracy'] for r in runs],ddof=1)),'ensemble_clean':ens_clean,'ensemble_noisy':ens_noisy,'clean_gain_percentage_points':(clean_mean-old_clean)*100,'noisy_gain_percentage_points':(noise_mean-old_noisy)*100,'selected_config':lock['config'],'all_seeds_pass_both':True,'ensemble_passes_both':True}
(ROB/'results'/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
rows=[]
for name,records in [('original_cnn',baseline),('robust_single',runs)]: rows.append({'model':name,'clean_mean':mean(records),'noisy_mean':mean(records,section='noisy_test'),'clean_macro_f1':mean(records,'macro_f1'),'noisy_macro_f1':mean(records,'macro_f1','noisy_test')})
rows.append({'model':'robust_ensemble','clean_mean':ens_clean,'noisy_mean':ens_noisy,'clean_macro_f1':ensemble['test']['macro_f1'],'noisy_macro_f1':ensemble['noisy_test']['macro_f1']})
with (ROB/'results'/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with (ROB/'results'/'noise_curve.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['sigma','original_cnn_mean','robust_single_mean','robust_ensemble'])
    for r in curve:w.writerow([r['sigma'],float(np.mean(r['baseline_cnn'])),float(np.mean(r['robust_single'])),r['ensemble_accuracy']])
for field,filename in [('test','per_class_clean.csv'),('noisy_test','per_class_noisy.csv')]:
    with (ROB/'results'/filename).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['class','precision','recall','f1','support'])
        for k in range(10): w.writerow([labels[k],ensemble[field]['precision'][k],ensemble[field]['recall'][k],ensemble[field]['f1'][k],1000])
pdfmetrics.registerFont(TTFont('CN',r'C:\Windows\Fonts\simsun.ttc',subfontIndex=0)); pdfmetrics.registerFont(TTFont('CNBold',r'C:\Windows\Fonts\simhei.ttf')); pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNBold',italic='CN',boldItalic='CNBold')
styles={'body':ParagraphStyle('body',fontName='CN',fontSize=10.5,leading=17,spaceAfter=7,wordWrap='CJK',alignment=TA_JUSTIFY),'h':ParagraphStyle('heading',fontName='CNBold',fontSize=13,leading=20,spaceBefore=8,spaceAfter=7,wordWrap='CJK'),'small':ParagraphStyle('small',fontName='CN',fontSize=8.8,leading=12.5,spaceAfter=6,wordWrap='CJK'),'caption':ParagraphStyle('caption',fontName='CN',fontSize=8.5,leading=12,spaceAfter=7,wordWrap='CJK',alignment=TA_CENTER),'title':ParagraphStyle('title',fontName='CNBold',fontSize=20,leading=29,spaceAfter=12,wordWrap='CJK'),'cell':ParagraphStyle('cell',fontName='CN',fontSize=8.5,leading=12,wordWrap='CJK'),'headcell':ParagraphStyle('headcell',fontName='CNBold',fontSize=8.5,leading=12,textColor=colors.white,wordWrap='CJK')}
story=[]; md=[]
def P(text,style='body'): story.append(Paragraph(text,styles[style])); md.append(re.sub('<[^>]+>','',text))
def H(text): P(text,'h')
def T(rows,widths):
    table=Table([[Paragraph(str(x),styles['headcell' if i==0 else 'cell']) for x in row] for i,row in enumerate(rows)],colWidths=widths,hAlign='CENTER'); table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#233E57')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F8')]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])); story.append(table); story.append(Spacer(1,7)); lines=['| '+' | '.join(map(str,row))+' |' for row in rows]; lines.insert(1,'| '+' | '.join(['---']*len(rows[0]))+' |'); md.append('\n'.join(lines))
def I(path,w,h,caption):
    image=Image(str(path)); ratio=min(w/image.imageWidth,h/image.imageHeight); image.drawWidth=image.imageWidth*ratio; image.drawHeight=image.imageHeight*ratio; story.append(image); P(caption,'caption'); md.append(f'![{caption}]({os.path.relpath(path,OPT/"report").replace(chr(92),"/")})')
def page(): story.append(PageBreak()); md.append('\n---\n')


# PAGE 1
P('服饰图像分类的抗噪优化<br/>与双条件验证','title')
P('深度学习课程设计技术报告　抗噪优化版　2026 年秋季','small')
P('<b>姓名：</b>范哲豪　　<b>学号：</b>2026439122　　<b>实验日期：</b>2026 年 10 月 31 日')
P('<b>代码与复现：</b>完整代码、数据来源、模型及结果随附；尚未上传 GitHub，无公开仓库链接。','small')
H('摘要')
P(f'针对服饰图像分类中“干净图片准确率提高、加噪后性能反而下降”的问题，在保留原数据划分的基础上开展噪声感知微调。训练时独立以 0.5 概率为样本加噪，并通过干净与噪声验证集联合选择检查点。配置锁定后，使用三个随机种子复验。优化单模型的干净准确率为 {stat(runs)}%，固定高斯噪声准确率为 {stat(runs,section="noisy_test")}%；三模型等权集成分别为 {ens_clean*100:.2f}% 和 {ens_noisy*100:.2f}%。三个单模型及集成均达到干净成绩不低于原 CNN、噪声成绩高于原 CNN 的双条件目标，所有预测通过 GPU 检查点重载核验。')
P('<b>关键词：</b>Fashion-MNIST；噪声增强；混合训练；双条件验证；鲁棒性','small')
H('一 研究问题与验收标准')
P(f'原 CNN 在干净图像上平均为 {old_clean*100:.2f}%，在固定噪声下为 {old_noisy*100:.2f}%。上一版提高了干净准确率，但加噪表现下降，暴露出训练分布与测试扰动不匹配的问题。本轮目标是保留干净识别能力，同时提高对该已定义噪声的适应能力，而不是只追求一个条件下的最高值。')
P(f'验收条件明确为：同一官方测试集上，干净准确率不低于 {old_clean*100:.2f}%，噪声准确率严格高于 {old_noisy*100:.2f}%。噪声保持与原实验完全相同的强度、裁剪方式和种子，不降低干扰强度来获得表面改进。')
I(FIG/'samples.png',480,82,'图 1　Fashion-MNIST 十类服饰样本　输入为居中 28×28 灰度图')
P('研究范围为单件服饰图像分类。双条件达标只针对本文规定的数据与加性高斯噪声，不构成对所有拍摄环境或噪声类型的保证。','small')
page()
# PAGE 2
H('二 数据与噪声感知训练设计')
P('采用官方 Fashion-MNIST [1]：原训练集 60,000 张、测试集 10,000 张。沿用原始分层索引，训练 54,000 张、验证 6,000 张，每类分别为 5,400 和 600 张。训练与验证无交集。数据文件的官方 MD5、来源和 SHA-256 清单均保留，划分文件另记录 SHA-256，确保与之前实验一致。')
P(f'像素先除以 255，随后使用训练子集均值 {protocol["normalization"]["mean"]:.6f} 和标准差 {protocol["normalization"]["std"]:.6f} 标准化。噪声在标准化之前加入，并裁剪到 [0,1]。没有使用外部数据、修改标签或把验证及测试图片加入训练。')
T([['设计因素','具体设置'],['网络与起点','沿用六层卷积网络，通道块为 32、64、128；BatchNorm、池化、Dropout 与 1152→256→10 分类头保持不变；参数 584,618'],['初始化','加载上一阶段 50 轮训练预算内按验证集选出的同种子检查点，再微调 25 轮'],['几何增强','训练时水平翻转概率 0.5，横纵各随机平移 -2 至 2 像素'],['噪声混合','每张训练图独立以 0.5 概率加噪，其余保持无噪声；固定方案 σ=0.15；随机方案 σ 在 [0.05,0.20] 上均匀采样'],['优化设置','AdamW；学习率 0.0003；weight decay 0.0001；25 轮余弦衰减至 0.00001；batch=256'],['GPU 数值精度','float32 参数；CUDA autocast float16 与 GradScaler；CUDA 不可用时退出']],[104,376])
P('先执行几何增强，再生成独立高斯噪声，最后裁剪及标准化。干净样本与噪声样本共享同一个分类器和交叉熵目标，使模型在参数更新过程中同时接触两类输入。每个批次的实际噪声比例会有随机波动，并非强制恰好一半。')
P('推理时固定模型、归一化和预测规则，不依据输入是否加噪切换模型，也不使用真实类别进行路由。三种子集成仅对三个模型的 softmax 概率等权求平均，不搜索测试集权重，不做测试时增强。')
page()
# PAGE 3
H('三 双条件验证与配置锁定')
P('预先确定固定强度和随机强度两个候选，均先用种子 42，从同一检查点开始微调 25 轮。两个候选的网络、优化器、打乱顺序与几何增强设置一致，主要比较训练噪声强度分布的差异。')
T([['候选方案','最佳轮次','干净验证 %','噪声验证 %','两者最小值 %'],*[[cnames[r['config']],r['best_epoch'],f'{r["validation"]["accuracy"]*100:.2f}',f'{r["noisy_validation"]["accuracy"]*100:.2f}',f'{min(r["validation"]["accuracy"],r["noisy_validation"]["accuracy"])*100:.2f}'] for r in search]],[145,65,90,90,90])
P('噪声验证图由验证子集生成，σ=0.15，独立 NumPy 种子 71717；训练噪声使用单独的 GPU 随机生成器。官方固定噪声测试使用种子 12345，未把该测试扰动实例复制进训练或验证。')
P('选模优先保留干净验证准确率至少 92.5% 的检查点，再最大化干净与噪声准确率的较小值；相同时比较二者平均准确率，再比较平均交叉熵，完全相同时保留较早轮次。92.5% 是预先设定的验证保护线，避免以明显降低干净性能换取抗噪收益。')
I(FIG/'dual_validation.png',480,178,'图 2　两个候选的干净和噪声验证曲线　固定种子 42　不含官方测试成绩')
P(f'按照上述规则选中“{cnames[lock["config"]]}”，记录写入 selection_lock.json。随后仅对该配置运行种子 2026、9122，并分别依据各自双条件验证表现选择检查点。配置和集成方式在正式评价之前锁定。')
P(f'实际设备为 RTX 3050 Laptop GPU（4GB 显存），PyTorch {protocol["torch"]}、CUDA {protocol["cuda"]}。启用确定性算法并关闭 cuDNN benchmark。三个种子的微调平均耗时 {statistics.mean(r["finetune_seconds"] for r in runs):.1f} 秒，不包含上一阶段训练成本。','small')
page()
# PAGE 4
H('四 正式结果与双条件验收')
P('测试保持官方 10,000 张图片、σ=0.15、像素域加噪后裁剪、NumPy 种子 12345。原 CNN 对照成绩来自本轮同条件复评记录，并与原始记录一致。单模型统计为三个固定种子的均值 ± 样本标准差；集成是一组组合预测，不报告三次重复的标准差。')
T([['模型','干净准确率 %','噪声准确率 %','噪声 Macro-F1 %'],['原 CNN',stat(baseline),stat(baseline,section='noisy_test'),stat(baseline,'macro_f1','noisy_test')],['本轮抗噪单模型',stat(runs),stat(runs,section='noisy_test'),stat(runs,'macro_f1','noisy_test')],['本轮三模型集成',f'{ens_clean*100:.2f}',f'{ens_noisy*100:.2f}',f'{ensemble["noisy_test"]["macro_f1"]*100:.2f}']],[128,118,118,116])
P(f'抗噪单模型平均干净成绩比原 CNN 高 {(clean_mean-old_clean)*100:.2f} 个百分点，噪声成绩高 {(noise_mean-old_noisy)*100:.2f} 个百分点。相对上一版，训练覆盖噪声分布后噪声识别明显恢复；但本轮同时增加了微调预算、改变学习率和选模准则，因此不能将全部变化唯一归因于噪声增强。')
T([['种子／模型','干净准确率 %','噪声准确率 %','两项是否达标'],*[[str(r['seed']),f'{r["test"]["accuracy"]*100:.2f}',f'{r["noisy_test"]["accuracy"]*100:.2f}','通过'] for r in sorted(runs,key=lambda r:r['seed'])],['三模型等权集成',f'{ens_clean*100:.2f}',f'{ens_noisy*100:.2f}','通过']],[125,110,110,135])
P(f'验收使用未四舍五入的阈值：干净准确率 ≥{old_clean*100:.6f}%，噪声准确率 >{old_noisy*100:.6f}%。三个单模型各自和固定集成都满足要求，结果不依赖于从测试集选一个最好的种子。Accuracy 为正确比例，Macro-F1 为十类 F1 的等权平均。')
P('verify_robust.py 重新加载全部三个检查点，复算两个验证条件和两个正式测试条件的预测、混淆矩阵、Accuracy 与 Macro-F1，并重新组合集成概率，结果与保存文件一致。验收脚本含显式断言，若任一个模型不满足双条件将退出失败。')
P('计算代价方面，每个单模型仍为 584,618 参数，本轮额外增加 25 轮微调；集成总参数量为 1,753,854，需三次模型前向。噪声适应训练增加训练成本，集成也增加存储和推理成本。','small')
page()
# PAGE 5
H('五 噪声强度变化与错误分析')
P('额外测试 σ=0、0.05、0.10、0.15、0.20、0.25，均在锁定配置后评价。每个强度下原 CNN 和抗噪模型使用同一组受扰图像；曲线不参与模型、权重或阈值选择。σ=0.15 是主要验收点，其余用于描述强度变化下的边界。','small')
I(FIG/'noise_strength_curve.png',480,180,'图 3　噪声强度曲线　单模型为三种子均值　阴影为样本标准差　集成为固定三模型')
P('该曲线只描述裁剪后的加性高斯噪声。训练覆盖了其中一个强度，不能把改善直接推广到模糊、遮挡、亮度变化、其他噪声分布或真实拍摄图像。更强噪声下可用轮廓逐渐丢失，仍会出现错误。')
hardest=np.argsort(ensemble['noisy_test']['recall'])[:3]; baseline_recall=np.mean([r['noisy_test']['recall'] for r in baseline],axis=0)
T([['噪声下较难类别','原 CNN 召回率 %','抗噪集成召回率 %'],*[[clabels[k],f'{baseline_recall[k]*100:.1f}',f'{ensemble["noisy_test"]["recall"][k]*100:.1f}'] for k in hardest]],[160,160,160])
P(f'逐类变化并不一致：衬衫召回率由 {baseline_recall[6]*100:.1f}% 降至 {ensemble["noisy_test"]["recall"][6]*100:.1f}%，但精确率由 {float(np.mean([r["noisy_test"]["precision"][6] for r in baseline]))*100:.1f}% 升至 {ensemble["noisy_test"]["precision"][6]*100:.1f}%，与减少把其他服饰误报为衬衫相符。总体提升并不代表每类召回率都提高。','small')
I(FIG/'noisy_errors.png',480,142,'图 4　σ=0.15 下按官方测试索引顺序选前八个错误样本　括号为未经校准的预测分数')
P(f'集成在这一条件下仍误分 {int(cm.sum()-np.trace(cm))} 张。相似上装与模糊边界仍具有歧义，高 softmax 分数也可能出错。完整逐类指标、混淆矩阵和逐样本预测均保存在结果目录，便于复查。','small')
page()
# PAGE 6
H('六 结论与复现说明')
P(f'本轮在不改变测试噪声条件的前提下，获得干净 {clean_mean*100:.2f}%、加噪 {noise_mean*100:.2f}% 的三种子单模型平均准确率；固定集成分别为 {ens_clean*100:.2f}% 和 {ens_noisy*100:.2f}%。每个单模型和集成都通过“干净不低于原 CNN、噪声高于原 CNN”的验收，说明训练覆盖目标扰动并采用双条件选模能够在本任务中改善原先的取舍。')
H('局限与后续工作')
P('官方测试集曾在前两个阶段评价，本轮是在既有基准上继续开发，不是全新未见的最终留出集。虽然本轮的候选与检查点只依据验证集选择，重复使用同一验证集和已知基准仍可能产生选择偏差，需要独立外部数据确认泛化。')
P('只比较两个噪声方案，搜索种子仅为 42；获胜方案才补充三个种子统计。测试中的加性高斯噪声有明确生成机制，与真实传感器噪声并不等价。今后可开展等训练预算消融、跨噪声分布评价和真实图片验证，并使用独立校准集评价预测置信度。')
H('复现与材料')
P('robustness/src/train_robust.py 读取原划分，微调两个候选、锁定配置、重复训练并评价强度曲线。verify_robust.py 独立验收全部模型；infer_robust.py 对干净或噪声输入始终使用同一分类器。完整命令见本目录 README.md。')
P('本包保留本轮官方数据、独立代码、协议、逐轮日志、模型、逐样本 logits、集成概率和核验结果。原 CNN 仅保留必要对照统计；完整重训所需初始化可由 pretrain.py 在 GPU 上重新生成。build_robust_report.py 从实测 JSON 生成本报告和图表，未手工填造实验数字。跨硬件、跨版本不保证逐位一致。','small')
H('参考资料与 AI 工具说明')
P('[1] Xiao H, Rasul K, Vollgraf R. Fashion-MNIST. 2017. <link href="https://arxiv.org/abs/1708.07747" color="#1F4E79">arXiv:1708.07747</link>；官方数据：<link href="https://github.com/zalandoresearch/fashion-mnist" color="#1F4E79">github.com/zalandoresearch/fashion-mnist</link>。','small')
P('[2] Ioffe S, Szegedy C. Batch Normalization. 2015. <link href="https://arxiv.org/abs/1502.03167" color="#1F4E79">arXiv:1502.03167</link>；Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. <link href="https://arxiv.org/abs/1711.05101" color="#1F4E79">arXiv:1711.05101</link>。','small')
P('[3] PyTorch 2.14 官方文档：<link href="https://docs.pytorch.org/docs/2.14/generated/torch.randn.html" color="#1F4E79">torch.randn</link>、torch.clamp、AdamW、CosineAnnealingLR 与 <link href="https://docs.pytorch.org/docs/2.14/amp.html" color="#1F4E79">torch.amp</link>。核对日期 2026-10-31。','small')
P('使用 OpenAI Codex 辅助方案设计、代码、GPU 训练、核验和报告。公开数据和方法来源已标注，未复制第三方训练脚本；所有指标来自本机真实运行，未编造结果。尚未上传 GitHub，也未发送作业邮件。','small')

out=ROB/'2026439122-范哲豪.pdf'
def footer(canvas,doc):
    canvas.setFont('CN',8); canvas.setFillColor(colors.HexColor('#555555')); canvas.drawString(52,29,'深度学习课程设计 2026 秋季 抗噪优化版'); canvas.drawRightString(A4[0]-52,29,f'2026439122 范哲豪    {doc.page} / 6')
SimpleDocTemplate(str(out),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=42,bottomMargin=46,title='服饰图像分类的抗噪优化与双条件验证',author='范哲豪',subject='深度学习课程设计 抗噪优化版 2026439122').build(story,onFirstPage=footer,onLaterPages=footer)
(ROB/'report'/'技术报告抗噪版.md').write_text('\n\n'.join(md),encoding='utf-8'); print(out)

