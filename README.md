# 本轮抗噪课程设计 独立版

姓名：范哲豪　学号：2026439122

正式报告为 `提交文件/2026439122-范哲豪.pdf`，共 6 页。前两轮的实验目录、旧模型、旧报告、日志和压缩包已经清理。本项目不依赖那些目录。

## 保留成绩

| 模型 | 干净准确率 | 固定噪声准确率 |
| --- | --- | --- |
| 原 CNN 对照统计 | 90.70% | 79.33% |
| 本轮单模型三种子平均 | 94.16% | 90.56% |
| 本轮固定三模型集成 | 94.36% | 91.15% |

三个单模型各自及集成都通过干净成绩不低于原 CNN、噪声成绩高于原 CNN 的验收。固定噪声为像素域高斯标准差 0.15，裁剪 [0,1]，NumPy 种子 12345。

## 目录

- data：官方四个 gzip、许可证、哈希清单、固定 54000/6000 划分；测试集 10000。
- models：本轮最终三个模型和本轮随机强度验证候选，不含旧阶段权重。
- results：本轮训练、验证、测试预测、集成概率、验收和核验结果。
- results/baseline_reference.json：为当前比较保留的原 CNN 必要统计及曲线；不含旧权重或旧训练日志。
- src/common.py：独立数据读取、网络、增强与评价代码。
- src/train_robust.py：本轮噪声混合微调与双条件验证、锁定及测试。
- src/pretrain.py：完整重训时，在 GPU 上重新生成所需初始化；无需已删除的旧权重。
- src/verify_robust.py：重新加载本轮模型，独立复算两个验证与两个测试条件。
- src/infer_robust.py：GPU 单模型或等权三模型集成推理，同一模型处理干净和噪声输入。
- src/build_robust_report.py：自动生成报告和图表。
- report：本轮图表及可编辑 Markdown；work/packages 是报告用绘图库，不属于旧实验结果。

## 本机运行

```powershell
Set-Location -LiteralPath 'D:\深度学习\robustness'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$py='D:\software\miniconda3\envs\torch\python.exe'
& $py .\src\verify_robust.py
& $py .\src\infer_robust.py --test-index 0
& $py .\src\infer_robust.py --test-index 0 --sigma 0.15
& $py .\src\infer_robust.py --test-index 0 --sigma 0.15 --single-seed 42
```

`运行抗噪实验.ps1` 会复用已经完成的本轮运行并重新评价及核验。若在独立副本中完整重训，pretrain.py 会先从官方训练数据重建初始化，再执行本轮微调，耗时比仅推理明显更多。初始化中间文件写入 work/pretraining，不依赖父目录文件。完整重训的基线曲线采用已记录的对照统计，不重新加载已删除的旧 CNN 权重。

训练设置：初始化训练预算 50 轮，本轮微调 25 轮；50% 概率加噪；比较固定 sigma=0.15 与随机 sigma∈[0.05,0.20]，双条件验证选择固定方案。三个种子为 42、2026、9122。AdamW 微调 lr=0.0003，weight_decay=0.0001，batch=256，余弦衰减，CUDA AMP。细节见 results/protocol.json。

报告生成使用 Codex 捆绑 Python；matplotlib 位于本项目 work/packages；中文字体使用 Windows 宋体和黑体。完整压缩包不含 work 依赖目录，其他环境可安装 requirements-report.txt 后生成报告。训练和推理依赖 requirements.txt。CUDA 不可用时退出，不回退 CPU。

官方测试集在历史开发中已被查看，本轮结果不是全新外部数据的性能保证。成绩仅对应规定条件；softmax 分数未经校准。历史初始化路径与哈希作为原始实验溯源信息保留在检查点/运行记录中，它们不是当前脚本的文件依赖。

## 来源与 AI 使用

Fashion-MNIST：https://github.com/zalandoresearch/fashion-mnist  
PyTorch：https://docs.pytorch.org/docs/2.14/  
BatchNorm：https://arxiv.org/abs/1502.03167  
AdamW：https://arxiv.org/abs/1711.05101

使用 OpenAI Codex 辅助代码、GPU 运行、核验与报告。数值来自真实实验，未复制第三方训练脚本，未编造结果。代码仓库：https://github.com/MoRan-FZH/fashion-mnist-robustness/tree/main ；未发送作业邮件。
