# cifar 10

Cifar10_aggre 0.956（sop）

Cifar10_random 0.953 (sop)

cifar10_worse_label 93.4(sop+comatch)

## worse_label

**固定种子没有重复实验**

SOP best acc 93%

**50%数据监督训练：**

监督训练（50%样本+mixup+增强）: best acc:93.07% (213服务器 train_sup_mixup.py)
监督训练（50%样本）：beast acc:90.42 （213服务器 clean train.py）

**SOP+偏标记(do no work)**

100 epoch的loss，predicition集成：

监督训练（50%样本+mixup+增强） + 偏标记训练（50%样本+候选集为4）：best acc 91.15%
监督训练（50%样本+mixup+增强） + 偏标记训练（50%样本+候选集为3）：best acc 91.95%
监督训练（50%样本+mixup+增强） + 偏标记训练（50%样本+候选集为2）：best acc 93.03%

**SOP+半监督**

50%干净 50%无标记 comatch ：best ema acc is 93.48% （ying-peng dir:comatch）
