# cifar 10

## worse_label

**固定种子没有重复实验**

SOP best acc 93%

**SOP+偏标记**

100 epoch的loss，predicition集成：

监督训练（50%样本+mixup+增强） + 偏标记训练（50%样本+候选集为4）：best acc 91.15%
监督训练（50%样本+mixup+增强） + 偏标记训练（50%样本+候选集为3）：best acc 91.95%
监督训练（50%样本+mixup+增强） + 偏标记训练（50%样本+候选集为2）：best acc 93.03%

监督训练（50%样本+mixup+增强）: best acc:93.07%
监督训练（50%样本）：beast acc:90.42

SOP+半监督
