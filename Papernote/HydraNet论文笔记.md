##HydraNet论文阅读笔记

2021.7.28

###TaBert
论文：《Hybrid Ranking Network for Text-to-SQL》

arvix，2020 by Microsoft

**观点**

尝试将预训练模型运用到text-to-sql任务中，对以往将所有列和自然语言问题直接拼接的方法进行探讨。

**论文创新点**

提出一种简单的方法：混合排序网络Hybrid Ranking Network (HydraNet)，该方法将问题分解成按列排序和解码，最后按列输出组合成一个SQL查询，按照简单的规则。encoder部分给定问题和单列，完全符合Bert和RoBerta的训练任务，因此我们避免了任何ad-hoc池化和额外的编码层。

**实验结果**：

针对WiKiSQL数据集。WiKiSQL数据集的局限在于：每个问题只涉及到单表，并且该表已知。但数据集仍具有挑战性，因为表和问题的多样性。

本文模型在WiKiSQL上取得了最前（top place）的效果。

**以往研究总结**

主要采用类似的encoder-decoder结构。对Nl问题和表模式进行编码，得到隐藏向量，再将隐藏向量解码成SQL查询。最近在确保SQL的语法正确（syntax correct）上有了突破。

最近由许多研究针对将预训练模型用于Text-to-SQL任务，取得了良好效果。

前面在WikiSQL的研究主要在几点：

1. 编码部分，融合NL问题和表模式的信息
2. 解码部分，输出的SQL准确和可执行
3. 利用预训练模型

论文将解码部分分成几个子任务：where，select等，和SQLNet中类似，但没有用到预训练模型，并且用了几个单独的模型做训练，而本文只用了一个。

---

####模型

**输入表示**

c=concat(type,table_name,column),q

[CLS] c1,c2,..ck [SEP] q1,q2,...[SEP]

