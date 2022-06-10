## HydraNet运行笔记

by Sunlly
2021.7.29

2022.6.2

论文：《Hybrid Ranking Network for Text-to-SQL》 2020

github代码：https://github.com/lyuqin/HydraNet-WikiSQL

---

### 步骤

1. 创建新容器：
```
docker run -itd -v [宿主机目录]:[容器目录] --gpus all --name [容器名] --shm-size="2g"  pytorch/pytorch
```
2. 创建目录和git clone项目
```
mkdir hydranet && cd hydranet
git clone https://github.com/lyuqin/HydraNet-WikiSQL.git
```
3. 安装依赖：

```
pip install -r requirements.txt
```
5. 数据预处理，用WiKiSQL数据集
```
mkdir data && mkdir output
```
```
git clone https://github.com/salesforce/WikiSQL
tar xvjf WikiSQL/data.tar.bz2 -C WikiSQL
```
```
python wikisql_gendata.py
```
6. 训练(--gpu项： 看自己有几张卡)
```
python main.py train --conf conf/wikisql.conf --gpu 0,1,2,3 --note "some note"
```
在output文件夹中可以找到训练的模型，目录内以训练开始的时间命名。

7. 评价

在wikisql_prediction.py中修改模型、输入和输出的设置

运行以下命令，获得官方发布的评价结果：
```
cd WikiSQL
```
```
python evaluate.py data/test.jsonl data/test.db ../output/test_out.jsonl
```
---

### 实验结果
main.py
config：bert-large:
跑了 4 个epoch :
wikidev.jsonl: overall:83.3, agg:91.0, sel:97.6, wn:98.5, wc:95.3, op:99.1, val:97.5
wikitest.jsonl: overall:83.2, agg:91.3, sel:97.4, wn:98.1, wc:94.8, op:99.1, val:97.4

论文结果：
dev 83.5, test 83.4
和论文相差不大，可能和batch大小有关：会在同一个batch中计算交叉熵损失。原文 batch=256，自己跑的时候batch=64

wikisql_prediction.py
top1000问题
data_preprocess/wikitest_top1000.jsonl loaded. Data shapes:
input_ids (6776, 96)
input_mask (6776, 96)
segment_ids (6776, 96)
num of samples: 1000
PyTorch model loaded from output/20220602_114141/model_3.pt
===HydraNet===
sel_acc: 0.934
agg_acc: 0.911
wcn_acc: 0.979
wcc_acc: 0.918
wco_acc: 0.976
wcv_acc: 0.938

===HydraNet+EG===
sel_acc: 0.934
agg_acc: 0.911
wcn_acc: 0.98
wcc_acc: 0.964
wco_acc: 0.979
wcv_acc: 0.967
---

## 附录

*：需要在创建容器时增加对shm共享内存的设置，否则后续运行会出错。

也可创建docker环境：
```
docker build -t hydranet -f Dockerfile .
```
在运行过程中，出现cuda_runout_memory错误：

此时在config中将batch_size改为原来的一半并保存：256->128或64(经测试，host5上跑不了128)
