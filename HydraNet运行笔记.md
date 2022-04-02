## HydraNet运行笔记

by Sunlly
2021.7.29

论文：《Hybrid Ranking Network for Text-to-SQL》 2020

github代码：https://github.com/lyuqin/HydraNet-WikiSQL

---

### 步骤

1. 创建新容器：
```
docker run -itd [宿主机目录]:[容器目录] --gpus all --name [容器名] --shm-size="2g"  pytorch/pytorch
```
2. 创建目录和git clone项目
```
cd hydranet
git clone https://github.com/lyuqin/HydraNet-WikiSQL.git
```
3. 创建conda虚拟环境
```
conda create -n hydranet python=3.8
conda activate hydranet
```
4. 安装依赖：安装pytorch with CUDA 11.0
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
```
pip install -r requirements.txt
```
5. 数据预处理，用WiKiSQL数据集
```
mkdir data && mkdir output
```
```
git clone https://github.com/salesforce/WikiSQL && tar xvjf WikiSQL/data.tar.bz2 -C WikiSQL
```
```
python wikisql_gendata.py
```
6. 训练
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


---

## 附录

*：需要在创建容器时增加对shm共享内存的设置，否则后续运行会出错。

也可创建docker环境：
```
docker build -t hydranet -f Dockerfile .
```
在运行过程中，出现cuda_runout_memory错误：

此时在config中将batch_size改为原来的一半并保存：256->128
