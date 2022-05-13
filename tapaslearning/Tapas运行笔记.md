## Tapas运行笔记

2021.9.9

参考：
+ transformers-安装：https://huggingface.co/transformers/installation.html
+ transformers-TAPAS：https://huggingface.co/transformers/model_doc/tapas.html

---

### 步骤

1. 创建新容器：
```
docker run -itd -v [宿主机目录]:[容器目录] --gpus all --name [容器名] --shm-size="2g"  pytorch/pytorch

<!-- docker run -itd -v /home/chensiqin/sqtorch2022:/files --gpus all --name sqtorch2022 --shm-size="2g" pytorch/pytorch -->
```

若不限定版本，则默认安装cpu版

2. 安装transformers
```
pip install transformers
```

2. 安装torch_scatter，适用于torch1.8.1的GPU、CPU版本
```
conda install pytorch-scatter -c pyg
```
或
```
pip install torch_scatter
```

3. 加载tapas预训练模型
```py
from transformers import TapasConfig, TapasForQuestionAnswering
# for example, the base sized model with default SQA configuration
model = TapasForQuestionAnswering.from_pretrained('google/tapas-base')
# or, the base sized model with WTQ configuration
config = TapasConfig.from_pretrained('google/tapas-base-finetuned-wtq')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-base', config=config)
# or, the base sized model with WikiSQL configuration
config = TapasConfig('google-base-finetuned-wikisql-supervised')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-base', config=config)
```

### 实验结果


---

## 附录




## 问题

#### 1.
问题：无法安装torch-scatter，报错信息：ERROR: Failed building wheel for torch-scatter

解决：安装g++。

```
apt-get update
apt-get upgrade
apt-get build-essential
```

#### 2.
问题：conda安装torch-scatter报错，报错信息：ERROR: Failed building wheel for torch-scatter

解决：conda update --all
