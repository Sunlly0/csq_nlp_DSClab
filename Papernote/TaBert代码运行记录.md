##TaBert代码运行过程记录

2021.7.9

###环境

VMWare Workstation 16 Pro + Ubuntu 20.04 虚拟机

安装anaconda3ls

###过程
####1. 安装Tabert

后续按TaBert步骤进行操作（https://github.com/facebookresearch/TaBERT）

下载代码，解压缩

在TABERT文件夹中进入终端

运行命令，建立新的tabert的conda环境：

```
bash scripts/setup_env.sh
```

由于其中部分依赖有冲突，比如下载的torch版本为1.3.1，和torch_scatter冲突，故对env.yml进行修改：
name: tabert
channels:
  - defaults
dependencies:
  - python=3.6
  - numpy
  - pip
  - pandas
  - pip:
    - cython
    - pandas +
    - tqdm
    - spacy +
    - redis +
    - pyzmq +
    - ujson  +
    - msgpack +
    - h5py +

完成后切换至(tabert)

```
conda activate tabert
```

其中，使用pip install手动安装，前三者需要先下载：

```
pip install torch-1.5.0+cpu-cp36-cp36m-linux_x86_64.whl
```

```
pip install torch_scatter-2.0.5+cpu-cp36-cp36m-linux_x86_64.whl
```

```
pip install fairseq-0.8.0.tar.gz
```

在TaBert目录下安装：
```
pip install --editable=git+https://github.com/huggingface/transformers.git@372a5c1ceec49b52c503707e9657bfaae7c236a0#egg=pytorch_pretrained_bert
```

使用conda install --offline 安装，需提前下载包：
```
conda install --offline torchvision-0.4.1-py36_cpu.tar.bz2
```

安装table-bert

```
pip install --editable .
```

#####下载预训练模型

安装gdown：

```
conda install gdown
````

下载预训练模型：

TaBERT_Base_(k=1)

```
gdown 'https://drive.google.com/uc?id=1-pdtksj9RzC4yEqdrJQaZu4-dIEXZbM9'
````

TaBERT_Base_(K=3)

```
gdown 'https://drive.google.com/uc?id=1NPxbGhwJF1uU9EC18YFsEZYE-IQR7ZLj'
````

TaBERT_Large_(k=1)

```
gdown 'https://drive.google.com/uc?id=1eLJFUWnrJRo6QpROYWKXlbSOjRDDZ3yZ'
````

TaBERT_Large_(K=3)

````
gdown 'https://drive.google.com/uc?id=17NTNIqxqYexAzaH_TgEfK42-KmjIRC-g'
````

此处若下载失败，可利用科学上网

解压缩待载入的模型。（tar zxvf xxx.tar.gz）

#####载入预训练模型

````python

from table_bert import TableBertModel

model = TableBertModel.from_pretrained(
    './tabert_base_k1/model.bin',
)

````

若有报错，转附录1。若无报错，则运行：

````
from table_bert import TableBertModel
from table_bert import Table, Column

model = TableBertModel.from_pretrained(
    './tabert_base_k1/model.bin',
)



table = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text', sample_value='United States'),
        Column('Gross Domestic Product', 'real', sample_value='21,439,453')
    ],
    data=[
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165'],
    ]
).tokenize(model.tokenizer)


context = 'show me countries ranked by GDP'

context_encoding, column_encoding, info_dict = model.encode(
    contexts=[model.tokenizer.tokenize(context)],
    tables=[table]
)

print(context_encoding.shape)
print(column_encoding.shape)
```
最后结果如下：

torch.Size([1, 7, 768])

torch.Size([1, 2, 768])

则说明环境安装成功，能顺利载入预训练模型。

####2. Tabert的应用案例：MAPO在WikiTableQuestions上使用

进入tabert的example文件，下载应用案例的源代码（https://github.com/pcyin/pytorch_neural_symbolic_machines）

根据依赖data/env.yml创建requiremnets.txt文件，并使用pip安装文件中的所有项目。

```
pip install -r requirements.txt
```

下载WikiTableQuestions dataset:

```
wget http://www.cs.cmu.edu/~pengchey/pytorch_nsm.zip
```

对其进行解压缩：

```
unzip pytorch_nsm.zip
```

运行代码，使用bert的预训练参数的vanilla bert，模型为bert-base-uncased，对模型进行训练（train）：

```
python -m table.experiments train --work-dir=runs/demo_run --config=data/config/config.vanilla_bert.json
```

完成训练后，在runs/demo_run的文件夹中能找到model.best.bin

运行代码，对该模型进行测试（test）：

```
python -m table.experiments test --model runs/demo_run/model.best.bin --test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/test_split.jsonl 
```

在后续载入Tabert预训练模型进行训练和测试。

在data/config/中新建config.tabert_bert.json，内容和config.vanilla_bert.json相同，只不过将"table_bert_model_or_config"一项更改为tabert预训练模型路径，比如: "/home/sunlly/code/TaBERT/tabert_base_k1/model.bin",

对tabert模型进行训练：

```
OMP_NUM_THREADS=1 python -m table.experiments train --work-dir=tabert_runs/demo_run --config=data/config/config.tabert_bert.json --extra-config='{"actor_use_table_bert_proxy": true}'
```

完成后对模型进行测试：

python -m table.experiments test --model tabert_runs/demo_run/model.best.bin --test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/train_examples.jsonl

####3. 中文情感分类

新建conda虚拟环境bert_tf

conda -n bert_tf python==3.6

安装环境：

pip install tensorflow==1.11.0

pip install pandas

微调训练：

python3 run_classifier.py --data_dir=data --task_name=sim --vocab_file=chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json --output_dir=sim_model --do_train=true --do_eval=true --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=70 --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=3.0

测试：

python3.6 run_classifier.py --task_name=sim --do_eval=true   --data_dir=data --vocab_file=chinese_L-12_H-768_A-12/vocab.txt   --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json   --init_checkpoint=sim_model --max_seq_length=70 --output_dir=output

####4. docker的pytorch容器环境搭建记录

容器初始环境为：pytorch1.7.0+cuda11.0

**安装tabert：**
准备工作，首先安装以下命令：

```
apt-get install git
```
```
apt-get install wget
```
```
apt-get install unzip
```

随后使用 pip install xxx 的方式安装以下包：
    - pandas +
    - tqdm
    - spacy +
    - redis +
    - pyzmq +
    - ujson  +
    - msgpack +
    - h5py +
    - fairseq
在TaBert目录下安装：
```
pip install --editable=git+https://github.com/huggingface/transformers.git@372a5c1ceec49b52c503707e9657bfaae7c236a0#egg=pytorch_pretrained_bert
```

安装与pytorch-gpu和cuda版本适配的torch_scatter，需提前下载包：

1.7.0的网站是：https://pytorch-geometric.com/whl/torch-1.7.0.html
随后使用pip install 包名 进行安装。

安装table-bert

```
pip install --editable .
```
测试是否安装成功的方法和上part1相同

**tabert的应用案例：**

基本同上part2。以下为使用gpu的情形的命令：


运行代码，使用bert的预训练参数的vanilla bert，模型为bert-base-uncased，对模型进行训练（train）：

```
python -m table.experiments train --cuda --work-dir=runs/demo_run --config=data/config/config.vanilla_bert.json --extra-config='{"actor_use_table_bert_proxy": true}'
```

完成训练后，在runs/demo_run的文件夹中能找到model.best.bin

运行代码，对该模型进行测试（test）：

```
python -m table.experiments test --cuda --model runs/demo_run/model.best.bin --test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/test_split.jsonl 
```

在后续载入Tabert预训练模型进行训练和测试。

在data/config/中新建config.tabert_bert.json，内容和config.vanilla_bert.json相同，只不过将"table_bert_model_or_config"一项更改为tabert预训练模型路径，比如: "/home/sunlly/code/TaBERT/tabert_base_k1/model.bin",

对tabert模型进行训练：

```
OMP_NUM_THREADS=1 python -m table.experiments train --cuda --work-dir=tabert_runs/demo_run --config=data/config/config.tabert_bert.json --extra-config='{"actor_use_table_bert_proxy": true}'
```

完成后对模型进行测试：

python -m table.experiments test --cuda --model tabert_runs/demo_run/model.best.bin --test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/train_examples.jsonl

python -m table.experiments test --cuda --model tabert_runs/demo_run/model.best.bin --test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/test_split.jsonl

###附录

######附录1
报错后按照提示，补充安装：

TypeError: __init__() got an unexpected keyword argument 'layer_norm_eps'
在config.py中将出错语句注释，有两句 #layer_norm_eps=1e-12,

安装git后，运行：
pip install --editable=git+https://github.com/huggingface/transformers.git@372a5c1ceec49b52c503707e9657bfaae7c236a0#egg=pytorch_pretrained_bert


conda install fairseq


AttributeError: module 'torch.nn' has no attribute 'GELU'
修改原nn.GELU()该句为activation=F.gelu,

ModuleNotFoundError: No module named 'torch_scatter'
尝试conda install torch_scatter，无效
在https://pytorch-geometric.com/whl/torch-1.5.0.html下载对应版本whl文件，我下载的是torch_scatter-latest+cpu-cp36-cp36m-linux_x86_64.whl
切换至tabert的conda环境，然后执行：
pip install torch_scatter-latest+cpu-cp36-cp36m-linux_x86_64.whl 
pip install torch_scatter-2.0.3+cpu-cp36-cp36m-linux_x86_64.whl 


OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory
初步判断是torch_scatter的问题，但是目前torch-geometric官网上
https://pytorch-geometric.com/whl/的wheel文件torch_scatter仅支持torch 1.4.0以上的版本
torch_scatter 的2.0.3,2.0.4,2.0.5均尝试，无果
tabert环境中的torch 版本为1.3.1
尝试安装torch1.5.0版本，由于python包冲突而无法安装
尝试使用pip安装torch_scatter1.4.0，下载tar.gz包，安装时出现大串红色报错

######附录2


fairseq=0.10.1要求pytorch>=1.5.0

conda install --use-local fairseq-0.8.0.tar.gz 


执行OMP_NUM_THREADS=1 python -m \
  table.experiments \
  train \
  seed 0 \
  --cuda \
  --work-dir=runs/demo_run \
  --config=data/config/config.vanilla_bert.json

报错：

意识到很有可能是fairseq和pytorch的版本
冲突，卸载fairseq-0.10.2,安装fairseq-0.8.0 失败

下载fairseq包，用pip install fairseq-v0.8.0.tar.bz2 报错

尝试，su root进入root，并安装：

apt-get update && apt-get install build-essential

su sunlly切换回普通用户。

再尝试pip install fairseq-v0.8.0.tar.bz2 

安装成功！

问题
class FairseqBMUF(FairseqOptimizer):
  File "/home/sunlly/anaconda3/envs/tabert/lib/python3.6/site-packages/fairseq/optim/bmuf.py", line 138, in FairseqBMUF
    @torch.no_grad()
TypeError: 'no_grad' object is not callable

网上查到说是因为pytorch版本太低的缘故，目前的pytorch版本为0.4.0

EDIT: I saw you detailed it in the title, you are using pytorch 0.4

Please update pytorch to version 0.4.1 or 1.0-preview, then all should be working normally.

下载pytorch0.4.1并尝试安装

AttributeError: module 'torch' has no attribute 'ops'

重装tabert3，和前面步骤一样，除了将torch版本变为0.4.1

得到    from torch_scatter import scatter_max, scatter_mean
  File "/home/sunlly/anaconda3/envs/tabert3/lib/python3.6/site-packages/torch_scatter/__init__.py", line 24, in <module>
    raise AttributeError(e)
AttributeError: module 'torch' has no attribute 'ops'

错误

将torch版本将为0.4.0

附录3 发现上面失误了，装的是torch-1.4.1

#######附录3

新建环境tabert3

torch==1.5.0
torch-scatter==2.0.5
fairseq==0.8.0
torchvision==0.4.1

预训练模型能正常运行

使用pip 对如下依赖包安装：
  - nltk==3.3
  - docopt
    - tqdm
    - bloom-filter==1.3
    - Babel==2.5.3
    - gensim==3.2.0
    - tensorboardX
    - editdistance

python -m table.experiments train --work-dir=runs/demo_run --config=data/config/config.vanilla_bert.json

OMP_NUM_THREADS=1 python -m \
  table.experiments \
  train \
  seed 0 \
  --cuda \
  --work-dir=runs/demo_run \
  --config=data/config/config.vanilla_bert.json

OMP_NUM_THREADS=1 python -m table.experiments train --work-dir=runs/demo_run --config=data/config/config.vanilla_bert.json


OMP_NUM_THREADS=1 python -m table.experiments train --work-dir=runs/demo_run --config=data/config/config.vanilla_bert.json --extra-config='{"actor_use_table_bert_proxy": true}'

中途报错OSError: [Errno 28] No space left on device

是因为ubuntu虚拟机的硬盘大小不足，

关闭虚拟机，在虚拟机设置-硬盘中选择扩展，下载Gparted扩充系统硬盘。

再次运行，出现：
Table Bert Config
{
  "base_model_name": "bert-base-uncased",
  "column_delimiter": "[SEP]",
  "context_first": true,
  "column_representation": "mean_pool_column_name",
  "max_cell_len": 5,
  "max_sequence_len": 512,
  "max_context_len": 256,
  "do_lower_case": true,
  "cell_input_template": [
    "column",
    "|",
    "type",
    "|",
    "value"
  ],
  "masked_context_prob": 0.15,
  "masked_column_prob": 0.2,
  "max_predictions_per_seq": 100,
  "context_sample_strategy": "nearest",
  "table_mask_strategy": "column",
  "vocab_size": 30522,
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "hidden_act": "gelu",
  "intermediate_size": 3072,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "max_position_embeddings": 512,
  "type_vocab_size": 2,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12
}
/home/sunlly/anaconda3/envs/tabert3/lib/python3.6/

模型运行时报错，运行终止：

multiprocessing/semaphore_tracker.py:143: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
  len(cache))

为解决警告，在命令窗口运行：
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

再次运行，得到：

tabert3) sunlly@ubuntu:~/code/pytorch_neural_symbolic_machines$ OMP_NUM_THREADS=1 python -m table.experiments train --work-dir=runs/demo_run --config=data/config/config.vanilla_bert.json
WARNING:root:You are using the old version of `pytorch_pretrained_bert`
load config file [data/config/config.vanilla_bert.json]
work dir [runs/demo_run]
creating work dir [runs/demo_run]
WARNING:root:You are using the old version of `pytorch_pretrained_bert`
Evaluator uses device cpu
initializing 16 actors
starting 16 actors
starting evaluator
starting learner
Learner process 5420, evaluator process 5419
WARNING:root:You are using the old version of `pytorch_pretrained_bert`
WARNING:root:You are using the old version of `pytorch_pretrained_bert`
WARNING:root:You are using the old version of `pytorch_pretrained_bert`
WARNING:root:You are using the old version of `pytorch_pretrained_bert`
WARNING:root:You are using the old version of `pytorch_pretrained_bert`
WARNING:root:You are using the old version of 
Loading table BERT model bert-base-uncased
Learner exited

比之前还迷惑

运行直到模型训练结束


执行下面代码运行测试：

python -m table.experiments test --model runs/demo_run2/model.best.bin --test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/test_split.jsonl 


python -m table.experiments test --model runs/demo_run2/model.best.bin --test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/train_examples.jsonl 

载入tabert作为预训练模型，修改路径和config.tabert_bert.json中的 'table_bert_model_or_config' 为下载的预训练模型路径：

OMP_NUM_THREADS=1 python -m table.experiments train --work-dir=tabert_runs/demo_run --config=data/config/config.tabert_bert.json --extra-config='{"actor_use_table_bert_proxy": true}'

python -m table.experiments test --model tabert_runs/demo_run/model.best.bin --test-file data/wikitable/wtq_preprocess_0805_no_anonymize_ent/train_examples.jsonl 


base_k3
```
OMP_NUM_THREADS=1 python -m table.experiments train --work-dir=tabert_runs/demo_run_base_k3 --config=data/config/config.tabert_bert_base_k3.json --extra-config='{"actor_use_table_bert_proxy": true}'
```
