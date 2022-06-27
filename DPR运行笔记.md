## DPR运行笔记

by Sunlly

2022.6.16

论文：《Dense Passage Retrieval for Open-Domain Question Answering》 2020

github代码：https://github.com/facebookresearch/DPR

---

### 步骤

1. 创建新容器：
```
docker run -itd -m 10g -v [宿主机目录]:[容器目录] --gpus all --name [容器名] --shm-size="2g"  pytorch/pytorch
```
2. 创建目录和git clone项目
```
git clone git@github.com:facebookresearch/DPR.git
cd DPR
```
3. 安装依赖：

```
pip install .
```
4. 运行：
```
python train_dense_encoder.py \
train_datasets=[list of train datasets, comma separated without spaces] \
dev_datasets=[list of dev datasets, comma separated without spaces] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```
使用 nq 的数据集作为测试：

```
python train_dense_encoder.py \
train_datasets=nq-train.json \
dev_datasets=nq-dev.json \
train=biencoder_local \
output_dir=test_nq_20220616
```

修改 train_dense_encoder.py 的 args 后可以直接运行和调试。

### 问题

1. 运行 train_dense_encoder.py

报错：OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.

原因：包下载有问题

解决： 尝试 python -m spacy download en_core_web_sm 无效；

下载：https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl

安装：pip install en_core_web_sm-3.3.0-py3-none-any.whl

安装成功

### 处理记录

**1. 将 wikisql 训练集、测试集做处理，训练 DPR**

处理数据集，将 前面筛除过的 test 集 中的table，出现过的table 保留，没有出现过的table删除(wikisql_remove_tables_out100.py)，形成新的 test.tables.jsonl(test.tables_remove_out100.jsonl)

test.tables.jsonl
total: 5230个表
count: 4628
remove: 602

修改 wikisql_generatedata 的代码，用 WikiSQL 生成符合 dpr 训练数据形式的数据集，格式如下：
```
[
    {
        "dataset": "nq_dev_psgs_w100",
        "question": "who sings does he love me with reba",
        "answers": [
            "Linda Davis"
        ],
        "positive_ctxs": [
            {
                "title": "Does He Love You",
                "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. ",
                "score": 13.394315,
                "title_score": 0,
                "passage_id": "11828866"
            },
            {
                "title": "Red Sandy Spika dress of Reba McEntire",
                "text": "Red Sandy Spika dress of Reba McEntire American recording artist Reba McEntire wore a sheer red dress to the 1993 Country Music Association Awards ceremony on September 29, 1993.",
                "score": 12.924647,
                "title_score": 0,
                "passage_id": "15632586"
            }
        ],
        "negative_ctxs": [
            {
                "title": "Cormac McCarthy",
                "text": "chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. ",
                "score": 0,
                "title_score": 0,
                "passage_id": "2145653"
            },
            {
                "title": "Pragmatic Sanction of 1549",
                "text": "one heir, Charles effectively united the Netherlands as one entity. ",
                "score": 0,
                "title_score": 0,
                "passage_id": "2271902"
            }
          ]
        },
        {
          "dataset": "nq_dev_psgs_w100",
          "question": "who sings does he love me with reba",
          "answers": [
              "Linda Davis"
          ],
          ...
        }
]
```

根据代码的 github 网站上所述，其实score 是没有在模型中用到的，但是包含在了数据集内。

** 不确定bm25的负样本是如何融入到模型的训练中去的

写了一个处理数据集的代码，将原数据集/筛选过后的数据集修改成DPR 的输入数据集：(data_generate_dpr.py),自己定义了标签名：

```python
neg_tables=[]
for i in range(neg_num):
    random_idx=random.randint(1,len(tables_ids))-1
    # print("random_idx:",random_idx)
    random_table_id=tables_ids[random_idx]

    while random_table_id == origin_table_id:
        random_idx=random.randint(1,len(tables_ids))-1
        random_table_id=tables_ids[random_idx]

    random_table_content=tables[random_table_id]

    neg_sample={}
    neg_sample["table_id"]=random_table_id
    neg_sample["content"]=random_table_content
    neg_tables.append(neg_sample)


item_out={}
item_out["dataset"]="wikisql_"+phase
item_out["question"]=raw_sample['question']
item_out["answer"]=[raw_sample["sql"]]
item_out["positive_ctxs"]=[pos_sample]
item_out["negative_ctxs"]=neg_tables

item_out=json.dumps(item_out)
f.write(item_out)
```

对于 negative_ctxs，在 table 中随机取 20 个，对应训练时的 batch

**（但是不太清楚dpr 原代码的实现中是如论文所说的 in-batch 的训练，即负样本取自 数据集中的 negative_ctxs ，还是同 batch ）

修改dpr 原代码中，标签名载入部分的代码：
```python
class JsonQADataset(Dataset):
...
  def create_passage(ctx: dict):
      return BiEncoderPassage(
          ## 改标签名 by Sunlly
          # normalize_passage(ctx["text"]) if self.normalize else ctx["text"],
          # ctx["title"],
          normalize_passage(ctx["content"]) if self.normalize else ctx["content"],
          ctx["table_id"],
      )
```

运行 train_dense_encoder.py 开始训练模型

问题：
由于table的 token 过长，输出警告：意思是会将超过的 table token截去，只保留前面的

**（论文上也提到了这一点，将 过长的 passage 分成多段，并且训练样本 passage 的长度最好一样）
```
Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy. So the returned list will always be empty even if some tokens have been removed.

```

解决：加入头文件
```
import transformers
transformers.logging.set_verbosity_error()
```
其含义：只报告错误信息，将详细程度设置为ERROR级别。

** 但是，截取表格后，表格内容信息不完整，会不会影响表格性能？（希望模型能学习到如果 question 和内容有token 相同，他们的相似度会更高 这一点）

** 技巧：dpr 通过log 打印了输出日志，效果比 nohup 的好，后续可以学一下。

在wikisql上训练后，用 test 做评估，结果不太理想：

仅在 test 的 pos/neg 样本上做测试，
correct prediction ratio  4541/13216 ~  0.343599

**2. 下载官方的模型检查点，生成 embeddings（将 passage 编码为向量形式）**

由于上面的模型效果不好，决定先用官方训练好的模型先把代码跑通。

下载官方检查点：

```Python
  "checkpoint.retriever.single.nq.bert-base-encoder": {
      "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriever/single/nq/hf_bert_base.cp",
      "original_ext": ".cp",
      "compressed": False,
      "desc": "Biencoder weights trained on NQ data and HF bert-base-uncased model",
  },
```

修改 generate_dense_embeddings.py 中的配置参数：

```Python
def main(cfg: DictConfig):

    ## add args by Sunlly
    # cfg.model_file="/nlp_files/DPR/outputs/2022-06-16/08-59-29/test_nq_20220616/dpr_biencoder.3"
    cfg.model_file="/nlp_files/DPR/model/hf_bert_base.cp"
    # cfg.ctx_src="/nlp_files/DPR/nq-dev-small.json"
    # cfg.ctx_src="/nlp_files/DPR/downloads/data/wikipedia_split/psgs_w100.tsv"
    cfg.ctx_src="dpr_wiki"
    cfg.out_file="/nlp_files/DPR/embeddings/nq"
```

修改 default_dources.yml:

```
dpr_wiki:
  _target_: dpr.data.retriever_data.CsvCtxSrc
  # file: data.wikipedia_split.psgs_w100
  file: "/nlp_files/DPR/downloads/data/wikipedia_split/psgs_w100.tsv"
  id_prefix: 'wiki:'

dpr_nq:
## 用于编码的text 形式
  _target_: dpr.data.retriever_data.CsvQASrc
  # file: /nlp_files/DPR/nq-dev-small.json
  file: /nlp_files/DPR/nq-test.csv
  # id_prefix: 'nq-small:'
```
注意： cfg.ctx_src 不能直接指定路径，需要在 default_dources.yml 中去找。如果是 dpr_wiki 并指定 `file: data.wikipedia_split.psgs_w100` 会自动下载 psgs_w100.tsv（12G）。没有用 nq 的数据集，此处相当于先用官网的例子跑通。

**后续需要按照 tsv 的格式构建 tables 的数据集。

修改了 gen_embs.yaml，用处不大。

修改了原代码中的 end_idx：
```Python
# end_idx = start_idx + shard_size
end_idx=start_idx +200
```
因为只想跑个例子，原数据集的 passage 太多了。所以相当于只取了前 200条生成 ctx 向量。

开始跑代码：
```
python generate_dense_embeddings.py
```

生成的向量结果在 embeddings 中，是个打不开的二进制文件

![](assets/DPR运行笔记-8d05c264.png)

**3. 根据生成好了的 ctx 向量，编码问题做检索**

代码在 dense_retriever.py。

修改配置：

```Python
@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)

## add args by Sunlly
    cfg.model_file="/nlp_files/DPR/model/hf_bert_base.cp"
    cfg.qa_dataset="nq_test"  #/nlp_files/DPR/conf/datasets/retriever_default.yaml
    cfg.ctx_datatsets=["dpr_wiki"] ## need [] is a dict
    cfg.encoded_ctx_files=["/nlp_files/DPR/embeddings/nq_0"] ## need [] is a dict
    cfg.out_file="/nlp_files/DPR/retriever_validation"
```

retriever_default.yaml:
```
nq_test:
  _target_: dpr.data.retriever_data.CsvQASrc
  # file: data.retriever.qas.nq-test
  file: "/nlp_files/DPR/nq-test.csv"
```

遇到问题：

IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)

解决：按：https://github.com/facebookresearch/DPR/issues/213

修改代码：顺利解决

```Python
# max_vector_len = max(q_t.size(1) for q_t in batch_tensors)
# min_vector_len = min(q_t.size(1) for q_t in batch_tensors)
max_vector_len = max(q_t.size(0) for q_t in batch_tensors)
min_vector_len = min(q_t.size(0) for q_t in batch_tensors)
```

run code:


```
python dense_retriever.py
```
结果：

![](assets/DPR运行笔记-7bccd49a.png)

![](assets/DPR运行笔记-213e978d.png)

找到了前 100 个匹配的 passage，由于数据集和 question 其实是对不上的，所以无法评估正确率。

检索的参数具备 score，可以用于后续的 rerank。

可以在 dense_retriever.yml 中设置检索的 passage 数量：

![](assets/DPR运行笔记-bde54c26.png)
