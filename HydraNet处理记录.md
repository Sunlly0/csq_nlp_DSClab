## HydraNet 处理记录

by Sunlly
2022.6.8

论文：《Hybrid Ranking Network for Text-to-SQL》 2020

github代码：https://github.com/lyuqin/HydraNet-WikiSQL

---

### 训练模型

1. 根据运行笔记训练HydraNet模型

2. 结果：达不到论文的准确度，但经过筛选，得到只比论文结果低0.2个点的模型

config：bert-large:
跑了 4 个epoch :
wikidev.jsonl: overall:83.3, agg:91.0, sel:97.6, wn:98.5, wc:95.3, op:99.1, val:97.5
wikitest.jsonl: overall:83.2, agg:91.3, sel:97.4, wn:98.1, wc:94.8, op:99.1, val:97.4

论文结果：
dev 83.5, test 83.4

### 以 前1000个question为例，比较随机打乱表格顺序后的模型预测结果

1. 修改wikisql_gendata.py，仅改了 dataset 中的 column_meta，得到一个问题-random table 的 test集。

2. 运行wikisql_prediction_test_top1000.py（经过修改），对比。

+ 问题1：base_model.py 中，predict_SQL->parse_output->get_span 函数报错：
  out of index。

  跟踪：where condition 预测的 span 超过原问题的长度。只有在 random_table 的数据集上会遇到，原 test 数据集没有遇到。

  解决：对于 get_span 中出错的情况，将返回的 span 直接设置为（0,0）(即没有where 条件)，
  ```
  ## for random dataset, when i>len(segment_ids), error occurs
  if i>len(input_feature.segment_ids)-1:
      return (0,0)
  ```
  parse_output 中对于 vs 和 ve 的 out of index，设置跳过和返回：
  ```
  _, op, vs, ve = conditions[wc]
  ## out of index error:
  if wc>len(input_feature.subword_to_word)-1:
      continue

  if vs>len(input_feature.subword_to_word[wc])-1 or ve>len(input_feature.subword_to_word[wc])-1:
      vs=0
      ve=0
  ```

+ 问题2：base_model.py 中，predict_SQL_with_EG->beam_parse_output->get_span 函数报错：
    out of index。

    由于 返回的 spans 多了一个 sum_logp 参数（不太理解其含义），暂时没有找到一个很好的解决方法。


没有 with_EG 的测试结果：

| origin table   | random_table   |
| -------------- | -------------- |
| sel_acc: 0.934 | sel_acc: 0.934 |
| agg_acc: 0.911 | agg_acc: 0.911 |
| wcn_acc: 0.979 | wcn_acc: 0.893 |
| wcc_acc: 0.918 | wcc_acc: 0.836 |
| wco_acc: 0.976 | wco_acc: 0.89  |
| wcv_acc: 0.938 | wcv_acc: 0.353 |

分析：对于全局的的预测，比如 agg、selnum和wcnum，应该准确率比较高；
从结果对比上看，sel的准确度很高，想不明白？
wcv的准确率很低，可以理解，因为预测的 span 真的不准，有些还超过了 question 的index， 都用（0,0） 代替了，间接导致了 wcn 等的不准确。

分析了一波原因，可能是因为预测的时候只用了列的序号，而没有用列名导致的。在 random_table的setting 中，即使列序号一样，预测出来的SQL 也大概率是不同的。对于 origin table 则不存在这个问题（列号相同，生成的SQL 肯定就对；不同，生成的SQL 肯定不对，因为table是定死了的。）

因此后续考虑在评估的时候，对列名是否相同而计算准确度，而不是列的序号是否相同。

3. 修改wikisql_gendata.py 代码，生成既包含 origin_column 又包含 random_column的 数据集，并且修prediction的代码。

在wikisql_gendata.py 保存 jsonl 的时候，增加origin 的各数据项（origin_table_id,origin_column_meta).

char_to_word：每个 char 在第几个单词中, word_to_char_start：每个单词的开始对应第几个 char，由 tokenizer生成

  修改 wikisql_prediction.py, base_model.py -> print_metric，增加 random_table_file 参数，从 origin_column_meta 、random_column_meta 中提取列名，以str的相等（以前是 列序号相等）作为 acc 的计算条件。

```python
def sel_match_with_column_name(s,r,o, p):
# for s,r,o, p in sp:
    ## 可能预测的列序号超过了表格原长（无法得到列名），此时直接返回 false
    if int(p["sel"])>len(r)-1:
        return False
    return str(r[int(p["sel"])][0]).lower() == str(o[int(s["sel"])][0]).lower()

def wcc_match(a,r,o, b):
    a = sorted(a, key=lambda k: k[0])
    b = sorted(b, key=lambda k: k[0])
    # return [c[0] for c in a] == [c[0] for c in b]
    return [str(r[int(c[0])]).lower() for c in a] == [str(o[int(c[0])]).lower() for c in b]
```
  对于random_table，准确度有明显下降,对于 origin_table，性能不变。
  + origin_table:
  ![](assets/HydraNet处理记录-3e76b825.png)
  + randowm_table:
  ![](assets/HydraNet处理记录-6050b0a2.png)

4. 修改wikisql_evaluate.py 代码，以最后的 SQL 准确度（可以不要求精确的 where condition order）作为 lf_acc , 以执行结果作为 ex_acc

目前 lf 和 ex 的计算均在原表进行，用的列序号：
+ origin_table:
![](assets/HydraNet处理记录-3c725689.png)
+ random_table:
![](assets/HydraNet处理记录-3fd96c83.png)

对于 random_table 来说，准确度是虚高的。基本相当于只有 where condition value 和原来不同。

修改wikisql_evaluate.py，将命令行执行修改成预设文件路径：
```python
# parser = ArgumentParser()
    # parser.add_argument('source_file', help='source file for the prediction')
    # parser.add_argument('db_file', help='source database for the prediction')
    # parser.add_argument('pred_file', help='predictions by the model')
    # parser.add_argument('--ordered', action='store_true', help='whether the exact match should consider the order of conditions')
    # args = parser.parse_args()

    source_file=os.path.join("data_preprocess", "test_1000.jsonl")
    pred_file="output/test_out_top1000.jsonl"
    db_file="WikiSQL/data/test.db"
    random_table_file = os.path.join("data_preprocess", "wikitest_top1000_random.jsonl")
    ordered=False
  ```

增加 random_table 读取；
```python
with open(source_file) as fs, open(pred_file) as fp, open(random_table_file) as fr:
       grades = []
       error_execute_count=0
       ## ls: source_file, lp: pred_file,eg，qg:原版，ep:预测
       for ls, lp ,lr in tqdm(zip(fs, fp, fr), total=count_lines(source_file)):
         eg = json.loads(ls)
         ep = json.loads(lp)
         er =json.loads(lr)
```
修改 query.py，增加 convert_col_index_to_name 方法，保存一个字典 self.col_index_to_name，能将 index 映射成 name。
修改 相等判断条件：从列序号映射成为列名再进行比较：
```python
def __eq__(self, other):
        if isinstance(other, self.__class__):
            ## 新增：将列序号转换为列名
            if len(self.col_index_to_name)>0 and len(other.col_index_to_name)>0:
                indices = str(self.col_index_to_name[self.sel_index]).lower() == str(other.col_index_to_name[other.sel_index]).lower() and self.agg_index == other.agg_index
                if other.ordered:
                    conds = [(str(self.col_index_to_name[col]).lower(), op, str(cond).lower()) for col, op, cond in self.conditions] == [(str(other.col_index_to_name[col]).lower(), op, str(cond).lower()) for col, op, cond in other.conditions]
                else:
                    conds = set([(str(self.col_index_to_name[col]).lower(), op, str(cond).lower()) for col, op, cond in self.conditions]) == set([(str(other.col_index_to_name[col]).lower(), op, str(cond).lower()) for col, op, cond in other.conditions])
            ###
            ## 原代码
            else:
                indices = self.sel_index == other.sel_index and self.agg_index == other.agg_index
                if other.ordered:
                    conds = [(col, op, str(cond).lower()) for col, op, cond in self.conditions] == [(col, op, str(cond).lower()) for col, op, cond in other.conditions]
                else:
                    conds = set([(col, op, str(cond).lower()) for col, op, cond in self.conditions]) == set([(col, op, str(cond).lower()) for col, op, cond in other.conditions])
            return indices and conds
        return NotImplemented
```
修改后，lf 准确度 是实际 SQL 的准确度。

虽然 excute 的代码没有修改，但表的 id 修改后，from ... col_name 也应该是跟着变了，取得random_table 上相应的列。

在 random_table 的生成结果的表格上，有242/1000 个问题无法执行。执行准确度不出意外的低，为0.005。origin_table 的结果不变。

random_table:
![](assets/HydraNet处理记录-bee0c41c.png)

origin_table:
![](assets/HydraNet处理记录-e510de95.png)
