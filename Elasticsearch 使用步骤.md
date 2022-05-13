# Elasticsearch 使用记录

## 介绍

#### ES核心概念

- 近实时（NRT Near RealTime） 写数据时：过1秒才会被搜索到，因为内部在分词、录入索引。 es搜索时：搜索和分析数据需要秒级出结果。

- 集群（Cluster） 包含一个或多个启动着es实例的机器群。通常一台机器起一个es实例。 默认集群名是“elasticsearch”，同一网络，同一集群名下的es实例会自动组成集群。

- 节点（Node） 一个es实例即为一个节点。

  

- 索引（Index） 即拥有相似文档的集合

- 类型（Type） 每个索引里都可以有一个或多个type，type是index中的一个逻辑数据分类，一个type下的document，都有相同的field。

- 文档（Document） es中的最小数据单元。一个document就像数据库中的一条记录。通常以json格式显示。 多个document存储于一个索引（Index）中。

- 映射（Mapping） 定义索引中的字段的名称； 定义字段的数据类型，比如字符串、数字、布尔； 字段，倒排索引的相关配置，比如设置某个字段为不被索引、记录 position 等。

#### 与关系型数据库核心概念对比

| Elasticsearch | 关系型数据库（如Mysql） |
| ------------- | ----------------------- |
| 索引Index     | 数据库Database          |
| 类型Type      | 表Table                 |
| 文档Document  | 数据行Row               |
| 字段Field     | 数据列Column            |
| 映射Mapping   | 约束 Schema             |

## 安装和启动

#### 安装 Elasticsearch

windows 版 安装：

https://www.elastic.co/cn/downloads/elasticsearch

下载 zip后解压

进入解压后的文件夹

启动 bin/elasticsearch.bat

浏览器中输入 http://localhost:9200/ ，可以看到 Rest形式的内容。

**问题：**

1. 遇到 received plaintext http traffic on an https channel, closing connection... 问题

**原因**：是因为默认开启了 [ssl](https://so.csdn.net/so/search?q=ssl&spm=1001.2101.3001.7020) 认证。

**解决：**修改[elasticsearch](https://so.csdn.net/so/search?q=elasticsearch&spm=1001.2101.3001.7020).yml配置文件

将 **xpack.security.enabled **设置为false即可

#### 安装可视化插件 head

下载 zip： https://github.com/mobz/elasticsearch-head

在已经安装好 node.js 的前提下继续操作：

进入 解压后的 head 文件夹，执行

```
npm install
npm run start
```

启动后，在浏览器输入[http://localhost:9100](http://localhost:9100/)

**问题：**

1. 无法访问到es

   **原因**：跨域问题：9100访问不了9200

   **解决：**找到es中config中的elasticsearch.yml文件

   为Various项增加配置：

   ```
   http.cors.enabled: true
   http.cors.allow-origin: "*"
   ```

#### 安装 Kibana

Kibana是一个为ElasticSearch 提供的数据分析的 Web 接口。可使用它对日志进行高效的搜索、可视化、分析等各种操作。

下载 Windows 版：https://www.elastic.co/cn/start

解压 zip

启动 bin/kibana.bat

浏览器输入 http://localhost:5601

左侧菜单栏选择：Management-->Dev Tools

## 使用

### ES查询

> 官方文档search基本用法：https://www.elastic.co/guide/en/elasticsearch/reference/current/search.html

#### 简单查询语法

- GET /<target>/_search
- GET /_search
- POST /<target>/_search
- POST /_search

#### 示例

**PUT:** 增加，修改

在进行添加数据时，我们通常使用PUT 并且指定id

```
PUT twitter/_doc/1  
{
  "user": "GB",
  "uid": 1,
  "city": "Beijing",
  "province": "Beijing",
  "country": "China"
}

```

执行结果: 为 created

第二次执行：结果为 updated

**问题：**

1. PUT 时报错： "error" : "no handler found for uri [/test/doc/1?pretty=true] and method [PUT]

   ```
   PUT test/doc/1
   {
     "name":"hello, Sunlly!"
   }
   ```

**解决：**

使用 POST 和 _doc

```
POST test/_doc/1
{
  "name":"hello, Sunlly!"
}
```



**GET:**

1. 获取索引 twitter 中 id 为1 的数据：

   ```
   GET twitter/_doc/1/
   ```
   执行结果：
   
   ```json
   {
     "_index" : "twitter",
     "_id" : "1",
     "_version" : 2,
     "_seq_no" : 1,
     "_primary_term" : 1,
     "found" : true,
     "_source" : {
       "user" : "GB",
       "uid" : 1,
       "city" : "Beijing",
       "province" : "Beijing",
       "country" : "China"
     }
   }
   ```

2. 获取 user 部分的数据:

   ```
   GET twitter/_doc/1?_source=user
   ```

   执行结果：

   ```json
   {
     "_index" : "twitter",
     "_id" : "1",
     "_version" : 2,
     "_seq_no" : 1,
     "_primary_term" : 1,
     "found" : true,
     "_source" : {
       "user" : "GB"
     }
   }
   
   ```

   

**POST：** 更新
在进行添加数据时，如果想要id自动增长，那么我们需要使用POST

1. POST 增加数据：

   ```json
   POST twitter/_doc
   {
     "user": "GB",
     "uid": 1,
     "city": "Beijing",
     "province": "Beijing",
     "country": "China"
   }
   ```

   执行结果：可以看到 id 是随机的

   ```json
   {
     "_index" : "twitter",
     "_id" : "AwsSmYABdFxCduRXgJke",
     "_version" : 1,
     "result" : "created",
     "_shards" : {
       "total" : 2,
       "successful" : 1,
       "failed" : 0
     },
     "_seq_no" : 2,
     "_primary_term" : 1
   }
   ```

2. 局部更新，需要加上 _update，语法： POST INDEX/_update/ID

   ```
   POST twitter/_update/1
   {
     "doc": {
       "city":"成都"
     }
   }
   ```



**DELETE** 删除一个文档 DELETE INDEX/_doc/ID

```
DELETE twitter/_doc/5
```



**批处理 _bulk** ：可以通过很多请求封装成一个请求进行批量处理,提高执行效率，注意 payload 不能过长，控制在5M~15M左右



### 导入 json:

参考了官网教程 https://www.elastic.co/guide/cn/kibana/current/tutorial-load-dataset.html，下载 account数据集。

但是 windows 上用

```
curl -H 'Content-Type: application/x-ndjson' -XPOST 'localhost:9200/bank/account/_bulk?pretty' --data-binary @accounts.json
```

会报错

最后是从 Kibana的 Observability-->upload file 导入的



2. 加载json 数据时报错： curl: (6) Could not resolve host: 'localhost

   ```
   curl -H 'Content-Type:application/x-ndjson' -XPOST 'localhost:9200/bank/account/_bulk?pretty' --data-binary @accounts.json
   ```

   解决： 去除空格，用双引号，不要用单引号，避免参数转义问题。





**问题：** 加载数据失败：Error creating index {"name":"TimeoutError","……

原因：不要用 jsonl格式上传，改扩展名为 json





### WikiSQL的简单处理和检索

1. 用 python 中的jsonline包，读取原文件并在写入时增加 index行

2. 将 header 和 rows 行拼接成为 content 字段，作为表格内容相似度判断的依据

3. 写入新文件 test_write.jsonl

4. 改成 json 格式，经过 observation-->upload file 导入 ElasticSearch. index 名字为 wikisqltest

5. 查找：

   ```
   GET /wikisqltest/_search
   {
       "query": {
           "match": {
                "id": "1-10015132-1"
            }
         }
   }
   ```

   6. 修改 setting 配置中，相似检索度为BM25。需要在关闭索引的时候操作

      ```
      POST wikisqltest/_close
      ```

      ```
      PUT wikisqltest/_settings
      {
        "index" : {
            "similarity" : {
              "default" : {
                "type" : "BM25"
              }
            }
        }
      }
      ```

      ```
      POST wikisqltest/_open
      ```

   7. 相似度检索：

      选一条问题：

      ```json
      {
          "phase": 1,
          "table_id": "1-1000181-1",
          "question": "What is the current series where the new series began in June 2011?",
          "sql": {
              "sel": 4,
              "conds": [
                  [5, 0, "New series began in June 2011"]
              ],
              "agg": 0
          }
      }
      ```

      检索：

      ```
      GET /wikisqltest/_search
      {
          "query": {
              "match": {
                   "content": "What is the current series where the new series began in June 2011?"
               }
            }
      }
      ```

      执行结果：

      ```json
      {
        "took" : 1,
        "timed_out" : false,
        "_shards" : {
          "total" : 1,
          "successful" : 1,
          "skipped" : 0,
          "failed" : 0
        },
        "hits" : {
          "total" : {
            "value" : 16,
            "relation" : "eq"
          },
          "max_score" : 27.699503,
          "hits" : [
            {
              "_index" : "wikisqltest",
              "_id" : "GLDWnYABkb_0lK1sWGUX",
              "_score" : 27.699503,
              "_source" : {
                "id" : "1-1000181-1",
                ...
              }
            },
            {
              "_index" : "wikisqltest",
              "_id" : "NLDWnYABkb_0lK1sWGUX",
              "_score" : 6.29582,
              "_source" : {
                "id" : "1-10054296-1",
                ...
              }
            },
            {
              "_index" : "wikisqltest",
              "_id" : "HLDWnYABkb_0lK1sWGUX",
              "_score" : 5.7653437,
              "_source" : {
                "id" : "1-10007452-3",
                ...
            },
          ]
        }
      }
      
      ```

      可以看到该问题 What is the current series where the new series began in June 2011? 检索时，所有表格中 BM 25 相似度分数最高的就是表 id: 1-1000181-1， 和 groundtruth 相同。
      
      8. 修改默认配置
      
         修改b=0.72
      
         ```
         POST wikisql_train/_close
         
         PUT wikisql_train/_settings
         {
           "index" : {
               "similarity" : {
                 "default" : {
                   "type" : "BM25",
                   "b":0.72
                 }
               }
           }
         }
         
         POST wikisql_train/_open
         ```
      
         修改查找返回结果的个数：调整 size=20，默认为10
      
         ```
         GET /wikisql_train/_search
         {
           "size":20,
             "query": {
                 "match": {
                      "content": "What is the current series where the new series began in June 2011?"
                  }
               }
         }
         ```
      
         
   

## python 中使用 elasticsearch

安装 elasticsearch 包：

```
pip install elasticsearch
```

使用

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(hosts="http://localhost:9200")

print(es.indices.exists(index="wikisqltest"))
```

执行结果：True

检索相似度：

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(hosts="http://localhost:9200")
print(es.indices.exists(index="wikisqltest"))

query={
    "match": {
         "content": "What is the current series where the new series began in June 2011?"
     }
}

res=es.search(index='wikisqltest',query=query)
print(res)
```

清空索引数据：

```
POST wikisql_train/_delete_by_query
{
  "query": {"match_all": {}}
}
```

修改查找返回结果的个数：调整 size=20，默认为10:

```
top_k = 20
res = es.search(index=index, size=top_k, query=query)
```

#### 自动化实现

自动修改索引设置：bm25 中的k1和b值，同时遍历list中的预设值，打印所有组合的预测结果

```
top_ks=[1,5,10,20,30,50,100,150,200] #8个
b_list=[0.6,0.7,0.72,0.75,0.8,0.9,1] #7个
k1_list=[20,10,5,2,1.5,1.2,1,0.9,0.75,0.5,0.4,0.3,0.2,0.1,0] #15 个
```

```python
def test_differ_k1_and_b():
    # k1= 1.2
    for k1 in k1_list:
        for b in b_list:
            es.indices.close(index=index)
            es.indices.put_settings(index=index, settings={
                "similarity" : {
                    "default" : {
                        "type" : "BM25",
                        "b":b,
                        "k1":k1
                    }
                }
            })
            es.indices.open(index=index)
            test_differ_top_k_in_dataset(b,k1)
    return

def test_differ_top_k_in_dataset(b,k1):
    for top_k in top_ks:
        test_in_dataset(top_k,b,k1)
    return
```

将结果以追加写的形式保存在`total_result.txt` 中

```python
    total_result_file = open(total_result_filename, "a+",encoding='UTF-8')
    print("bm25_test: k1=",k1,", b=",b,", top_k=",top_k,", hit_accuracy=", count / total,file=total_result_file)
```



## 使用 excel 做简单的结果分析

打印的结果如下：

```
bm25_test: k1= 20 , b= 0.6 , top_k= 1 , hit_accuracy= 0.18188688751731957
bm25_test: k1= 20 , b= 0.6 , top_k= 5 , hit_accuracy= 0.3454465297896461
bm25_test: k1= 20 , b= 0.6 , top_k= 10 , hit_accuracy= 0.41888147121803754
bm25_test: k1= 20 , b= 0.6 , top_k= 20 , hit_accuracy= 0.49817357349792163
bm25_test: k1= 20 , b= 0.6 , top_k= 30 , hit_accuracy= 0.5533442499055297
bm25_test: k1= 20 , b= 0.6 , top_k= 50 , hit_accuracy= 0.6269051517823403
bm25_test: k1= 20 , b= 0.6 , top_k= 100 , hit_accuracy= 0.7245874795314271
bm25_test: k1= 20 , b= 0.6 , top_k= 150 , hit_accuracy= 0.7898979720367805
bm25_test: k1= 20 , b= 0.6 , top_k= 200 , hit_accuracy= 0.8295125330646177
bm25_test: k1= 20 , b= 0.7 , top_k= 1 , hit_accuracy= 0.20254440105806776
bm25_test: k1= 20 , b= 0.7 , top_k= 5 , hit_accuracy= 0.36900113364403575
bm25_test: k1= 20 , b= 0.7 , top_k= 10 , hit_accuracy= 0.4444514422471344
bm25_test: k1= 20 , b= 0.7 , top_k= 20 , hit_accuracy= 0.521476256455473
bm25_test: k1= 20 , b= 0.7 , top_k= 30 , hit_accuracy= 0.5755762690515178
bm25_test: k1= 20 , b= 0.7 , top_k= 50 , hit_accuracy= 0.6435319309736742
bm25_test: k1= 20 , b= 0.7 , top_k= 100 , hit_accuracy= 0.7362388210102028
bm25_test: k1= 20 , b= 0.7 , top_k= 150 , hit_accuracy= 0.7984632825292858
bm25_test: k1= 20 , b= 0.7 , top_k= 200 , hit_accuracy= 0.8353067136919008
bm25_test: k1= 20 , b= 0.72 , top_k= 1 , hit_accuracy= 0.20651215518327246
bm25_test: k1= 20 , b= 0.72 , top_k= 5 , hit_accuracy= 0.3742284922534324
...
```

使用 excel 对结果进行简单的分析，找到一个比较合适的 bm25 (k1，b） 组合。

1. 载入excel: 菜单-->数据-->从文本/csv导入，选择分隔符：自定义、数据类型检测：不检测数据类型

2. 删除多余的列，只保留 k1,b,top_k，accuracy 这4列。

3. 使用 `方方格子` 工具，将 accuracy内容选中，合并转换-->行列转换-->每行为 9 列（根据 top_k 的数量来）

4. 将k1,b 内容复制到另一个工作簿，使用：方方格子-->批量删除-->间隔删除行/列，删除行数为 8 (9-1)，间隔行数为1，工具可以实现预览功能

5. 拼接结果，删除多余的 top_k、accuracy 列和多余的行，得到处理后的表格

   ![img](file:///D:\TIM\593086079\Image\C2C\}5XLS1YWP2CMFYFIBR%_5DO.png)

结论：

1. 通过观察，较大的 b 对于 top50 的准确率提升有较好的作用。较低的 k1和适中的 b时 top200 的准确度较高。对默认的数值做修改对结果有一定的提升。
2. 对比 train、dev、test，当表格数越少时(dev)，bm25 运行的效果越好。