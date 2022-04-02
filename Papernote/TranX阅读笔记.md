## TranX论文阅读笔记

2021.7.26

### TranX
论文：《TRANX: A Transition-based Neural Abstract Syntax Parser for Semantic
Parsing and Code Generation》

EMNLP 2018

**论文创新点：**

1. 提出TranX，一个基于转换（transition_based）的神经语义解析器（neural semantic parsing），能实现将自然语言（NL）转换成形式意义表示（meaning representations，MRs）。使用过渡系统，用抽象语义描述语言来表示目标的MR。具备两个主要优点：准确度高，用目标MR的语法来限制输出空间并对信息流进行建模；同时高度通用，只需要编写一个新的抽象语义描述，就能轻易地适用于新的MR形式。

**实验结果：**

将TranX在四种语义分析任务（分别为：GEO、ATIS、DJANGO/python代码生成、WIKISQL/SQL代码生成）中做对比。优于当前的神经网络方法。

---

####背景

语义分析任务（Semantic parsing），将自然语言（NL）转换为形式含义表示（formal meaning representation，MR），MR可以有多种形式，如SQL，robotic commands，smart phone instructions，编程语言（Python和Java）等。由于MR的形式很不一样，一般的神经网络方法都针对一个小范围的任务子集，为了使得语义分析器能在模型结构中，和领域依赖性强的MR的语法匹配。

为了缓解这种情况，已经提出了一些通用目的的神经语义解析器。比如Yin等人提出seq2seq模型，用树的构建过程描述树形的MR的生成。Rabinovich等人提出抽象语义网络（abstract syntax networks,ASNs），将特定领域的MR表示成抽象语义树（AST），由抽象语义描述语言（ASDL）框架来指定。

受以上研究的启发，本文提出了TranX（a TRANsition-based abstract syntaX parser）。

---

####模型

TranX的总流程如图1,以python代码生成作为示例：

![TranX示意图](Paperphoto/tranx_overview.png)

TranX的核心是一个转换系统。

给定自然语言描述x，TranX将其用树的生成动作映射为抽线语义树（AST）z。AST作为中间的含义表达，是对具体领域的MR做的抽象。使用概率模型p(z|x)并通过神经网络参数化，给每个假定的AST进行评分。

语义分析过程中，使用用户定义的ASDL函数，给定了目标MR的特定领域的语法特殊化。对于前面得到的AST z，解析器调用用户给定的函数AST_to_MR(.)，将中介AST转换为MR y，就此分析过程完成。

**ASDL：**由两部分组成：类型(types)和构造函数(constructors)。

**AST：**由多个构造函数复合。树上的每个节点对应构造函数的类型字段（除了根节点）。

![TranX ast](Paperphoto/tranx_ast.png)

**Transition System：**转换系统，将AST的生成过程分解成为树的构造动作。

图2的例子中，从一个单根节点的初始分支AST开始，按照自顶向下，从左到右顺序遍历AST。每一步都会接着以下三种动作之一：ApplyConstr[c]、Reduce、GenToken[v]。直到派生过程中没有frontier字段，生成过程完成。

之后调用用户自定义的AST_to_MR（.）函数，将中间的AST z转换为MR y。

**计算动作概率函数：**使用一个增强递归链接的encoder-decoder神经网络，以反映AST的拓扑结构。

encoder是一个标准的双向LSTM。

decoder也是一个LSTM网络。
