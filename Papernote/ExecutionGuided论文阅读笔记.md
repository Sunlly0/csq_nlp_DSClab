##Execution Guided decoding论文阅读笔记

2021.7.29

###Execution Guided
论文：《Robust Text-to-SQL Generation with Execution-Guided Decoding》

2018

**论文创新点：**

本文引入了一种新机制：执行引导（execution guidance），以利用SQL语义。
对部分生成的程序执行条件处理，达到在decoding过程中检测和排查错误。

该机制可用于任何自回归生成模型（autoregressive generative model），我们在四个先进的循环或基于模板的语义分析模型中演示。

**实验结果**

我们证明了EG模型普遍提高了模型在不同规模和不同复杂度的text-to-SQL数据集：WikiSQL、ATIS、GeoQuery上的性能。

四种模型分别为：
a. 自回归生成模型：Pointer-SQL和带注意力的Seq2Seq
b. 基于模板和槽填充的模型：template-based baseline model和Coarse2Fine。

加入EG的Coarse2Fine在WikiSQL上取得了83.8%的执行准确率。

---

####背景

将NL翻译成可以执行的标准化SQL语言，能帮助非专业用户进行数据库查询。目前，深度学习引入的带注意力seq2seq方法，虽然有效，但常常生成在语法上无法执行的查询。

最近，一些研究尝试用基于语法的seq2tree模型。再次基础上，我们进一步扩展了这个思想展示了如何调整这些模型，以避免各类型的语义错误，即查询运行错误和不生成结果的查询错误（queries that generate no results）。

---

####方法

在SQL语言中，能被执行的部分SQL查询，可以用于指导生成的过程。我们将这个过程成为执行引导（execution guidance），如图1所示：

![eg_overview](Paperphoto/execution_guidance_overview.png)

虽然我们只使用部分程序的执行来过滤不正确的结果，但更高级的执行指导可以用结果来改进决策。执行引导扩展了标准的自回归编码器，在执行结果上增加了附加条件。

本文的EG仅用于过滤运行错误（a runtime error）和无结果的查询。

**Execution error:**

本文将Execution error分为两类：

分析错误（parsing error）：语法上的错误。在自回归模型上出现频繁，而模板或填空方法中不易出现。

执行错误（runtime error）：在执行时抛出错误，比如运算符和操作类型不匹配。这种情况下，无法执行SQL语句，也无法获得正确答案。

如果假设每条查询都有结果域，则可以考虑如下附加错误：

输出为空（empty output）：谓词限制过多，返回空结果，如以c=v为限制条件，但v在C列中并未出现。在实际查询中，往往期望查询输出是非空的。

**以EG作为decoding：**

为了避免上述讨论的错误，将生成过程和SQL执行组件集成到一起。扩展模型，需要选择生成过程的特定阶段执行部分结果，然后用结果来完善剩下的生成过程。

本文给出了标准自回归递归decoder的解码伪代码（pseudocode），可以被看作是标准束搜索（standard beam search）在解码器单元的扩展。

![eg_overview](Paperphoto/execution_guidance_pseudocode.png)

在非自回归模型，比如前馈神经网络，EG可以直接用于最后的解码过程，删除错误的结果程序。

---

####模型实验

**Seq2Seq with Attention：**

在经典的序列到序列模型中，很难确定解码的哪个阶段有一个有效的部分程序。因此，很难静态地实现部分程序执行。因此，使用最简单的执行指导形式: 我们首先执行宽度为 k 的标准波束解码，然后选择在解码结束时没有执行错误排名最高的生成 SQL 程序。

**Pointer-SQL：**

对于聚合符f和列c，判断他们是否兼容。对于c1 op c2，评估他们是否触发类型错误，以及返回空输出。对于没有通过检查的结果，用f',c'和c1' op' c2'的次高概率的联合，替换原来的预测。

**Template-based model:**

选择联合概率最高并不会导致执行错误的程序，模型无法实现部分程序执行。

**Coarse 2 Fine：**

两个阶段：先生成草图（模板）,再填补缺失的槽。

挑选最有可能的草图，再实施下一阶段的解码。EG过程类似pointer-SQL。如果没有可正确执行的结果，回溯挑选粗模型中的别的草图。


---

####总结

执行引导是个简单并有效的工具，可以广泛改进现有的text-to-SQL模型。

但该方法只提高了有语义的程序数量，而没有提升语义正确的程序的数量。