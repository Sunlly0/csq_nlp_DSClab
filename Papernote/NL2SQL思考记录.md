NL2SQL思考记录

1. 针对模型不适合多表的情况，考虑充分利用主外键的约束条件（数据库模式），进行多表的合并操作。

2. Tapas采用端对端的方法，将整个表作为模型的输入，太过于庞大。有没有能够将自然语言和SQL的解析过程合并的方法呢？

3. 当前的数据集都只涉及到查询操作。是否可以增加更新、删除等操作的SQL语句和自然语言做数据集呢？

4. 在数据集方面，考虑跨数据集或合并数据集对模型进行训练，以丰富训练数据集的句式，提升模型的泛化能力。目前已有先从其他数据集进行预训练，再用于评估数据集的迁移学习方式。