# Sequence Labeling
## 模型结构
![模型结构](/img/model_structure_sequence_labeling.png)
## 训练
```
cd src
./run_pipeline or ./run_joint
```

## 测试

```
cd src
./eval
```

## cfg文件

cfg文件位于cfgs文件夹中，用于设置不同的训练任务和超参数设置，在调用eval.sh和run_x.sh时使用不同的cfg文件即可运行不同的任务。

## 数据集
数据集都位于data文件夹下。
#### 训练集
训练集使用的是oie2016，其格式如下：
```
words       pre     ext1    ext2
Courtaulds  O       A0-B    O
'           O       A0-I    O
spinoff     O       A0-I    O
reflects    P-B     P-B     O
pressure    O       A1-B    O
on          O       A1-I    O
British     O       A1-I    A1-B
industry    O       A1-I    A1-I
to          O       O       O
boost       P-B     O       P-B
share       O       O       0-B
prices      O       O       0-I
beyond      O       O       A2-B
the         O       O       A2-I
reach       O       O       A2-I
of          O       O       A2-I
corporate   O       O       A2-I
raiders     O       O       A2-I
.           O       O       O
```
第一列为文本，第二列句子中蕴含的所有relation，第三列之后每一列都对应一个extraction的序列标注。

#### 开发集
使用的是CaRB的开发集，其格式如下：
```
32.7 % of all households were made up of individuals and 15.7 % had someone living alone who was 65 years of age or older .	were made up of	32.7 % of all households	individuals
```
是 sentence \t realtion \t argument1 \t argument2 \t argument3.... 的形式
#### 测试集
使用的是CaRB的测试集，其格式同上

### result

|   模型结构 | 训练集    | 测试集 | P | R | F1|
|   ----    | ----      | ---- | ---- | ---- | ----|
|   joint   | oie2016   | CaRB | 0.449 | 0.319 | 0.373|
|   joint   | oie2016   | CaRB | 0.449 | 0.319 | 0.373|
|joint_gold|oie2016|CaRB|0.439|0.345|0.386|
|joint_soft|oie2016|CaRB|0.479|0.343|0.400|
|pipeline|oie2016|CaRB|0.486|0.339|0.399|
|pipeline_gold|oie2016|CaRB|0.460|0.315|0.374|
|pipeline_soft|oie2016|CaRB|0.424|0.294|0.347|	