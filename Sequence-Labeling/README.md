# Sequence Labeling

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
