# OpenIE

## Sequence Labeling

### 训练
```
cd src/
./run_pipeline.sh or ./run_joint.sh
```

### 测试
```
cd src/
./eval.sh
```

### cfg文件
通过修改cfg文件来执行不同的训练任务和设置超参数，在.sh文件中选择对应的cfg文件。cfg文件位于/cfgs文件夹中

### 数据集

训练数据集使用的是oie2016的数据集，其数据格式是：
```
word        pre     ext1    ext2
QVC	        O	    A0-B	A1-B
Network     O	    A0-I	A1-I
Inc.	    O	    A0-I	A1-I
said	    P-B	    P-B	    O 
it	        O	    A1-B	O
completed	P-B	    A1-I	P-B
its	        O	    A1-I	A0-B
acquisition	O	    A1-I	A0-I
of	        O	    A1-I	A0-I
CVN	        O	    A1-I	A0-I
Cos.	    O	    A1-I	A0-I
for	        O	    A1-I	O
about	    O	    A1-I	A2-B
$	        O	    A1-I	A2-I
423	        O	    A1-I	A2-I
million	    O	    A1-I	A2-I
.	        O	     A1-I	O
```

其中第一列为文本，第二列为句子中所包含的所有relation，后面每一列均为一组extraction对应的label序列

