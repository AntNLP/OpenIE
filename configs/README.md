# cfg文件参数设置说明
```
[IO]
# 训练数据文件夹路径
data_dir = ./data
# ckpt文件夹路径
ckpt_dir = ./ckpts/${TASK}/${METHOD}/${PREDICATE_FOR_LEARNING_ARGUMENT}
# 训练集选择
train_dataset = oie2016/01-col
# 开发集选择
dev_dataset = 
# 测试集选择
test_dataset = 
# 预测predicate的策略 solf/gold/default
PREDICATE_FOR_LEARNING_ARGUMENT = soft
# 模型实现方式 pipeline/joint
METHOD = pipeline
# 是否debug debug./'',这么设置是为了后面设置路径方便
DEBUG = debug.
# 选择测试任务
TASK = CaRB
# 训练/开发/测试集文件路径
TRAIN = ./data/corups/${train_dataset}/${DEBUG}train
DEV = ./data/corups/${dev_dataset}/${DEBUG}dev
DEV_GOLD = ./data/corups/${dev_dataset}/${DEBUG}dev.gold
TEST = ./data/corups/${test_dataset}/${DEBUG}dev
TEST_GOLD = ./data/corups/${test_dataset}/${DEBUG}dev.gold
# 训练/开发/测试集临时文件路径，主要用于pipeline模式生成中间数据
TRAIN_TMP = ${ckpt_dir}/${DEBUG}train.tmp
DEV_TMP = ${ckpt_dir}/${DEBUG}dev.tmp
TEST_TMP = ${ckpt_dir}/${DEBUG}test.tmp
# 训练日志
LOG = ${ckpt_dir}/log
# 当前一个模型
LAST = ${ckpt_dir}/last.pt
# dev集表现最好的模型
BEST = ${ckpt_dir}/best.pt
# 预测predicate的模型，主要用于pipeline模式
PRE = ${ckpt_dir}/pre.pt
# 输出以及日志
DEV_OUTPUT = ${ckpt_dir}/dev_output
DEV_OUT = ${ckpt_dir}/dev_out
DEV_ERROR_LOG = ${ckpt_dir}/dev_error_log
TEST_OUTPUT = ${ckpt_dir}/test_output
TEST_OUT = ${ckpt_dir}/test_out
TEST_ERROR_LOG = ${ckpt_dir}/error_log

[Train]
SEED = 42
N_EPOCH = 200
N_BATCH = 128
STEP_UPDATE = 3
STEP_VALID = 1
N_WORKER = 0
IS_RESUME = False
LR = 0.0001
D_MODEL = 768

[NN]
# 选择模型组件
ENCODER = 'bert-bilstm'
DECODER = 'crf'
BERT_TYPE = bert-base-cased
# 分词使用的tokenizer
TOKENIZER = bert-base-cased
# 是否再训练中使用seg tag/gold tag
DEV_SEG_TAG = False
TRAIN_SEG_TAG = False
DEV_GOLD_TAG = False
TRAIN_GOLD_TAG = False
# 保存模型的训练epoch阈值，主要用于debug，正式训练时一般设为0
CKPT_LIMIT = 300
# WARM_UP的比率
WARM_UP_STEPS = 0.1
# 是否读取CHECK_POINT模型
CHECK_POINT = False
# 选择tagset适用的任务
TAG_SET_TYPE = oie2016
# 预测predicate的数量上限
PREDICATE_LIMIT = 3
# seg_tag的数量
SEG_NUM = 3
```