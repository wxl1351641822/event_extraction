# 1 任务

* id\t篇章\t事件\t主体
* 困难
	* 多事件：一个篇章可能多个事件
		* 大多在1-4之间，>5的仅有一个
	* 多实体：一个事件可能有多个主体
	* 重叠实体：一个实体可能参与多个事件

# 2 解决方案
## 2.1 序列标注
感觉可以端到端解决这个问题，但是却不work？
### 2.1.1 paddlehub:robert+crf
* 采用chunk_eval的f1，precision,recall
* 下列策略下结果都差不多……
* train 0.8,dev 0.8,test 0.01??
	* test完全不行，没有一个完全正确的实体，但每次预测的结果都在正确结果附近？
	* 我没有test的答案，但一眼就能看出来他完全不对……
* 可能原因：
	* train的实体集合的22%在dev的实体集合占比60%
	* --它可能记住了？？
	* 可以做的：去重叠，降低train的实体在dev的占比
	
#### 2.1.1.1 多次复制句子
* paddlehub
	* 输入的句子裁剪为100后，复制事件数次。
		* 裁剪中去头、去尾，保证其内实体数目最多
		* 复制：为保证能预测多组不同的结果
	* 结果与不复制差别不大……
#### 2.1.1.2 paddlehub_maxlen
* maxlen=202
* 不复制
* 不考虑重叠事件
#### 2.1.1.3 paddlehub_maxlen_nocrf
* maxlen=202
* 不复制
* 不考虑重叠事件
* 无crf，换普通fc
### 2.1.2 CopyMTL
参考CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning

[Paper](https://arxiv.org/abs/1911.10438) accepted by AAAI-2020 

* 用于多关系、重叠实体的关系抽取任务
* 输出：关系1，实体1的位置，实体2的位置，关系2，实体3的位置，实体4的位置
	* 多余的补[<eos>,max_seq_len,max_seq_len]
* 单词实体
* 每次对所有词计算copy概率，
* 编码层：双向rnn
* attention：上轮decoder_state和encoder输出-->context
* [context,上一次预测实体/关系的嵌入]
* rnn-->这时刻的结果
* --编码器可以是一个也可以是多个

* 这里是中文数据集，对于实体的识别，也难以使用词向量进行。
* 问题：
	* 随机的字嵌入可能不大能够表达？
#### 2.1.2.1 copy-->多次序列标注：CCKS（model里）
* 改变其输出为：事件1，序列标注1，事件2，序列标注2
* 序列标注的长度：最大句子长度
* 可能存在的问题：
	* 事件的顺序？
* 遇到的问题：
	* 全部预测为O
	* 疑问
##### 2.1.2.2 model_crf
* 增加crf，看能否解决全为O的问题
	* 目前问题：loss变成了负的了？
## 2.2 MRC阅读理解
### 2.2.1 CopyMTL-->阅读理解
####2.2.1.1 实体1为主体开始位置，2为实体结束为止
* 事件1，主体1beg,主体2beg。
