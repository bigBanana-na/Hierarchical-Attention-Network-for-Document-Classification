import json
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

# # punkt资源下载,punkt的主要作用是通过识别文本中的标点符号和其他上下文线索，将连续的文本分割成独立的句子。
# nltk.download('punkt')

# 使用nltk默认分词分句器对词句进行拆分
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()
# 新建字典，用于记录每个单词及其出现的频率
word_freq = defaultdict(int)

# 读取json数据
with open("yelp_academic_dataset_review.json", 'rb') as f:
    # nltk分词分句器
    for line in f:
        review = json.loads(line)
        words = word_tokenizer.tokenize(review['text'])
        for word in words:
            # print(word)
            word_freq[word] += 1
    print("File successfully loaded and processed")

# 新建文件用于保存词频表
with open('word_freq.pickle', 'wb') as g:
    # 保存词频表
    pickle.dump(word_freq, g)
    print (len(word_freq))
    print ("word_freq save finished")

# 将词频排序
sort_words = list(sorted(word_freq.items(), key=lambda x:-x[1]))
# 查看最高频词与最低频词
#print (sort_words[:10], sort_words[-10:])

# 构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW。保留所有出现次数大于5次的高频词，
# 为每个高频词用i分配一个索引值，最后的i能代表一共有多少个高频词
vocab = {}
i = 1
vocab['UNKNOW_TOKEN'] = 0
for word, freq in word_freq.items():
    if freq > 5:
        vocab[word] = i
        i += 1
#print (i)

# UNKNOWN表示这是低频词
UNKNOWN = 0
data_x = []
data_y = []

# 将所有的评论文件都转化为30*30的索引矩阵（索引到vocab里面的词语），也就是每篇都有30个句子，每个句子有30个单词
# 不够的补零，多余的删除，并保存到最终的数据集文件之中
max_sent_in_doc = 30
max_word_in_sent = 30
num_classes = 5
with open('yelp_academic_dataset_review.json', 'rb') as f:
    for line in f:
        doc = []
        review = json.loads(line)
        sents = sent_tokenizer.tokenize(review['text'])
        # 遍历所有句子
        for i, sent in enumerate(sents):
            # 最多处理30个句子
            if i < max_sent_in_doc:
                word_to_index = []
                for j, word in enumerate(word_tokenizer.tokenize(sent)):
                    # 最多处理30个单词
                    if j < max_word_in_sent:
                        word_to_index.append(vocab.get(word, UNKNOWN))
                doc.append(word_to_index)
        # 将评分转换成独热标签
        label = int(review['stars'])
        # 创建了一个长度为 num_classes（这里为5）的列表，所有元素都初始化为0
        labels = [0] * num_classes
        #print(labels)
        labels[label-1] = 1
        #print(labels)
        data_y.append(labels)
        data_x.append(doc)

    # 保存词索引矩阵与对应矩阵对应的标签
    pickle.dump((data_x, data_y), open('yelp_data', 'wb'))
    print (len(data_x)) 
    # length = len(data_x)
    # train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
    # train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]

#todo:  动态batch获取每个样本的大小