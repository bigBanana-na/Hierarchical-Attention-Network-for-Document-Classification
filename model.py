import tensorflow as tf
from tensorflow.keras import layers

class HAN(tf.keras.Model):
    def __init__(self, vocab_size, num_classes, embedding_size=200, hidden_size=50):
        super(HAN, self).__init__()
        
        # 初始化模型参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.num_classes = num_classes  # 分类数量
        self.embedding_size = embedding_size  # 词嵌入维度
        self.hidden_size = hidden_size  # GRU隐藏层维度

        # 模型构件定义
        # 定义嵌入层，将输入的单词索引转换为嵌入向量
        self.embedding = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size)
        # 定义前向和后向GRU编码器，用于单词级别编码
        # return_sequences=True 表示返回每个时间步的输出
        self.word_encoder_fw = layers.GRU(self.hidden_size, return_sequences=True)
        self.word_encoder_bw = layers.GRU(self.hidden_size, return_sequences=True, go_backwards=True)
        # 定义注意力层，使用全连接层将GRU输出转化为注意力权重
        self.sent_attention = layers.Dense(self.hidden_size * 2, activation='tanh')
        # 定义上下文向量，用于计算注意力权重
        self.u_context = self.add_weight(shape=(self.hidden_size * 2,), initializer='random_normal', trainable=True)
        # 定义前向和后向GRU编码器，用于句子级别编码
        self.sent_encoder_fw = layers.GRU(self.hidden_size, return_sequences=True)
        self.sent_encoder_bw = layers.GRU(self.hidden_size, return_sequences=True, go_backwards=True)
        # 定义注意力层，使用全连接层将GRU输出转化为注意力权重
        self.doc_attention = layers.Dense(self.hidden_size * 2, activation='tanh')
        # 定义分类器，全连接层将文档向量映射到输出标签
        self.classifier = layers.Dense(self.num_classes)

    # 前向传播
    def call(self, inputs):
        # 获取输入的最大句子数和最大句子长度
        max_sentence_num = tf.shape(inputs)[1]
        max_sentence_length = tf.shape(inputs)[2]
        # 词嵌入，将输入的单词索引转换为嵌入向量.神经网络无法直接处理单词，需要将单词转换为向量形式。
        word_embedded = self.embedding(inputs)
        word_embedded = tf.reshape(word_embedded, [-1, max_sentence_length, self.embedding_size])
        # 单词级别编码，使用前向和后向GRU编码每个句子的单词。论文中提到GRU可以捕获句子中单词的上下文信息，效果优于RNN
        word_encoded_fw = self.word_encoder_fw(word_embedded)
        word_encoded_bw = self.word_encoder_bw(word_embedded)
        word_encoded = tf.concat([word_encoded_fw, word_encoded_bw], axis=-1)
        # 单词级别注意力机制，计算每个单词的注意力权重并生成句子向量
        word_attention = tf.nn.softmax(tf.reduce_sum(self.sent_attention(word_encoded) * self.u_context, axis=-1, keepdims=True), axis=1)
        sent_vec = tf.reduce_sum(word_encoded * word_attention, axis=1)
        # 恢复句子向量的形状
        sent_vec = tf.reshape(sent_vec, [-1, max_sentence_num, self.hidden_size * 2])
        # 句子级别编码，使用前向和后向GRU编码每个文档的句子
        sent_encoded_fw = self.sent_encoder_fw(sent_vec)
        sent_encoded_bw = self.sent_encoder_bw(sent_vec)
        sent_encoded = tf.concat([sent_encoded_fw, sent_encoded_bw], axis=-1)
        # 句子级别注意力机制，计算每个句子的注意力权重并生成文档向量
        doc_attention = tf.nn.softmax(tf.reduce_sum(self.doc_attention(sent_encoded) * self.u_context, axis=-1, keepdims=True), axis=1)
        doc_vec = tf.reduce_sum(sent_encoded * doc_attention, axis=1)

        # 分类器，将文档向量映射到输出标签
        out = self.classifier(doc_vec)
        return out