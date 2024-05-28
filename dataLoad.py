import pickle
import numpy as np

# 读取数据集（dataUtils.py中生成），并分成训练集与验证集
def read_dataset():
    with open('yelp_data', 'rb') as f:
        data_x, data_y = pickle.load(f)
        length = len(data_x)
        train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
        train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]
        return train_x, train_y, dev_x, dev_y

# 将文本数据进行批处理，转换为固定形状的三维数组，方便后续的神经网络输入
def batch(inputs):
    if not inputs:
        raise ValueError("Inputs should not be empty")

    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    if not all(sentence_sizes_):
        raise ValueError("All documents should have at least one sentence")
    sentence_size = max(map(max, sentence_sizes_))

    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32) # == PAD

    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, sentence in enumerate(document):
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            for k, word in enumerate(sentence):
                b[i, j, k] = word

    return b, document_sizes, sentence_sizes

# 为不同批次生成随机数据
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = [data[i] for i in shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
