# HAN Text Classification

本项目是学习 [https://github.com/lc222/HAN-text-classification-tf/tree/master](https://github.com/lc222/HAN-text-classification-tf/tree/master) 项目的仓库，主要用于文本分类任务，基于 Yelp 数据集。

## 项目结构

- **dataUtils.py**：用于 Yelp 数据集的预处理，生成词频表。将每个评论中的词语转换成词频表对应的索引，标签为每个评论对应的 star 值（评分值）。
- **dataLoad.py**：用于读取预处理后的数据，并生成批次数据供训练使用。
- **model.py**：定义 Hierarchical Attention Network（HAN）模型。
- **train.py**：用于模型训练。
- **runs**：用于存放训练的保存点

## 使用方法

1. **下载数据集**
   下载链接：https://github.com/rekiksab/Yelp/tree/master/yelp_challenge/yelp_phoenix_academic_dataset

2. **预处理数据**：
    ```bash
    python dataUtils.py
    ```

3. **训练模型**：
    ```bash
    python train.py
    ```

## 依赖项

- TensorFlow
- NumPy
- NLTK

## 参考

- [HAN-text-classification-tf](https://github.com/lc222/HAN-text-classification-tf/tree/master)
