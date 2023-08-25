import os
import pickle   # 实现对象的序列化和反序列化。它提供了一种将对象转换为字节流的方法，以便在存储或传输时进行持久化或重新创建对象。

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence # 处理文本数据中的批量序列，例如将一批句子转换为张量形式进行输入。填充后的张量可以直接用于神经网络的输入，方便进行批处理操作和并行计算。
from torch.utils.data import DataLoader, Dataset    # DataLoader从自定义的数据集中加载数据，并支持批处理、数据打乱、多线程加载等功能。可以方便地将数据集按照设定的批次大小拆分成小批量进行训练。Dataset提供了一些必要的接口方法，包括 __len__ 和 __getitem__，用于获取数据集的长度和索引对应的数据样本。
# 通过导入 DataLoader 和 Dataset，可以使用 PyTorch 提供的数据加载和处理功能，方便地构建和处理自定义的数据集，并进行批量加载和训练。
from Utils import whether_type_str  # 判断是否字符串类型
from Config import Data_path_SDK    # 把数据路径引入

# MOSI Structure
mosi_l_features = ["text", "glove", "last_hidden_state", "masked_last_hidden_state", "pooler_output", "summed_last_four_states"]
# MOSI文本特征包括文本，使用GloVe（Global Vectors for Word Representation）生成词向量算法生成的文本特征，BERT等模型的最后一个隐藏状态，BERT等模型在进行掩码处理后的最后一个隐藏状态，BERT等模型的池化输出，BERT等模型最后四个隐藏状态的和。
mosi_a_features = ["covarep", "opensmile_eb10", "opensmile_is09"]
# MOSI音频特征包括音频特征提取的covarep表示方法，使用OpenSmile工具包提取的第一种音频特征，使用OpenSmile工具包提取的第二种音频特征。
mosi_v_features = ["facet41", "facet42", "openface"]
# MOSI视频特征包括FACET算法提取的facet41表示方法，facet42表示方法，使用OpenFace工具包提取的视频特征。
# [[l_features, a_features, v_features], _label, _label_2, _label_7, segment] 数据集结构是列表，每个元素包含[文本特征，音频特征和视频特征]，标签，二分类标签，七分类标签，数据片段。
    
# MOSEI Structure 与MOSI结构类似
mosei_l_features = ["text", "glove", "last_hidden_state", "masked_last_hidden_state", "pooler_output", "summed_last_four_states"]
mosei_a_features = ["covarep"]
mosei_v_features = ["facet42"]
# [[l_features, a_features, v_features], _label, _label_2, _label_7, segment]

# POM Structure 与MOSI结构类似
pom_l_features = ["text", "glove", "last_hidden_state", "masked_last_hidden_state", "pooler_output", "summed_last_four_states"]
pom_a_features = ["covarep"]
pom_v_features = ["facet42"]
# [[l_features, a_features, v_features], _label, _label_7, segment]

DATA_PATH = Data_path_SDK   # 建立数据路径

def multi_collate_mosei_mosi(batch):    # 数据批处理，返回处理后的数据
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)    # 对批次样本排序，按每个样本第一个特征长度降序排，方便后续填充，将较长样本放在前边，这样有效填充和处理。
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.Tensor([sample[3] for sample in batch]).reshape(-1,).float() # 从排序后批次取出标签，将其转换为浮点张量，labels是一维张量，使用.reshape(-1,)的目的是将它们转换为形状为(batch_size,)的张量，其中batch_size是当前批次的样本数量。这里的标签数据在数据集结构的第四个位置（索引为3）
    labels_2 = torch.Tensor([sample[4] for sample in batch]).reshape(-1,).long()    # 从排序后批次取出标签，将其转换为长整形张量，labels_2是一维张量，使用.reshape(-1,)的目的是将它们转换为形状为(batch_size,)的张量，其中batch_size是当前批次的样本数量。这里的标签数据在数据集结构的第五个位置（索引为4）
    if whether_type_str(batch[0][0][0]):    # 检查样本第一个特征是否是字符串，即是否为文本信息
        sentences = [sample[0].tolist() for sample in batch]    # 如果是，则把文本转换为列表，并储存在sentences中
    else:   # 将批次中的文本数据进行填充，使得每个样本的文本特征具有相同的长度。即：将每个样本的句子部分提取出来，并将其转换为 torch.FloatTensor 类型的张量，每个张量表示一个样本的句子序列。使用 pad_sequence 函数对这个句子张量列表进行填充使得它们具有相同的长度，以适应最长的句子长度。填充的值为0。
        sentences = pad_sequence([torch.FloatTensor(sample[0]) for sample in batch], padding_value=0).transpose(0, 1)   # 最后，通过转置把输入维度顺序从(batch_size, sequence_length)调整到(sequence_length, batch_size)
    acoustic = pad_sequence([torch.FloatTensor(sample[1]) for sample in batch], padding_value=0).transpose(0, 1)    # 将批次中的音频特征进行填充，使得每个样本的音频特征具有相同的长度。使用pad_sequence函数将每个样本的音频特征转换为浮点型的张量，并填充0。转置，使得维度顺序变为(sequence_length, batch_size)。
    visual = pad_sequence([torch.FloatTensor(sample[2]) for sample in batch], padding_value=0).transpose(0, 1)  # 将批次中的声音特征进行填充，使得每个样本的音频特征具有相同的长度。使用pad_sequence函数将每个样本的音频特征转换为浮点型的张量，并填充0。转置，使得维度顺序变为(sequence_length, batch_size)。
        
    lengths = torch.LongTensor([sample[0].shape[0] for sample in batch])    # 遍历批次中的每个样本 sample，并获取其句子部分 sample[0] 的长度。这样就得到了一个列表，其中每个元素表示对应样本的句子长度。并转换为长整型的张量放入lengths中。
    return sentences, acoustic, visual, labels, labels_2, lengths   # 返回处理后的数据，包括填充后的文本特征(sentences)、音频特征(acoustic)、视频特征(visual)、标签(labels)、另一种标签(labels_2)和文本长度(lengths)。

# 获取MOSEI数据集的特征和标签。它根据给定的模式（训练集、验证集或测试集）以及选择的文本、音频和视频特征类型，从预先处理的数据文件中加载相应的数据。
def get_mosi_dataset(mode='train', text='glove', audio='covarep', video='facet42', normalize=[True, True, True]):   # 训练模式，指定训练集；使用glove方式加载文本特征类型；使用covarep指定音频特征类型；使用facet42指定视频特征类型；并将对相应的三个特征进行最大最小归一化，使其值在[-1, 1]范围内，预先设置三个模态是否归一化选择都为TRUE。
    with open(os.path.join(DATA_PATH, 'mosi_'+mode+'.pkl'), 'rb') as f: # 打开MOSEI的pkl文件，根据给定模式（训练，验证和测试）而变化，使用rb二进制读取文件。
        data = pickle.load(f)   # 从pkl中加载数据，储存进data中。
        
    assert text in mosi_l_features  # 确保选择的文本特征类型在MOSEI数据集支持的特征列表mosi_l_features中。如果不在列表中，将会抛出AssertionError异常。
    assert audio in mosi_a_features # 同理
    assert video in mosi_v_features # 同理
    # 遍历data中的每个元素data_,处理后的特征数据储存在l/a/v_feature列表中。处理方式是选取data_列表第一个元素的第一个子元素的text/audio/video位置索引，获取data_列表第一个元素的第一个子元素的数据特征，使用np中的nan_to_num函数将其中的NaN值替换为0.0，正无穷和负无穷都替换为0.
    l_features = [np.nan_to_num(data_[0][0][mosi_l_features.index(text)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    a_features = [np.nan_to_num(data_[0][1][mosi_a_features.index(audio)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    v_features = [np.nan_to_num(data_[0][2][mosi_v_features.index(video)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    # 对三个模态进行归一化处理
    if normalize[0]:
        max_l, min_l = max([np.max(f) for f in l_features]), min([np.min(f) for f in l_features])   # 使用列表推导式和np函数计算特征数据l_features中的最大值max_l和最小值min_l，比如[np.max(f) for f in l_features]使用列表推导式获取l_features中每个元素的最大值，再取max就是整个特征数据l_features中的最大值了。最小值同理。
        l_features = [2*(f-min_l)/(max_l-min_l)-1 for f in l_features]  # 归一化处理。遍历l_features的所有元素f，使用这个公式将每个元素通过减去最小值，除以最大值与最小值之差的一半，然后乘以2，最后减去1，以实现将每个元素归一化到范围[-1, 1]，并储存回l_features列表中。
    if normalize[1]:    # 同理
        max_a, min_a = max([np.max(f) for f in a_features]), min([np.min(f) for f in a_features])
        a_features = [2*(f-min_a)/(max_a-min_a)-1 for f in a_features]
    if normalize[2]:    # 同理
        max_v, min_v = max([np.max(f) for f in v_features]), min([np.min(f) for f in v_features])
        v_features = [2*(f-min_v)/(max_v-min_v)-1 for f in v_features]
        
    labels = [data_[1] for data_ in data]   # 遍历data，每次记为data_，将data_[1]标签数据拿出来，整合成一个列表
    labels_2 = [data_[2] for data_ in data] # 遍历data，每次记为data_，将data_[2]标签数据拿出来，整合成一个列表
    
    return l_features, a_features, v_features, labels, labels_2

# MOSEI数据集处理方式和MOSI一样
def get_mosei_dataset(mode='train', text='glove', audio='covarep', video='facet42', normalize=[True, True, True]):
    with open(os.path.join(DATA_PATH, 'mosei_'+mode+'.pkl'), 'rb') as f:
        data = pickle.load(f)
        
    assert text in mosei_l_features  
    assert audio in mosei_a_features  
    assert video in mosei_v_features
    l_features = [np.nan_to_num(data_[0][0][mosei_l_features.index(text)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    a_features = [np.nan_to_num(data_[0][1][mosei_a_features.index(audio)], nan=0.0, posinf=0, neginf=0) for data_ in data]
    v_features = [np.nan_to_num(data_[0][2][mosei_v_features.index(video)], nan=0.0, posinf=0, neginf=0) for data_ in data]

    if normalize[0]:
        max_l, min_l = max([np.max(f) for f in l_features]), min([np.min(f) for f in l_features])
        l_features = [2*(f-min_l)/(max_l-min_l)-1 for f in l_features]
    if normalize[1]:
        max_a, min_a = max([np.max(f) for f in a_features]), min([np.min(f) for f in a_features])
        a_features = [2*(f-min_a)/(max_a-min_a)-1 for f in a_features]
    if normalize[2]:
        max_v, min_v = max([np.max(f) for f in v_features]), min([np.min(f) for f in v_features])
        v_features = [2*(f-min_v)/(max_v-min_v)-1 for f in v_features]
        
    labels = [data_[1] for data_ in data]
    labels_2 = [data_[2] for data_ in data]
    labels_7 = [data_[3] for data_ in data] # 和之前类似，遍历data列表中的每个元素data_，并将data_[3]（第四个元素，即标签数据）添加到labels_7列表中。结果是将每个样本的标签数据（7个类别）提取出来，存储在labels_7列表中。
    
    return l_features, a_features, v_features, labels, labels_2, labels_7

class CMUSDKDataset(Dataset):   # 定义了一个名为CMUSDKDataset的类，它是torch.utils.data.Dataset的子类，用于表示CMUSDK数据集
    def __init__(self, mode, dataset='mosi', text='glove', audio='covarep', video='facet42', normalize=[True, True, True]): # 构造函数，接受多个参数，包括mode、dataset、text、audio、video和normalize。
        assert mode in ['test', 'train', 'valid']   #对mode参数进行断言验证，确保它们的取值符合预期。
        assert dataset in ['mosei', 'mosi', 'pom']  #对dataset参数进行断言验证，确保它们的取值符合预期。

        self.dataset = dataset  # 取dataset
        if dataset == 'mosi':   # 如果dataset是mosi，则是对mosi数据集属性进行初始化，调用get_mosi_dataset函数获取相应的数据集，并将返回的结果分别赋值给对应的属性。
            self.l_features, self.a_features, self.v_features, self.labels, self.labels_2 = get_mosi_dataset(mode=mode, text=text, audio=audio, video=video, normalize=normalize)
        elif dataset == 'mosei':    # 同理
            self.l_features, self.a_features, self.v_features, self.labels, self.labels_2, self.labels_7 = get_mosei_dataset(mode=mode, text=text, audio=audio, video=video, normalize=normalize)
        else:
            raise NotImplementedError

    def __getitem__(self, index):   # 用于获取数据集中指定索引位置的数据样本。根据dataset的取值，返回相应的数据样本。
        if self.dataset == 'mosi': #mosi返回的样本
            return self.l_features[index], self.a_features[index], self.v_features[index], self.labels[index], self.labels_2[index]
        elif self.dataset == 'mosei':   #mosei返回的比mosi多一个7分类样本
            return self.l_features[index], self.a_features[index], self.v_features[index], self.labels[index], self.labels_2[index], self.labels_7[index]
        else:
            raise NotImplementedError
            
    def __len__(self):  # 返回数据集长度，即样本数。
        return len(self.labels)
