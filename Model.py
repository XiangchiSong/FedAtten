import torch
import torch.nn as nn   # 定义神经网络层、损失函数、优化器
import torch.nn.functional as F # 一些非线性函数和损失函数的实现
from MLPProcess import MLPEncoder   #   MLPsBlock和MLP实现过程引入
from Utils import mean_temporal, get_mask_from_sequence, to_cpu #引入掩码张量生成和CPU计算模块
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# pack_padded_sequence 函数接受一个填充过的序列（padded sequence）和对应的长度列表，并将其转换为可供循环神经网络（RNN）处理的紧凑形式。该函数会将序列按照长度从长到短进行排序，并且将序列中的填充部分去除，以减少计算量。返回的结果是一个打包后的序列对象（PackedSequence），其中包含了打包后的序列数据和对应的长度信息。
# pad_packed_sequence 函数用于将打包后的序列对象（PackedSequence）重新恢复为填充过的序列。它接受一个打包后的序列对象和排序后的原始序列长度列表，并将序列恢复为原始的填充形式，使其可以进行后续的处理。返回的结果是一个填充过的序列和对应的长度列表。
from transformers import BertModel, BertConfig, BertTokenizer   # 导入 Hugging Face 的 Transformers 库中与 BERT 相关的模型、配置和分词器的模块，从而方便地进行文本相关的任务，如文本分类、命名实体识别等。
# BertModel用于进行预训练和微调任务。基于 Transformer 模型结构，在大规模的无标签文本数据上进行预训练，可以用于各种自然语言处理任务的特征提取和表示学习。
# BertConfig用于配置 BERT 模型的超参数和设置。它包含了模型的层数、隐藏层维度、注意力头数等参数，可以根据需要进行自定义和调整。
# BertTokenizer是用于将文本输入转换为 BERT 模型所需的输入格式的分词器。将文本分割为单词（或子词）并为每个单词（或子词）分配唯一的编号，还会生成输入的特殊标记（如 [CLS] 和 [SEP]）和注意力掩码（用于处理可变长度的序列）。
def get_output_dim(features_compose_t, features_compose_k, d_out, t_out, k_out):    # 根据不同的特征组合方式和输出维度配置，接受参数为“时间维度特征合成方法，类别特征合成方法，特征经过编码器和合成操作后得到的特征向量的维度，时间维度大小（时间步数量），类别特征大小，计算并返回最终的分类器输入维度classify_dim。
    if features_compose_t in ['mean', 'sum']:   # 时间维度特征合成方法为平均或求和
        classify_dim = d_out    # 在时间维度进行平均或求和操作，维度保持不变
    elif features_compose_t == 'cat':   # 若特征合成方式为拼接(concatenate)
        classify_dim = d_out * t_out    # 在时间维度进行拼接操作。这意味着时间维度的特征被拼接为一个更长的特征向量，特征维度等于当前的输出特征维度 d_out 乘以时间维度 t_out。
    else:
        raise NotImplementedError

    if features_compose_k in ['mean', 'sum']:   # 在时间维度操作后，继续进行特征类别的合成，方法若为平均或求和
        classify_dim = classify_dim # 在特征类别上进行平均或求和操作，同时间维度的合成原理一样，维度保持不变，因此classify_dim 不需要改变。
    elif features_compose_k == 'cat':   # 若类别特征合成方式为拼接(concatenate)
        classify_dim = classify_dim * k_out # 进行拼接操作。这意味着类别特征被拼接为一个更大的特征向量，其维度是当前特征维度 classify_dim 乘以类别特征 k_out。
    else:
        raise NotImplementedError
    return classify_dim # 返回分类器的输入维度大小。
    # classify_dim表示分类器的输入维度大小。特征编码器的输出维度d_out根据不同的情况，可能会直接作为分类器的输入维度，或者与其他维度（例如t_out）相乘后作为分类器的输入维度。


class Model(nn.Module): # 深度模型
    def __init__(self, opt):
        super(Model, self).__init__()
        d_t, d_a, d_v, d_common, encoders = opt.d_t, opt.d_a, opt.d_v, opt.d_common, opt.encoders   # 定义了文本特征维度，音频特征维度，视频特征维度，共享特征维度（用于定义不同模态（文本，音频和视频）经过特征编码器（encoder）后的共同特征维度），编码器类型。
        features_compose_t, features_compose_k, num_class = opt.features_compose_t, opt.features_compose_k, opt.num_class   # 时间维度特征合成方法，类别特征合成方法，模型的输出类别数量（需要分类的类别数目）
        self.time_len = opt.time_len    # 存储模型中的时间长度

        # 上边先把opt对象属性赋值给变量，避免模型方法内部频繁访问opt对象的属性值；下边再把这些变量赋值给模型属性，方便模型其他方法直接使用这些模型属性。

        self.d_t, self.d_a, self.d_v, self.d_common = d_t, d_a, d_v, d_common
        self.encoders = encoders
        assert self.encoders in ['lstm', 'gru', 'conv'] # 检测是否为三个编码器之一
        self.features_compose_t, self.features_compose_k = features_compose_t, features_compose_k
        assert self.features_compose_t in ['mean', 'cat', 'sum']    # 检测是否为三个组合方式之一
        assert self.features_compose_k in ['mean', 'cat', 'sum']

        # Bert Extractor
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True) # BertConfig.from_pretrained() 函数从预训练的 "bert-base-uncased" 模型中加载配置。output_hidden_states=True 的设置表示要输出所有隐藏状态，而不仅仅是最后一层的隐藏状态。
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)  # 加载BERT模型，通过传递之前加载的配置bertconfig可以确保加载的模型与配置一致。然后储存在变量bertmodel中

        # Extractors
        if self.encoders == 'conv': # 使用卷积神经网络，则定义两个一维卷积层，处理视频特征和音频特征，具有相同的输出通道数d_common, 卷积核大小为3，步长1，填充1（在输入的音频特征和视频特征的两侧各添加一个零值，以确保卷积运算后输出的特征尺寸与输入相同）。
            self.conv_a = nn.Conv1d(in_channels=d_a, out_channels=d_common, kernel_size=3, stride=1, padding=1)
            self.conv_v = nn.Conv1d(in_channels=d_v, out_channels=d_common, kernel_size=3, stride=1, padding=1)
        elif self.encoders == 'lstm':   # 使用LSTM，定义两个LSTM层，处理音频和视频特征，输入维度是d_v,d_a，共享状态维度为 d_common，层数为1，启用双向计算，且批量维度在第一维。这意味着输入的特征是一个形状为 (batch_size, sequence_length, d_v/a) 的张量。
            self.rnn_v = nn.LSTM(d_v, d_common, 1, bidirectional=True, batch_first=True)
            self.rnn_a = nn.LSTM(d_a, d_common, 1, bidirectional=True, batch_first=True)
        elif self.encoders == 'gru':    # 使用GRU，与LSTM类似，只不过层数为2。
            self.rnn_v = nn.GRU(d_v, d_common, 2, bidirectional=True, batch_first=True)
            self.rnn_a = nn.GRU(d_a, d_common, 2, bidirectional=True, batch_first=True)
        else:
            raise NotImplementedError

        # LayerNormalize & Dropout
        self.ln_a, self.ln_v = nn.LayerNorm(d_common, eps=1e-6), nn.LayerNorm(d_common, eps=1e-6)   # 两个LayerNorm层对特征进行归一化，用于音频特征和视频特征，输入维度相同都为d_common,即特征通道数。并使用一个较小值防止数值计算问题。
        self.dropout_t, self.dropout_a, self.dropout_v = nn.Dropout(opt.dropout[0]), nn.Dropout(opt.dropout[1]), nn.Dropout(opt.dropout[2]) # 定义三个Dropout层用于文本音频和视频特征，通过随机将一部分元素置为零来进行特征的随机失活。

        # Projector
        self.W_t = nn.Linear(d_t, d_common, bias=False) # 定义一个投影层，一个线性变换操作，将文本特征的维度从d_t投影到d_common,接受d_t的输入，并输出d_common的特征表示，权重参数可学习且没有偏置项。

        # MLPsEncoder
        self.mlp_encoder = MLPEncoder(activate=opt.activate, d_in=[opt.time_len, 3, d_common], d_hiddens=opt.d_hiddens, d_outs=opt.d_outs, dropouts=opt.dropout_mlp, bias=opt.bias, ln_first=opt.ln_first, res_project=opt.res_project)
        # 使用了之前定义的 MLPEncoder 类来构建一个具有多个隐藏层和输出层的编码器模型。包括激活函数，输入维度为[时间维度，3并表示音频视频和文本的特征维度，经过投影后的特征维度]，隐藏层维度，输出层维度，dropout概率列表，是否偏执，是否层归一化在FC前，是否残差投影。

        # Define the Classifier 定义分类器，用于将编码后的特征输入维度进行分类预测，根据输入维度 classify_dim 的大小选择不同的网络结构，即决定使用单层线性层还是多层线性层。
        classify_dim = get_output_dim(self.features_compose_t, self.features_compose_k, opt.d_outs[-1][2], opt.d_outs[-1][0], opt.d_outs[-1][1])    # 调用get_output_dim确定输入维度的具体值，输入参数为时间特征的合成方法，类别特征合成方法，最后一个特征编码层的输出维度，最后一个特征编码层的时间特征输出维度，最后一个特征编码层的空间特征输出维度。
        if classify_dim <= 128: # 如果输入维度小于128
            self.classifier = nn.Sequential(    # 那么直接创建一个Sequential方法（定义简单的前馈神经网络，它可以接受任意数量的模块作为输入，并按照顺序依次调用它们的 forward 方法）并添加一个线性层及逆行预测，该线性层将输入维度为 classify_dim 的特征映射到输出类别为 num_class 的预测结果。
                nn.Linear(classify_dim, num_class)
            )
        else:   # 如果输入维度大于128
            self.classifier = nn.Sequential(
                nn.Linear(classify_dim, 128),   # 用线性层将维度映射到128维空间
                nn.ReLU(),  # 非线性激活增强表达能力
                nn.Dropout(opt.dropout[3]), # 正则化一次
                nn.Linear(128, num_class),  # 用线性层再将128维映射到输出类别为 num_class 的预测结果
            )

    # tav:[bs, len, d]; mask:[bs, len]
    def forward(self, bert_sentences, bert_sentence_types, bert_sentence_att_mask, a, v, return_features=False, debug=False):   # 模型前向传播方法，输入：经过BERT编码后的输入序列的整数表示，输入序列中所属的句子类型的整数表示，输入序列中每个单词的注意力掩码信息。视觉特征[形状为[bs, len, d]]，音频特征[形状为[bs, len, d]]，默认只返回输出中的预测结果而不返回输出中的特征，默认不输出调试信息。
        l_av = a.shape[1]   # 音频或视频的原始长度是 a 的序列长度即shape[1]

        # Extract Bert features 提取BERT特征
        t = self.bertmodel(input_ids=bert_sentences, attention_mask=bert_sentence_att_mask, token_type_ids=bert_sentence_types)[0]  # 使用bertmodel实例，传入BERT句子编码，句子注意力掩码，句子类型，调用[0]索引获取BERT输出，即编码后的句子表示，并赋值给变量t
        if debug:
            print('Origin:', t.shape, a.shape, v.shape) # debug参数为真则打印原始tav的尺寸
        mask_t = bert_sentence_att_mask # Valid = 1 赋值方便后续使用
        t = self.W_t(t) # 通过投影层对t线性变换

        # Pad audio & video
        length_padded = t.shape[1]  # 根据t的形状确定填充后的长度
        pad_before = int((length_padded - l_av)/2)  # 前填充长度
        pad_after = length_padded - l_av - pad_before   # 后填充长度
        a = F.pad(a, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)    # 在张量 a 的第三个维度上进行填充，前面填充 pad_before 个元素，后面填充 pad_after 个元素，其他维度不进行填充，填充为 0
        v = F.pad(v, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)    # 同理
        a_fill_pos = (get_mask_from_sequence(a, dim=-1).int() * mask_t).bool()  # 调用函数在张量最后一个维度生成掩码，意味着掩码张量形状与a相同，然后转换为整型与mask_t（BERT句子的注意力掩码）逐个元素相乘，这样做的目的是保留 a 张量的有效位置，并将填充位置设为 0。再将结果张量转换为布尔型，得到最终的填充位置掩码。
        v_fill_pos = (get_mask_from_sequence(v, dim=-1).int() * mask_t).bool()  # 同上
        a, v = a.masked_fill(a_fill_pos.unsqueeze(-1), 1e-6), v.masked_fill(v_fill_pos.unsqueeze(-1), 1e-6) # 根据填充掩码位置将张量 a 和 v 中最后一维的对应位置值替换为一个小的非零值 1e-6。这样做的目的是在计算中避免使用填充位置上的值。
        if debug:
            print('Padded:', t.shape, a.shape, v.shape) # 打印张量tav形状
        mask_a = get_mask_from_sequence(a, dim=-1) # Valid = False  在张量 a 最后一个维度上生成掩码。这个掩码用于标记填充位置，即在掩码中填充位置的值为 False，有效位置的值为 True。
        mask_v = get_mask_from_sequence(v, dim=-1) # Valid = False  同理
        if debug:
            print('Padded mask:', mask_t, mask_a, mask_v, sep='\n') # 打印掩码张量
        lengths = to_cpu(bert_sentence_att_mask).sum(dim=1) # 将BERT句子的注意力掩码转移到CPU上，计算第一维度的和，即对样本中有效元素进行求和，这样可以得到一个表示每个样本有效元素个数的张量 lengths
        l_av_padded = a.shape[1]    # 将a张量的第1维度的长度赋值给l_av_padded，表示a和v张量填充后的长度，即每个样本的特征数，a和v是一样的所以可以共用

        # Extract features  根据不同的编码器类型进行特征提取
        if self.encoders == 'conv': # 卷积
            a, v = self.conv_a(a.transpose(1, 2)).transpose(1, 2), self.conv_v(v.transpose(1, 2)).transpose(1, 2)   # 将a和v的通道维度转置后做卷积，因为卷积要求通道维度在最后，然后再转置回来。
            a, v = F.relu(self.ln_a(a)), F.relu(self.ln_v(v))   #   对卷积后的a和v做归一化和激活
        elif self.encoders in ['lstm', 'gru']:  #LSTM或GRU
            a = pack_padded_sequence(a, lengths, batch_first=True, enforce_sorted=False)    # 对a和v进行打包，只保留有效长度lengths，去除填充部分，保持批次大小在第一维度（时间步在第二维度，特征维度在第三维度），并且不要求输入序列按照长度降序排序
            v = pack_padded_sequence(v, lengths, batch_first=True, enforce_sorted=False)
            self.rnn_a.flatten_parameters() # 调用 flatten_parameters 函数对循环神经网络（LSTM 或 GRU）的参数进行扁平化，以便更高效地进行计算
            self.rnn_v.flatten_parameters()
            (packed_a, a_out), (packed_v, v_out) = self.rnn_a(a), self.rnn_v(v) # 对音频a进行RNN编码，返回一个元组，packed_a是打包后的音频张量，a_out是编码后的输出张量
            a, _ = pad_packed_sequence(packed_a, batch_first=True, total_length=l_av_padded)    # 将打包（经过RNN编码后）的音频张量packed_a解包，保持批次维度在第一维度，并以l_av_padded（即每个样本特征数）作为总长度重新填充，得到原始张量a，
            v, _ = pad_packed_sequence(packed_v, batch_first=True, total_length=l_av_padded)
            if debug:
                print('After RNN', a.shape, v.shape)    # debug一下看看RNN编码后的a和v的形状
            if self.encoders == 'lstm':
                a_out, v_out =a_out[0], v_out[0]    # 如果使用LSTM那么编码器输出的a_out和v_out是一个元组，我们将其只设置为第一个元素，即最终时间步的输出，因为我们只需要时间步，不需要其他RNN的输出
            a = torch.stack(torch.split(a, self.d_common, dim=-1), -1).sum(-1)  # 使用split沿着a的最后一个维度（特征维度）分割成多张量，每个张量大小都是d_common，然后用stack函数进行堆叠，在对最后一个维度求和得到最终结果。
            v = torch.stack(torch.split(v, self.d_common, dim=-1), -1).sum(-1)  # 这样做的目的是将原始的特征维度划分为大小为 self.d_common 的子空间，并对子空间内的特征进行求和。这样做可以减少特征的维度，并且通过求和操作可以将每个子空间的信息进行合并。
            if debug:
                print('After Union', a.shape, v.shape)  # 在合并后看一下形状
            # a, v = F.relu(a), F.relu(v)
            a, v = F.relu(self.ln_a(a)), F.relu(self.ln_v(v))   # 对a和v进行特征维度归一化，和relu激活，进一步增强特征表示能力
            # t = F.relu(self.ln_t(t))
        else:
            raise NotImplementedError

        t, a, v = self.dropout_t(t), self.dropout_a(a), self.dropout_v(v)   #tav操作都完成后，对文本音频和视频特征通过以概率opt.dropout[0]、[1]、[2]随机将一些元素置为零来进行dropout处理，即特征的随机失活以减少过拟合。
        if debug:
            print('After Extracted', t.shape, a.shape, v.shape) # 在特征提取后查看形状

        # Padding temporal axis 时间维度填充
        t = F.pad(t, (0, 0, 0, self.time_len-l_av_padded, 0, 0), "constant", 0) # 对时间维度进行填充，让总长度变为time_len，填充数量为self.time_len - l_av_padded，这样可以让音频和视频长度与文本长度对齐，可以让整个序列并行计算。
        a = F.pad(a, (0, 0, 0, self.time_len-l_av_padded, 0, 0), "constant", 0)
        v = F.pad(v, (0, 0, 0, self.time_len-l_av_padded, 0, 0), "constant", 0)

        # Union 特征融合
        # x = torch.stack([t, a, a], dim=2)
        x = torch.stack([t, a, v], dim=2)   # 特征融合，使用stack函数将tav按第三维度进行堆叠，即让他们的特征值沿着新的维度组合在一起，形成新的张量x，其中每个元素都由tav的特征值组成，例如x[i，j]表示第i个样本在j个时间步的文本音频和视频特征。
        # 特征融合后即完成了不同模态的特征整合，为后续的分类任务提供更有代表性的特征表示。

        if debug:
            print('After Padded and Unioned on Temporal', t.shape, a.shape, v.shape, x.shape)   # 在对时间维度进行填充和特征融合后看一下形状变化

        # Encoding  对特征进行编码
        x = self.mlp_encoder(x, mask=None)  # 传入融合后的特征张量x，不使用掩码，对x进行非线性变换和映射，提取高阶表示。这个过程会让x经过多个线性层和激活函数的堆叠，对特征进行变换和抽象，特征会越来越丰富
        features = x    # 得到编码后的特征张量并赋值给feature
        if debug:
            print('After Encoder', x.shape) # 看一下x的形状

        # Compose [bs, t, k, d] 即 [bs, l, k, d] 融合张量，即对特征进行组合或合并操作, 以生成更综合的特征表示, 这样做可以捕捉到时间序列中的整体趋势、统计信息或全局特征。   (l变为t是因为l填充到了time_len的长度，为了让音频和视频长度与文本长度对齐，可以让整个序列并行计算。)
        if self.features_compose_t == 'mean':   # 先处理时间维度t[序列长度l]
            fused_features = x.mean(dim=1)  # 若使用均值融合，那么让x沿着第1维度（时间维度）求均值, 即将长度为t的时间维度降为1，最终得到形状为 [bs, k, d] 的张量。
        elif self.features_compose_t == 'sum':
            fused_features = x.mean(dim=1)  # 若使用求和融合，那么让x沿着第1维度求和，这将对每个样本在时间维度上进行求和，因此这里和mean操作一样，会将长度为t的时间维度降为1，最终得到形状为 [bs, k, d] 的张量。
        elif self.features_compose_t == 'cat':
            fused_features = torch.cat(torch.split(x, 1, dim=1), dim=-1).squeeze(1) # 若使用拼接融合，首先使用split沿第二维度（时间维度[序列长度]）将[bs,t,k,d]分割为t个子张量[bs, 1, k, d]，然后使用cat让他们沿着最后一个维度（特征维度）拼接在一起，得到[bs, 1, k, d*t] 的合成特征张量。然后使用squeeze(1)去除第二维度得到[bs, k, d*t]
        else:
            raise NotImplementedError

        if self.features_compose_k == 'mean':   # 对于类别特征
            fused_features = fused_features.mean(dim=1) # 使用均值融合，fused_features已经是经过时间特征处理后的结果，可以直接对其进行类别特征的操作。操作方法同上。
        elif self.features_compose_k == 'sum':
            fused_features = fused_features.mean(dim=1) # 使用求和融合，沿着第二维度求和得到特征向量
        elif self.features_compose_k == 'cat':
            fused_features = torch.cat(torch.split(fused_features, 1, dim=1), dim=-1).squeeze(1)    # 首先使用split沿第二维度（类别特征）将[bs, k, d]分割为k个子张量[bs, 1, d]，然后使用cat让他们沿着最后一个维度（特征维度）拼接在一起，得到[bs, 1, d*k] 的合成特征张量。然后使用squeeze(1)去除第二维度得到[bs, d*k]
        else:
            raise NotImplementedError

        if debug:
            print('Fused', fused_features.shape)    # 查看融合张量的形状
            
        # Predictions
        output = self.classifier(fused_features)    # 将融合张量输入分类器得到预测输出
        if return_features: # 若forward函数要求返回特征
            return [output, features]   # 返回预测输出及特征
        else:   # 若默认不返回特征
            return [output] # 只返回预测输出


if __name__ == '__main__':
    from Utils import to_gpu

    print('='*40, 'Testing Model', '='*40)  # 分隔符测试模型
    from types import SimpleNamespace    # 创建一个名为opts的对象空间
    opts = SimpleNamespace(d_t=768, d_a=74, d_v=35, d_common=128, encoders='gru', features_compose_t='cat', features_compose_k='mean', num_class=7, # BERT文本特征维度为768，音频特征维度74，视频特征维度35，共享特征维度128，编码器选择GRU，时间特征组合方式为cat拼接，空间特征组合方式为mean均值，分类类别为7
            activate='gelu', time_len=50, d_hiddens=[[20, 3, 128],[10, 2, 64],[5, 2, 32]], d_outs=[[20, 3, 128],[10, 2, 64],[5, 1, 32]],    # 激活函数GeLU，时间序列长50，隐藏层维度[[20, 3, 128],[10, 2, 64],[5, 2, 32]]，输出层维度[[20, 3, 128],[10, 2, 64],[5, 1, 32]]
            dropout_mlp=[0.3,0.4,0.5], dropout=[0.3,0.4,0.5,0.6], bias=False, ln_first=False, res_project=[True,True,True]  # MLP的dropout列表[0.3, 0.4, 0.5]，模型dropout列表[0.3,0.4,0.5,0.6]，不使用偏置，LN放置在NLP之后，对MLP全部输出进行残差投影
            )
    print(opts) # 打印opts看参数值

    t = [
        ["And", "the", "very", "very", "last", "one", "one"],
        ["And", "the", "very", "very", "last", "one"],
    ]   # 定义一个t的示例，包含两个样本的文本序列列表，每个样本是由单词组成的列表。
    a = torch.randn(2, 7, 74).cuda()    # a是一个（2，7，74）的张量，第一个维度是样本数，表示有两个样本的音频特征，即每个样本的音频特征是一个形状为 (7, 74) 的张量。第二个维度7代表时间步，表示音频序列被分割成7个连续的时间片段，每个时间步包含特定的音频或视频特征。第三个维度74表示音频特征维度。
    v = torch.randn(2, 7, 35).cuda()    # 数据转移到cuda上。
    mask_a = torch.BoolTensor([[False, False, False, False, False, False, False],
        [False, False, False, False, False, False, True]]).cuda()   # 一个形状为 (2, 7) 的布尔型张量，表示音频特征 a 中的填充位置，表示只有第二个样本的最后一个时间步被标为为填充位置。因为第一个样本的音频序列长度为 7，第二个样本的音频序列长度为 6，因此最后一个时间步需要被填充。
    mask_v = torch.BoolTensor([[False, False, False, False, False, False, False],
        [False, False, False, False, False, False, True]]).cuda()   # 转移到cuda上。
    a = a.masked_fill(mask_a.unsqueeze(-1), 0)  # mask_a.unsqueeze(-1) 将 mask_a 张量最后增加一个维度，的形状从 (2, 7) 变为 (2, 7, 1)，以便与 a 张量形状相匹配，然后将音频特征 a 中的填充位置的值替换为 0，而保留其他位置的元素值不变。
    v = v.masked_fill(mask_v.unsqueeze(-1), 0)
    # print(get_mask_from_sequence(a, dim=-1))

    model = Model(opts).cuda()  # 模型实例化，使用了之前定义的配置参数 opts，这样可以通过 model 来调用模型的前向传播方法进行预测或训练。
    
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # 从预训练的 bert-base-uncased 模型中创建了一个 tokenizer 实例 bert_tokenizer。
    sentences = [" ".join(sample) for sample in t]  # 对t中每个样本，将单词连接成字符串，保存在sentences中
    bert_details = bert_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)   # 使用batch_encode_plus方法对sentences字符串进行编码，设置add_special_tokens=True 来添加特殊标记，并设置 padding=True 来进行填充，以使所有输入序列的长度一致。编码结果储存在bert_details中。
    bert_sentences = to_gpu(torch.LongTensor(bert_details["input_ids"]))    #bert_details["input_ids"] 包含了经过BERT编码后的输入序列的整数表示。这些编码后的输入序列被转换为 torch.LongTensor 类型的张量，并以如GPU计算并保存在bert_sentences中了。
    bert_sentence_types = to_gpu(torch.LongTensor(bert_details["token_type_ids"]))  # 规则同上，只是把bert_details["token_type_ids"] 包含了每个单词在输入序列中所属的句子类型的整数表示进行计算和储存在bert_sentence_types中了。
    bert_sentence_att_mask = to_gpu(torch.LongTensor(bert_details["attention_mask"]))   # 规则同上，只是把bert_details["attention_mask"] 包含了输入序列中每个单词的注意力掩码信息（用于指示哪些单词是真实输入，哪些单词是填充的）进行计算和储存在bert_sentence_att_mask中了。

    result = model(bert_sentences, bert_sentence_types, bert_sentence_att_mask, a, v, return_features=True, debug=True) # 调用model函数，传入BERT 编码的输入句子，句子类型标识符，注意力掩码，音频张量，视频张量，要求返回输出中的特征，打印出中间过程的调试信息
    print([r.shape for r in result])    # 打印每个张量的形状。