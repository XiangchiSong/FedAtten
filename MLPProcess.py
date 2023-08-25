import torch
import torch.nn as nn
import torch.nn.functional as F # 一些非线性函数和损失函数的实现
from Utils import get_activation_function   #导入激活函数模块


# Perform MLP on last dimension
# activation: elu elu gelu hardshrink hardtanh leaky_relu prelu relu rrelu tanh
class MLP(nn.Module):   #  定义一个MLP类，继承 nn.Module 类的属性和方法
    def __init__(self, activate, d_in, d_hidden, d_out, bias):  #引入了激活函数；输入特征的总维度，即输入数据的特征数；隐藏层的维度，表示模型中隐藏层的单元数或输出特征的维度；输出特征的维度，即模型输出的特征数；一个布尔值，用于控制线性层是否包含偏置项。如果设置为 True，线性层会包含偏置项。
        super(MLP, self).__init__() # 通过调用 super(MLP, self).__init__()，子类 MLP 可以继承父类 nn.Module 的功能，并在自身的构造函数中添加额外的逻辑来定制化模型的行为。
        self.fc1 = nn.Linear(d_in, d_hidden, bias=bias) #创造一个输入维度为 d_in ,输出维度为 d_hidden 的全连接层，bias控制是否偏置。
        self.fc2 = nn.Linear(d_hidden, d_out, bias=bias)    #创造一个输入维度为 d_hidden，输出维度为 d_out 的全连接层，bias控制是否偏置。
        self.activation = get_activation_function(activate) #根据给定的激活函数名称 activate 获取对应的激活函数，并将其赋值给 self.activation。这里的激活函数将在模型接下来的前向传播过程中使用。
    
    # x: [bs, l, k, d] k=modalityKinds mask: [bs, l]    bs表示batch size批量大小，l是序列长度，k是模态数量，d是特征维度，输入的张量 x 的维度是[bs, l, k, d]，mask: [bs, l]：表示输入的掩码张量 mask 的维度是[批量数，序列长]，与x中的值相同。
    # mask 掩码张量来指示哪些位置是有效的（非填充部分），哪些位置是无效的（填充部分）。掩码张量通常是一个二进制张量，与输入数据的维度相同，其中 1 表示有效位置，0 表示无效位置。通过将掩码张量与输入数据相乘，可以将填充部分的影响消除，只对有效位置进行计算。
    def forward(self, x, mask=None):    # 输入x，不需要掩码张量
        x = self.fc1(x) # 第一层线性层
        x = self.activation(x)  # 应用激活函数，对元素进行非线性变换
        x = self.fc2(x) # 第二层线性层
        return x

# d_ins=[l, k, d]'s inputlength same for d_hiddens,dropouts
class MLPsBlock(nn.Module): # 一个由多个 MLP 组成的模块。它包含三个独立的 MLP，分别处理不同类型（长度、类型和特征维度）的输入数据。
    def __init__(self, activate, d_ins, d_hiddens, d_outs, dropouts, bias, ln_first=False, res_project=False):
    # 接受激活函数，lkd的输入维度列表（长度，类型，特征维度），lkd的隐藏层维度，lkd的输出维度列表，lkd的Dropout概率，是否偏置，是否在MLP的每个子模块前应用LayerNormalization，是否在MLP模型输出上应用残差投影允许模型输出与输入之间存在直接的跳跃连接。
        super(MLPsBlock, self).__init__()   # 继承父类 nn.Module 的功能
        self.mlp_l = MLP(activate, d_ins[0], d_hiddens[0], d_outs[0], bias) # 创建mlp_l模型，指定了激活函数类型，输入层维度，隐藏层维度和输出层维度并指定是否偏置。通过调用MLP类完成模型创建，模型由fc1和fc2组成，fc1表示输入层到隐藏层的线性层，输入维度为d_ins[0]，输出维度为d_hiddens[0]。fc2表示隐藏层到输出层的线性层，它的输入维度为d_hiddens[0]，输出维度为d_outs[0]。
        self.mlp_k = MLP(activate, d_ins[1], d_hiddens[1], d_outs[1], bias) #同理
        self.mlp_d = MLP(activate, d_ins[2], d_hiddens[2], d_outs[2], bias) #同理
        self.dropout_l = nn.Dropout(p=dropouts[0])  # 创建dropout_l,使用对应mlp_l的丢弃率参数p=dropouts[0]进行初始化。它是一个介于0和1之间的浮点数，控制输入层中的神经元被丢弃的概率。正则化用于减少神经网络的过拟合现象，在训练过程中随机丢弃一部分神经元的输出，这样可以增加网络的鲁棒性和泛化能力。
        self.dropout_k = nn.Dropout(p=dropouts[1])  #同理
        self.dropout_d = nn.Dropout(p=dropouts[2])  #同理
        if ln_first:    # 层归一化的位置选择，True则用于激活函数之前
            self.ln_l = nn.LayerNorm(d_ins[0], eps=1e-6)    # 使用输入维度d_ins来初始化归一化层。这是因为输入的维度信息是在d_ins中提供的。d_ins[0][1][2]是MLPsBlock的输入维度，分别对应着l、k、d三个维度，用于指定nn.LayerNorm操作的特征维度。eps=1e-6是用于稳定归一化操作的小常数值。
            self.ln_k = nn.LayerNorm(d_ins[1], eps=1e-6)
            self.ln_d = nn.LayerNorm(d_ins[2], eps=1e-6)
        else:   # 层归一化的位置选择，False则用于激活函数之后
            self.ln_l = nn.LayerNorm(d_outs[0], eps=1e-6)   # 使用输出维度d_outs来初始化归一化层。这是因为输出的维度信息是在d_outs中提供的。
            self.ln_k = nn.LayerNorm(d_outs[1], eps=1e-6)
            self.ln_d = nn.LayerNorm(d_outs[2], eps=1e-6)

        self.ln_fist = ln_first
        self.res_project = res_project
        if not res_project: # 如果使用了残差投影
            assert d_ins[0]==d_outs[0], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."    # 必须保证输入维度 d_ins 和输出维度 d_outs 在没有进行残差投影的情况下是相等的，否则输出AssertionError异常
            assert d_ins[1]==d_outs[1], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
            assert d_ins[2]==d_outs[2], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
        else:   # 如果没有。那么要创建残差投影
            self.res_projection_l = nn.Linear(d_ins[0], d_outs[0], bias=False)  # 使用nn.Linear创建线性投影层 ,用于对输入维度 d_ins 进行投影，将其映射到输出维度 d_outs,这样就实现了残差投影的功能。这样在正常的MLP接传播之外添加一个额外的线性投影层，可以更好传递梯度和信息，有助于模型训练和优化。
            self.res_projection_k = nn.Linear(d_ins[1], d_outs[1], bias=False)
            self.res_projection_d = nn.Linear(d_ins[2], d_outs[2], bias=False)
    
    # x: [bs, len, k, d] k=modalityKinds  mask: [bs, len]
    def forward(self, x, mask=None):    # 前向传播方法，mask 参数默认设置为 None
        if mask is not None:    #   如果mask不为none则警告，因为如果使用了掩码，输入维度 d_in 应该等于输出维度 d_out。
            print("Warning from MLPsBlock: If using mask, d_in should be equal to d_out.")
        if self.ln_fist:    # 根据 ln_fist 属性的值，选择使用 forward_ln_first 或 forward_ln_last 方法进行前向传播。
            x = self.forward_ln_first(x, mask)  # 若值为true，首先应用 Layer Normalization，然后进行前向传播
        else:
            x = self.forward_ln_last(x, mask)  # 否则，首先进行前向传播，然后应用 Layer Normalization。
        return x

    def forward_ln_first(self, x, mask):    # 首先应用 Layer Normalization，然后进行前向传播
        if self.res_project:    # 如果使用了残差投影则通过使用res_projection_l对输入的维度转换后的x进行投影操作，投影完后重新排列为原始维度顺序。维度转换的原因：上边在定义时提到了”Perform MLP on last dimension“
            residual_l = self.res_projection_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # 具体操作：将x张量的维度重新排列为 (bs,k,d,l) 的顺序，把要处理的l放在最后，这是为了满足线性投影操作的要求。然后通过一个nn.Linear线性层进行线性投影操作，再把投影后的张量恢复到原始维度(bs,l,k,d)。
        else:   # 如果不使用残差投影就直接将x本身作为残差项。
            residual_l = x
        x = self.ln_l(x.permute(0, 2, 3, 1))    # 维度转换把l放在最后，forward_ln_first要先进行归一化操作再送入mlp模块
        x = self.mlp_l(x, None).permute(0, 3, 1, 2) # 送入mlp_l后进行MLP处理，得到结果后再把l换回去。这个过程因为mask默认参数为none，所以不会做任何掩码操作。
        if mask is not None:    # 若需要掩码张量 （根据任务要求，只有l轴可能需要掩码张量，因此下边的k和d都没有写这个过程）
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).bool(), 0.0) # 如果存在mask参数，即对应于输入数据中的掩码张量，将使用mask对输出x进行掩码填充操作，将掩码为True的位置的值填充为0.0。
            # mask.unsqueeze(-1).unsqueeze(-1)对掩码张量进行维度扩展，通过unsqueeze函数，两次在维度末尾插入新的维度，将掩码张量的形状从(bs,l)扩展为(bs,l,1,1)，目的是保持和x的维度相同。然后利用.bool()将0变为False，1变为True。最后使用mask_fill对输入张量x进行掩码填充操作。掩码张量中为True的位置，对应的x张量中的元素将被替换为指定的填充值0.0。
        x = self.dropout_l(x)   # Dropout对输入x进行了随机失活操作，随机地将部分元素置为0，以防止过拟合
        x = x + residual_l  # 将Dropout后的x与残差相加，保留原始输入信息的同时，有助于梯度反向传播时更好传递，进而改善网络效果。
        
        if self.res_project:    # 同理
            residual_k = self.res_projection_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)   # 把k放在最后，处理完后再换回来
        else:
            residual_k = x
        x = self.ln_k(x.permute(0, 1, 3, 2))
        x = self.dropout_k(self.mlp_k(x, None).permute(0, 1, 3, 2))
        x = x + residual_k
        
        if self.res_project:
            residual_d = self.res_projection_d(x)   # d本身就是在最后，不用permute了
        else:
            residual_d = x
        x = self.ln_d(x)
        x = self.dropout_d(self.mlp_d(x, None))
        x = x + residual_d

        return x

    def forward_ln_last(self, x, mask): # 同上，只不过这里归一化操作在MLP之后，而不是先进行归一化再送入mlp
        if self.res_project:
            residual_l = self.res_projection_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            residual_l = x
        x = self.mlp_l(x.permute(0, 2, 3, 1), None).permute(0, 3, 1, 2) # 维度转换把l放在最后，forward_ln_last要先送入mlp模块再进行归一化操作
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).bool(), 0.0) # Fill mask=True to 0.0
        x = self.dropout_l(x)
        x = x + residual_l
        x = self.ln_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    # 维度转换把l放在最后后归一化，再把l换回来
        
        if self.res_project:    # 同上
            residual_k = self.res_projection_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        else:
            residual_k = x
        x = self.dropout_k(self.mlp_k(x.permute(0, 1, 3, 2), None).permute(0, 1, 3, 2))
        x = x + residual_k
        x = self.ln_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        
        if self.res_project:
            residual_d = self.res_projection_d(x)
        else:
            residual_d = x
        x = self.dropout_d(self.mlp_d(x, None))
        x = x + residual_d
        x = self.ln_d(x)

        return x


# d_in=[l,k,d], hiddens, outs = [[l,k,d], [l,k,d], ..., [l,k,d]] for n layers
class MLPEncoder(nn.Module):    # MLP编码器，对数据进行多层处理，包含了多个MLPsBlock层，每个层都是一个包含了多个MLP的块
    def __init__(self, activate, d_in, d_hiddens, d_outs, dropouts, bias, ln_first=False, res_project=[False, False, True]):    #这里指定最后一层d需要残差投影
        super(MLPEncoder, self).__init__()
        assert len(d_hiddens)==len(d_outs)==len(res_project)    # 确保d_hiddens、d_outs和res_project的长度相等，否则引发AssertionError异常
        self.layers_stack = nn.ModuleList([ # 创建一个包含多个MLPsBlock对象的列表。列表的长度由len(d_hiddens)确定，每个元素对应于一个隐藏层。
            MLPsBlock(activate=activate, d_ins=d_in if i==0 else d_outs[i-1], d_hiddens=d_hiddens[i], d_outs=d_outs[i], dropouts=dropouts, bias=bias, ln_first=ln_first, res_project=res_project[i])
            # d_hiddens[i]表示当前迭代的MLPsBlock的隐藏层维度，它是从d_hiddens列表中取出的值，d_outs同理，res_project[i]来获取对应层级的残差投影标志来指示每个MLPsBlock是否需要进行残差投影。
            # 如果i=0（即当前迭代的是第一个MLPsBlock实例），那么d_ins将被赋值为d_in，这是因为在MLPsBlock中的第一个层级，输入维度d_ins与整个MLPEncoder的输入维度d_in相同。否则（即当前迭代的是后续的MLPsBlock实例），d_ins将被赋值为d_outs[i-1]，即输入维度是前一个MLPsBlock实例的输出维度。
        for i in range(len(d_hiddens))])    # 每个MLPsBlock代表MLPEncoder中的一个编码层，每个MLPsBlock又是由一个MLP结构（2全连接+1激活）组成。通过循环迭代，有多少隐藏层就决定了有多少MLPsBlock堆叠在一起。计算几次隐藏就代表要把几次outs中的一个[l,k,d]MLPsBlock块做一次计算。

    def forward(self, x, mask=None):    # 将输入数据x通过多个MLPsBlock层进行串联处理，每个MLPsBlock层都会对输入进行编码处理，并将输出作为下一个MLPsBlock层的输入，最终返回最后一个MLPsBlock层的输出结果作为整个MLPEncoder模型的输出。
        for enc_layer in self.layers_stack: # 使用循环遍历layers_stack中的每个enc_layer（每个enc_layer实际上就是每个MLPsBlock实例）
            x = enc_layer(x, mask)  # 通过layers_stack方法进入ModuleList进而调用MLPsBlock类的前向传播方法来实现数据的编码处理
        return x


if __name__ == '__main__':
    from Utils import to_gpu, get_mask_from_sequence    # 引用GPU张量转移函数和掩码张量生成函数

    print('='*40, 'Testing Mask', '='*40)   # 打印分割线区分测试结果
    x = torch.randn(2, 3, 4)    # 创建一个形状为 (2, 3, 4) 的随机张量 x。
    mask = get_mask_from_sequence(x, -1) # valid=False/0    表示在最后一个维度上生成掩码张量。
    print(mask) # 打印值
    print(mask.shape)   # 打印形状

    x = to_gpu(torch.randn(2, 3, 5, 6)) # 创建一个形状为 (2, 3, 5, 6) 的随机张量 x
    mask = to_gpu(torch.Tensor([    # 创建一个形状为 (2, 3) 的掩码张量 mask，其中包含了一些布尔值。这里使用 to_gpu 函数将张量 mask 移动到 GPU 上。
        [False, False, False],
        [False, False, True],
    ]))

    # print('='*40, 'Testing MLP', '='*40)
    # mlp = to_gpu(MLP('gelu', 6, 16, 26, bias=False))
    # y = mlp(x, mask)
    # print(y.shape)

    # print('='*40, 'Testing MLPsBlock', '='*40)
    # mlpsBlock = to_gpu(MLPsBlock(activate='gelu', d_ins=[3, 5, 6], d_hiddens=[13, 15, 16], d_outs=[23, 25, 26], bias=False, res_project=True, dropouts=[0.1, 0.2, 0.3], ln_first=False))
    # y = mlpsBlock(x, mask=None)
    # print(y.shape)

    print('='*40, 'Testing MLPEncoder', '='*40) # 打印分割
    x = to_gpu(torch.randn(2, 100, 3, 128)) # 建立一个(2, 100, 3, 128) 的随机张量 x，并将其移动到 GPU 上
    encoder = to_gpu(MLPEncoder(activate='gelu', d_in=[100, 3, 128], d_hiddens=[[100, 3, 128], [100, 3, 128], [50, 2, 64], [50, 2, 64]], d_outs=[[100, 3, 128], [50, 2, 64], [50, 2, 64], [10, 1, 32]], dropouts=[0.3,0.5,0.6], bias=False, ln_first=True, res_project=[True, True, True, True]))
    # 创建一个MLPEncoder类的实例encoder，并将其移动到GPU上，参数设置：激活函数使用GeLU；输入维度[100, 3, 128]；隐藏层维度[[100, 3, 128], [100, 3, 128], [50, 2, 64], [50, 2, 64]]共有四个隐藏层；输出维度为[100, 3, 128], [50, 2, 64], [50, 2, 64], [10, 1, 32]共有四个输出；每个隐藏层的dropout概率为 [0.3, 0.5, 0.6]；不使用偏置；在每个层前进行归一化；对每个隐藏层都使用残差投影。
    y = encoder(x, mask=None)   # 将x通过encoder前向传播，得到输出张量y，不使用掩码
    print(y.shape)  # 输出张量y的形状


