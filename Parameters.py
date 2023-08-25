import argparse # 解释命令行参数和选项并生成易于使用的界面
from Utils import str2bools, str2floats, str2listoffints    # 导入字符串转换为布尔值，浮点数和子字符串函数

def parse_args():
    parser = argparse.ArgumentParser()  # 创建命令行解析器来解析命令行参数，可以使用 add_argument() 方法添加各种参数规范，例如位置参数、可选参数、标志参数等。

    # Names, paths, logs
    parser.add_argument("--ckpt_path", default="./ckpt")    # 指定检查点（checkpoint）的保存路径
    parser.add_argument("--log_path", default="./log")  # 指定日志文件的保存路径
    parser.add_argument("--task_name", default="test")  # 指定任务名称

    # Data parameters
    parser.add_argument("--dataset", default='mosi_SDK', type=str)  # 默认指定数据集为mosi_SDK，类型为字符串
    parser.add_argument("--normalize", default='0-0-0', type=str2bools) # 指定是否对数据进行归一化操作，默认为 '0-0-0'，使用 str2bools 函数将字符串转换为布尔值列表。
    parser.add_argument("--text", default='text', type=str) # Only for CMUSDK dataset 指定文本数据的类型，默认为 'text'，类型为字符串。
    parser.add_argument("--audio", default='covarep', type=str) # Only for CMUSDK dataset 指定音频数据的类型，默认为 'covarep'，类型为字符串。
    parser.add_argument("--video", default='facet41', type=str) # Only for CMUSDK dataset 指定视频数据的类型，默认为 'facet41'，类型为字符串。
    parser.add_argument("--d_t", default=768, type=int) # 指定文本特征的维度，默认为 768，类型为整数。
    parser.add_argument("--d_a", default=74, type=int)  # 指定音频特征的维度，默认为 74，类型为整数。
    parser.add_argument("--d_v", default=47, type=int)  # 指定视频特征的维度，默认为 47，类型为整数。
    parser.add_argument("--batch_size", default=16, type=int)   # 指定批次大小，默认为 16，类型为整数。
    parser.add_argument("--num_workers", default=4, type=int)   # 指定数据加载器的并行工作线程数，默认为 4，类型为整数。
    parser.add_argument("--persistent_workers", action='store_true')    # 添加指定是否使用持久化工作线程的参数
    parser.add_argument("--pin_memory", action='store_true')    # 添加指定是否将数据加载到固定内存中的参数
    parser.add_argument("--drop_last", action='store_true') # 添加指定是否丢弃最后一个不完整的批次的参数
    parser.add_argument("--task", default='regression', type=str, choices=['classification', 'regression']) # 指定任务类型，默认为 'regression'，字符串类型，可选项为 'classification' 和 'regression'。
    parser.add_argument("--num_class", default=1, type=int) # 指定分类任务的类别数量，默认为 1，类型为整数。
    
    # Model parameters
    parser.add_argument("--d_common", default=128, type=int)    # 指定共享特征的维度，默认为 128，类型为整数。
    parser.add_argument("--encoders", default='gru', type=str)  # 指定编码器的类型，默认为 'gru'，类型为字符串。
    parser.add_argument("--features_compose_t", default='cat', type=str)    # 指定时间特征的组合方式，默认为 'cat'，类型为字符串。
    parser.add_argument("--features_compose_k", default='cat', type=str)    # 指定空间特征的组合方式，默认为 'cat'，类型为字符串。
    parser.add_argument("--activate", default='gelu', type=str) # 指定激活函数的类型，默认为 'gelu'，类型为字符串。
    parser.add_argument("--time_len", default=100, type=int)    # 指定时间维度的长度，默认为 100，类型为整数。
    parser.add_argument("--d_hiddens", default='10-2-64=5-2-32', type=str2listoffints)  # 指定 MLP 编码器隐藏层的维度，默认为 '10-2-64=5-2-32'，使用 str2listoffints 函数将字符串转换为嵌套列表的整数形式。
    parser.add_argument("--d_outs", default='10-2-64=5-2-32', type=str2listoffints) # 指定 MLP 编码器输出层的维度，默认为 '10-2-64=5-2-32'，使用 str2listoffints 函数将字符串转换为嵌套列表的整数形式。
    parser.add_argument("--dropout_mlp", default='0.5-0.5-0.5', type=str2floats)    # 指定 MLP 编码器的丢弃率，默认为 '0.5-0.5-0.5'，使用 str2floats 函数将字符串转换为浮点数列表。
    parser.add_argument("--dropout", default='0.5-0.5-0.5-0.5', type=str2floats)    # 指定模型中的丢弃率，默认为 '0.5-0.5-0.5-0.5'，使用 str2floats 函数将字符串转换为浮点数列表。
    parser.add_argument("--bias", action='store_true')  # 添加指定是否在模型中使用偏置项的参数
    parser.add_argument("--ln_first", action='store_true')  # 添加指定是否将LN层放置在MLP编码器之前的参数
    parser.add_argument("--res_project", default='1-1-1', type=str2bools)   # 指定是否在 MLP 编码器中使用残差投影，默认为 '1-1-1'，使用 str2bools 函数将字符串转换为布尔值列表。
    
    # Training and optimization
    parser.add_argument("--seed", default=0, type=int)  # 指定随机种子，默认为 0，类型为整数。
    parser.add_argument("--loss", default='MAE', choices=['Focal', 'CE', 'BCE', 'RMSE', 'MSE', 'SIMSE', 'MAE']) # 指定损失函数的类型，默认为 'MAE'，类型为字符串，可选值包括：'Focal'、'CE'、'BCE'、'RMSE'、'MSE'、'SIMSE'、'MAE'。
    parser.add_argument("--gradient_clip", default=1.0, type=float) # 指定梯度裁剪的阈值，默认为 1.0，类型为浮点数。
    parser.add_argument("--epochs_num", default=70, type=int)   # 指定训练的总轮数，默认为 70，类型为整数。
    parser.add_argument("--optm", default="Adam", type=str, choices=['SGD', 'SAM', 'Adam']) # 指定优化器的类型，默认为 'Adam'，类型为字符串，可选值包括：'SGD'、'SAM'、'Adam'。
    parser.add_argument("--bert_lr_rate", default=-1, type=float)   # 指定 BERT 模型的学习率倍率，默认为 -1，类型为浮点数。
    parser.add_argument("--weight_decay", default=0.0, type=float)  # 指定权重衰减的系数，默认为 0.0，类型为浮点数。
    parser.add_argument("--learning_rate", default=1e-4, type=float)    # 指定初始学习率，默认为 1e-4，类型为浮点数。
    parser.add_argument("--lr_decrease", default='step', type=str, choices=['multi_step', 'step', 'exp', 'plateau'])    # 指定学习率下降策略，默认为 'step'，类型为字符串，可选值包括：'multi_step'、'step'、'exp'、'plateau'。
    parser.add_argument("--lr_decrease_iter", default='60', type=str) # 50, or 50-75    指定学习率下降的迭代轮数，默认为 '60'，类型为字符串，可以是单个整数或由连字符分隔的整数序列。指定学习率下降的迭代轮数可以影响模型训练的效果和速度。具体而言，它决定了在训练过程中何时降低学习率以调整参数更新的步长。
    parser.add_argument("--lr_decrease_rate", default=0.1, type=float) # 0.1/0.5 for exp    指定学习率下降的倍率，默认为 0.1，类型为浮点数，用于指定指数衰减时的衰减倍率。
    parser.add_argument("--cuda", default="0", type=str)    # 指定要使用的 CUDA 设备编号，默认为 "0"，类型为字符串。
    parser.add_argument("--parallel", action='store_true')  # 添加指定是否在多个 GPU 上并行训练的参数
    parser.add_argument("--print_param", action='store_true')   # 添加指定是否打印模型参数信息的参数
    parser.add_argument("--bert_freeze", default='no', type=str, choices=['part', 'no', 'all']) # 指定是否冻结 BERT 模型的参数，默认为 'no'，类型为字符串，可选值包括：'part'、'no'、'all'。


    opt = parser.parse_args()   #解 析命令行参数，并将解析结果存储在opt对象中。

    return opt  # 将opt对象作为返回值返回


if __name__ == '__main__':
    args = parse_args() # 调用了之前定义的parse_args()函数来解析命令行参数，并将解析结果存储在args对象中。
    print(args) # 打印出args对象的内容，显示解析后的参数值。
