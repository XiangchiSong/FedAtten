import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from sklearn.metrics import classification_report, accuracy_score   # 导入两个库：返回包括精确率（precision）、召回率（recall）、F1-score和支持数（support）的字典(output_dict=True)或字符串分类报告(output_dict=False)；正确分类的样本数量(normalize=False)或比例(normalize=True)的库函数


def set_logger(log_path):   # 设置日志记录器，并将日志同时记录到文件和控制台，以便进行调试和日志信息的跟踪，接受参数为日志文件路径。
    import logging  # 导入日志模块
    logger = logging.getLogger()    # 获取一个默认的日志记录器对象logger
    logger.setLevel(logging.DEBUG)  # 将日志记录器的级别设置为logging.DEBUG，即调试级别，记录所有日志消息。
    if not logger.handlers: # 检测日志记录器是否有处理器来避免重复添加处理器，如果没有，则：
        file_handler = logging.FileHandler(log_path)    # 创建一个loggingFileHandler对象，将日志记录到指定路径的指定文件。
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))  # 设置文件处理器的日志格式为%(asctime)s:%(levelname)s: %(message)s"，其中%(asctime)s表示日志记录的时间，%(levelname)s表示日志级别，%(message)s表示日志消息的内容。
        logger.addHandler(file_handler) # 将文件处理器添加到日志记录器的处理器列表中

        stream_handler = logging.StreamHandler()    # 接下来创建一个logging.StreamHandler对象，将日志记录输出到控制台。
        stream_handler.setFormatter(logging.Formatter("%(message)s"))   # 设置控制台处理器的日志格式为"%(message)s"，也就是仅包含日志消息的内容，不包括时间和级别。
        logger.addHandler(stream_handler)   # 然后，再将控制台处理器添加到日志记录器的处理器列表中。


def get_activation(activation): # 用于根据给定的激活函数名称返回对应的激活函数 [来自torch.nn模块]
    activation_dict = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "hardshrink": nn.Hardshrink,
        "hardtanh": nn.Hardtanh,
        "leakyrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "relu": nn.ReLU,
        "rrelu": nn.RReLU,
        "tanh": nn.Tanh,
    }
    return activation_dict[activation]


def get_activation_function(activation):    # 用于根据给定的激活函数名称返回对应的激活函数 [来自torch.nn.functional模块]
    activation_dict = { # 定义一个字典，包含了不同激活函数的名称和对象。
        "elu": F.elu,
        "gelu": F.gelu,
        "hardshrink": F.hardshrink,
        "hardtanh": F.hardtanh,
        "leakyrelu": F.leaky_relu,
        "prelu": F.prelu,
        "relu": F.relu,
        "rrelu": F.rrelu,
        "tanh": F.tanh,
    }
    return activation_dict[activation]  # 函数通过索引 activation_dict 字典，根据给定的 activation 参数返回对应的激活函数对象。例如，如果 activation 参数为 "relu"，则函数将返回 F.relu 函数对象。


def multiclass_acc(preds, truths):  # 用于计算多分类准确率。它接受预测值 preds 和真实标签 truths 作为输入参数。
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths)) # 首先对预测值和真实标签进行四舍五入操作，将连续的预测值和真实标签转换为离散的类别。然后，通过比较四舍五入后的预测值和真实标签，计算预测正确的样本数量。最后，将预测正确的样本数量除以总样本数量，得到多分类准确率。


def get_seperate_acc(labels, predictions, num_class):
    accs = [0 for i in range(num_class)]
    alls = [0 for i in range(num_class)]
    corrects = [0 for i in range(num_class)]
    for label, prediction in zip(labels, predictions):
        alls[label] += 1
        if label == prediction:
            corrects[label] += 1
    for i in range(num_class):
        accs[i] = '{0:5.1f}%'.format(100 * corrects[i] / alls[i])
    return ','.join(accs)


# For mosi mosei
def calc_metrics(y_true, y_pred, to_print=True):    # 计算评估指标，接受真实值，预测值和是否打印结果。
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """
    y_true, y_pred = y_true.reshape(-1,), y_pred.reshape(-1,)   # 先把真实值和预测值重新整形为一维。-1表示自动推断维度的大小，而逗号后的空格表示将数组转换为一维。假设二维数组形状为(3, 4)，可以使用reshape(-1,)将其转换为(12,)。

    test_preds = y_pred #使用test表示预测值和真实值
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])   # 获取真实值中不为0的元素的索引，以便后续在计算指标时进行相关的操作或筛选。首先遍历真实值中的元素，使用enumerate()和条件判断获取不等于0的值的索引i，返回并构建一个列表，其中只包含非零值，最后转换为一个np数组赋值给non_zeros。
    # print(test_preds.shape, test_truth.shape)

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)  # 将test_preds和test_truth的值限制在范围[-3.0, 3.0]内，使用np.clip函数进行截断。
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)  # 将test_preds和test_truth的值限制在范围[-2.0, 2.0]内，使用np.clip函数进行截断
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truthstest_truth   test_preds和test_truth之间的平均绝对误差（MAE）
    corr = np.corrcoef(test_preds, test_truth)[0][1]    # 计算test_preds和test_truth之间的相关系数，衡量了两个变量之间的线性关系的强度和方向。np.corrcoef()函数返回一个2x2的相关系数矩阵，其中对角线上的元素是各自变量的方差，而非对角线上的元素是两个变量之间的协方差。[0][1]表示相关系数矩阵中的第一行第二列元素，即test_preds和test_truth之间的相关系数。
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)  # 使用multiclass_acc函数计算了基于阈值[-3.0, 3.0]和[-2.0, 2.0]的多分类准确率，传入参数是经过截断处理的预测值和真实值。
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)  # 将截断后的test_preds和test_truth传入multiclass_acc()函数中，先将连续值四舍五入转换为离散的类别标签。然后，计算模型在这些离散类别上的准确率。这样做的目的是根据不同的阈值评估模型在不同类别划分上的性能，以获得更全面的分类准确率信息。

    # pos - neg 基于原始数据中的非零值划分，即非零值作为正例，零值作为负例。这样可以评估模型在区分正例和负例方面的性能。
    binary_truth = test_truth[non_zeros] > 0    # 基于非零元素的索引，将test_truth和test_preds中的值大于0的元素设置为True，否则为False，生成二进制标签。
    binary_preds = test_preds[non_zeros] > 0

    if to_print:
        logging.log(msg="MAE: "+str(mae), level=logging.DEBUG)  # 打印平均绝对误差（MAE）的值，表示预测值与真实值之间的平均绝对差异。
        logging.log(msg="Corr: "+str(corr), level=logging.DEBUG)    # 打印相关系数（Corr）的值，表示预测值与真实值之间的相关性。
        logging.log(msg="Acc5: "+str(mult_a5), level=logging.DEBUG) # 打印基于阈值[-2.0, 2.0]的多分类准确率（Acc5）的值。
        logging.log(msg="Acc7: "+str(mult_a7), level=logging.DEBUG) # 打印基于阈值[-3.0, 3.0]的多分类准确率（Acc7）的值。
        logging.log(msg="Acc2 (pos/neg): "+str(accuracy_score(binary_truth, binary_preds)), level=logging.DEBUG)    # 打印基于正负类别的二分类准确率（Acc2）的值，通过调用accuracy_score()函数计算。
        logging.log(msg="Classification Report (pos/neg): ", level=logging.DEBUG)   # 打印基于正负类别的分类报告（Classification Report）
        logging.log(msg=classification_report(binary_truth, binary_preds, digits=5), level=logging.DEBUG)   # 根据二分类的真实标签和预测结果，计算精确度、召回率、F1值等指标，精度为小数位后5位，并生成分类报告。

    # non-neg - neg 基于原始数据中的正负两类进行划分的，即将非负值（包括零值和正值）作为正例，负值作为负例。这样可以评估模型在将非负值与负值区分开的能力。
    binary_truth = test_truth >= 0  # 创建一个布尔数组binary_truth，用于表示正负类别的真实标签和预测结果。
    binary_preds = test_preds >= 0

    if to_print:
        logging.log(msg="Acc2 (non-neg/neg): " +str(accuracy_score(binary_truth, binary_preds)), level=logging.DEBUG)   # 打印基于正负类别的二分类准确率（Acc2）的值，通过调用accuracy_score()函数计算。
        logging.log(msg="Classification Report (non-neg/neg): ", level=logging.DEBUG)   # 打印基于正负类别的分类报告（Classification Report）
        logging.log(msg=classification_report(binary_truth, binary_preds, digits=5), level=logging.DEBUG)   # 根据二分类的真实标签和预测结果，计算精确度、召回率、F1值等指标，精度为小数位后5位，并生成分类报告。

    return accuracy_score(binary_truth, binary_preds)   # 返回基于正负类别的二分类准确率的值，它使用accuracy_score()函数计算binary_truth和binary_preds之间的准确率。

def calc_metrics_pom(y_true, y_pred, to_print=False):   # 与上同理
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """

    test_preds = y_pred
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    # pos - neg
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0

    if to_print:
        logging.log(msg="MAE: "+str(mae), level=logging.DEBUG)
        logging.log(msg="Corr: "+str(corr), level=logging.DEBUG)
        logging.log(msg="Acc2 (pos/neg): "+str(accuracy_score(binary_truth, binary_preds)), level=logging.DEBUG)
        logging.log(msg="Classification Report (pos/neg): ", level=logging.DEBUG)
        logging.log(msg=classification_report(binary_truth, binary_preds, digits=5), level=logging.DEBUG)

    # non-neg - neg
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0

    if to_print:
        logging.log(msg="Acc2 (non-neg/neg): " +str(accuracy_score(binary_truth, binary_preds)), level=logging.DEBUG)
        logging.log(msg="Classification Report (non-neg/neg): ", level=logging.DEBUG)
        logging.log(msg=classification_report(binary_truth, binary_preds, digits=5), level=logging.DEBUG)

    return accuracy_score(binary_truth, binary_preds)


def str2listoffints(v):
    temp_list = v.split('=')    # 将字符串 v 按照 '=' 进行分割，得到一个包含多个子字符串的列表 temp_list。
    temp_list = [list(map(int, t.split("-"))) for t in temp_list]   # 首先使用 t.split("-") 将其按照 '-' 分隔，然后使用 map(int, ...) 将每个部分转换为整数，最后使用 list(...) 将转换后的整数对象转换为列表。
    return temp_list    # 例如：如果输入的 v 为 "1-2-3=4-5-6=7-8-9"，那么将返回 [[1, 2, 3], [4, 5, 6], [7, 8, 9]]。

def str2bool(v):    # 将字符串 v 转换为布尔值
    """string to boolean"""
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True # 肯定返回true
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False # 否定返回false
    else:   # 抛出异常消息：Boolean value expected. + v
        import argparse
        raise argparse.ArgumentTypeError("Boolean value expected." + v) # 例如：如果调用 str2bool("true")，函数将返回 True。如果调用 str2bool("no")，函数将返回 False。如果调用 str2bool("maybe")，函数将引发一个异常。


def str2bools(v):   # 接受一个字符串 v 作为输入，并将其按照 '-' 分隔成多个部分，然后将每个部分转换为布尔值，最后将这些布尔值组成一个列表并返回。
    return list(map(str2bool, v.split("-")))    # 例如：如果输入的 v 为 "True-False-True"，那么 str2bools(v) 将返回 [True, False, True]。

def str2floats(v):  # 接受一个字符串 v 作为输入，并将其按照 '-' 分隔成多个部分，然后将每个部分转换为浮点数，最后将这些浮点数组成一个列表并返回。
    return list(map(float, v.split("-")))   # 例如：如果输入的 v 为 "1.2-3.4-5.6"，那么 str2floats(v) 将返回 [1.2, 3.4, 5.6]。

def whether_type_str(data):
    return "str" in str(type(data))


def get_predictions_tensor(predictions):
    pred_vals, pred_indices = torch.max(predictions, dim=-1)
    return pred_indices


# 0.5 0.5 norm
def showImageNormalized(data):
    import matplotlib.pyplot as plt

    data = data.numpy().transpose((1, 2, 0))
    data = data / 2 + 0.5
    plt.imshow(data)
    plt.show()


def rmse(output, target):   # 计算均方根误差（RMSE），接受参数是模型的输出和目标值
    output, target = output.reshape(-1,), target.reshape(   # 先用reshape把output和target调整成一维张量，以便后续计算。-1表示自动计算该维度的大小，而,表示保持该维度的长度不变。因此，reshape(-1,)将输入张量的所有元素按照一维顺序重新排列，并返回一个形状为(N,)的一维张量，其中N是输入张量的总元素数量。
        -1,
    )
    rmse_loss = torch.sqrt(((output - target) ** 2).mean()) # 计算输出和目标值差的平方，然后使用mean函数计算差的平方的均值，最后计算均方根，就可以得到rmse均方根误差。
    return rmse_loss


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

# 该函数的作用是根据输入的序列张量，在指定维度上生成一个掩码张量，用于标记序列中无效的位置。
def get_mask_from_sequence(sequence, dim):  # 生成掩码张量。sequence: 输入的序列张量。dim: 指定在哪个维度上生成掩码张量。
    return torch.sum(torch.abs(sequence), dim=dim) == 0 # torch.abs 函数对输入的序列张量 sequence 进行取绝对值操作；torch.sum 函数对取绝对值后的张量在指定的维度 dim 上进行求和；将求和结果与零进行比较，生成一个布尔张量，其中元素为True的表示对应位置的序列在指定维度上的求和为零，即为无效的，然后返回生成的掩码张量。


def lock_all_params(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False
    return model


def to_gpu(x, on_cpu=False, gpu_id=None):   # 辅助函数，用于将张量转移到GPU上； 输入一个张量，默认不使用CPU；若有多GPU，则NONE，即使用默认GPU
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:    # 检查当前是否可用 GPU（即 CUDA）并且 on_cpu 参数为 False
        x = x.cuda(gpu_id)  # 如果满足转移条件，则调用张量的 cuda(gpu_id) 方法将其转移到指定的 GPU 设备上
    return x


def to_cpu(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():   # 检查当前系统是否支持 GPU 计算
        x = x.cpu() # 将输入变量 x 移动到 CPU 上
    return x.data   # 返回移动到 CPU 上的张量数据。


def topk_(matrix, K, axis=1):   # 获取矩阵中每行或每列的前K个最大值的数据和对应的索引。接受一个输入矩阵，要获取最大值的个数，1为操作轴方向默认按行操作
    if axis == 0:   # 如果为0则按列操作。
        row_index = np.arange(matrix.shape[1 - axis])   # 先创建一个长度为矩阵列数的行索引数组row_index。matrix.shape返回一个元组，包含矩阵的维度信息，[1 - axis]用于根据axis的值选择维度信息中的行数或列数，然后，np.arange()函数根据给定的长度创建一个等差数组，其元素从0开始递增，步长为1。
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :] # 最终得到的topk_index是一个二维数组，其中包含了前K个最大值在原数组中的位置索引。首先使用np.argpartition(-matrix, K, axis=axis)函数获取每列中前K个最大值在原数组中的位置索引，负号表示按降序排列，然后使用切片操作，0:K表示从索引0开始到K-1的范围，而:表示选择所有的列。
        topk_data = matrix[topk_index, row_index]   # 根据给定的行索引和列索引，从矩阵中选择对应位置的元素值，并将结果保存在topk_data中。topk_index和row_index分别是行索引和列索引的数组，对应位置的值分别表示所需元素的行索引和列索引。通过将它们作为索引数组传递给二维数组matrix可以获取对应位置的元素值，然后赋值给topk_data，它将成为一个与索引数组形状相同的数组，其中包含了矩阵中根据索引选择的元素值。
        topk_index_sort = np.argsort(-topk_data,axis=axis)  # 对topk_data进行排序，并返回排序后的索引数组topk_index_sort。负号实现对数据进行降序排序，axis=axis是argsort()函数的参数，用于指定排序的维度。如果axis为0，则按列进行排序；如果axis为1，则按行进行排序。最后用argsort()函数返回数组排序后的索引。
        topk_data_sort = topk_data[topk_index_sort,row_index]   # 根据索引数组topk_index_sort和row_index获取对应位置的元素，得到一个新的数组topk_data_sort。topk_data: 是一个二维数组，表示需要进行取值操作的数据。topk_index_sort: 是一个索引数组，用于指定要获取的元素的位置。row_index: 是一个数组，用于指定元素的行索引。
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]  # 选择topk_index的前K行生成一个新的子数组，然后，该子数组与topk_index_sort和row_index一起用作索引数组，用于获取对应位置的元素。topk_index_sort: 是一个索引数组，用于指定要获取的元素的位置。row_index: 是一个数组，用于指定元素的行索引。
    else:   # 否则按行操作
        column_index = np.arange(matrix.shape[1 - axis])[:, None]   # 生成一个列索引数组，[:, None]将列索引数组的维度从一维转换为二维
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K] # 生成一个包含每行前K个最大值的索引的二维数组
        topk_data = matrix[column_index, topk_index]    # 得到一个包含每行前K个最大值的数据的二维数组topk_data
        topk_index_sort = np.argsort(-topk_data, axis=axis) # 得到排序后的索引数组topk_index_sort
        topk_data_sort = topk_data[column_index, topk_index_sort]   # 得到排序后的数据的二维数组topk_data_sort
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]   # 得到排序后的索引数组中前K个最大值的索引topk_index_sort
    return topk_data_sort, topk_index_sort  # 返回排序后的数据和对应的索引


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)


class PadCollateForSequence:
    def __init__(self, dim=0, pad_tensor_pos=[2, 3], data_kind=4):
        self.dim = dim
        self.pad_tensor_pos = pad_tensor_pos
        self.data_kind = data_kind

    def pad_collate(self, batch):
        new_batch = []

        for pos in range(self.data_kind):
            if pos not in self.pad_tensor_pos:
                if not isinstance(batch[0][pos], torch.Tensor):
                    new_batch.append(torch.Tensor([x[pos] for x in batch]))
                else:
                    new_batch.append(torch.stack([x[pos] for x in batch]), dim=0)
            else:
                max_len = max(map(lambda x: x[pos].shape[self.dim], batch))
                padded = list(
                    map(lambda x: pad_tensor(x[pos], pad=max_len, dim=self.dim), batch)
                )
                padded = torch.stack(padded, dim=0)
                new_batch.append(padded)

        return new_batch

    def __call__(self, batch):
        return self.pad_collate(batch)


class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(torch.nn.Module):   # PyTorch模块类SIMSE，用于计算预测值和真实值之间的SIMSE（Squared Integrated Mean Square Error）损失
    def __init__(self):
        super(SIMSE, self).__init__()   # 调用父类的构造函数super().__init__()来初始化模块

    def forward(self, pred, real):  # 定义了一个前向传播方法，接受两个输入参数pred和real表示预测值和真实值
        diffs = torch.add(real, -pred)  # 首先计算预测值和真实值之间的差异，使用torch.add函数来将真实值减去预测值
        n = torch.numel(diffs.data) # 用torch.numel函数计算差异张量的元素总数
        simse = torch.sum(diffs).pow(2) / (n ** 2)  # 计算差异张量的和，并平方，再除以元素总数的平方，得到SIMSE损失

        return simse    # 通过定义SIMSE类并实现其前向传播方法，可以将其作为一个模块在神经网络中使用，用于计算预测值和真实值之间的SIMSE损失。


class SAM(torch.optim.Optimizer):   # 继承自torch.optim.Optimizer类。SAM代表Sharpness-Aware Minimization，是一种优化算法，用于优化神经网络的参数
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs): # 接收参数：需要优化的模型参数，基础优化器（用于执行实际的参数更新，可以继承torch.optim.Optimizer任何的优化器类，例如torch.optim.SGD、torch.optim.Adam等），rho参数用于调整sharpness-aware正则化项的权重。**kwargs用于接收任意数量和任意名称的关键字参数，而不需要事先定义它们。
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"    # rho取值范围为非负数。默认值为0.05。

        defaults = dict(rho=rho, **kwargs)  # 使用dict()函数创建一个defaults字典，其中包含了rho参数和其他可能传入的关键字参数。
        super(SAM, self).__init__(params, defaults) # 调用父类的构造函数super().__init__(params, defaults)来初始化优化器

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)   # 基础优化器的实例，通过传入参数和关键字参数创建。用于执行实际的参数更新操作。
        self.param_groups = self.base_optimizer.param_groups    # 通过将基础优化器的参数组（param_groups）赋值给SAM的参数组，以便在外部调用SAM的参数组时能够访问基础优化器的参数组。

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


def aug_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    std_features = torch.std(data, dim=aug_dim)
    max_features, _ = torch.max(data, dim=aug_dim)
    min_features, _ = torch.min(data, dim=aug_dim)
    union_feature = torch.cat(
        (mean_features, std_features, min_features, max_features), dim=-1
    )
    return union_feature


def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features



if __name__ == "__main__":
    print(str2listoffints('10-2-64=5-2-32'))
