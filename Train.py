import os
import random   # 生成随机数
import logging  # 日志记录

import numpy as np
import torch
from scipy.stats.stats import pearsonr  # 统计计算，应用皮尔逊相关系数
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error   # 从sklearn引入准确率，F1和MAE的metric
from torch.utils.tensorboard import SummaryWriter   # 用于将训练过程中的数据写入TensorBoard可视化工具的摘要文件

from Config import CUDA # 调用Config文件里设置的CUDA
from DataLoaderLocal import mosi_r2c_7, pom_r2c_7, r2c_2, r2c_7    # 调用DataLoaderLocal文件里的标签二分类和七分类标准
from DataLoaderUniversal import get_data_loader #调用DataLoaderUniversal并进一步调用DataLoaderCMUSDK的数据预处理与数据集对象设置
from Model import Model # 具体模型导入（核心）
from Parameters import parse_args   # 引入默认参数和设置列表
from Utils import SAM, SIMSE, get_mask_from_sequence, rmse, set_logger, topk_, calc_metrics, calc_metrics_pom, to_gpu
# 导入优化神经网络的参数的优化算法SAM，计算预测值和真实值之间的SIMSE损失，用于标记序列中无效的位置掩码张量的生成函数，均方根误差RMSE(损失)，日志记录器，获取矩阵中每行或每列的前K个最大值的数据和对应的索引的函数，真实值与预测值之间的评估指标与评估性能的输出，pom预测指标同理，gpu转移函数

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 引用HuggingFace库中的BERT分词器，创建一个实例对文本进行分词


def set_random_seed(opt):  # 设置随机种子，用于实现训练的可重复性，确保在相同的种子下，每次运行代码时得到相同的随机结果，从而实现训练的可重复性。
    random.seed(opt.seed)   # 设置Python标准库中的random模块的种子，用于生成伪随机数。
    np.random.seed(opt.seed)    # 设置NumPy库中的随机数生成器的种子，用于生成伪随机数。
    torch.manual_seed(opt.seed)  # 保证在相同种子下Pytorch随机数生成器产生的随机序列是相同的
    torch.cuda.manual_seed_all(opt.seed)    # 保证在相同种子下CUDA随机数生成器产生的随机序列是相同的，如果系统中有多个GPU，则此函数会为每个GPU设置相同的种子。
    torch.backends.cudnn.deterministic = True   # 设置PyTorch库中的cuDNN库的确定性模式，以确保每次运行的结果一致。cuDNN是用于加速深度神经网络的GPU库。
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN的自动调优机制，以确保每次运行的结果一致。cuDNN的自动调优机制可以根据硬件和输入数据的特性来选择最优的算法和配置，但会导致结果的非确定性。


def set_cuda(opt):  # 设置CUDA环境
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA   # 限定Pytorch只在GPU计算
    torch.cuda.set_device("cuda:" + opt.cuda)   # 将指定的GPU设备设置为当前设备，以便PyTorch使用该设备进行CUDA计算


def prepare_ckpt_log(opt):  # 准备训练过程中的保存和记录文件的路径
    os.makedirs(os.path.join(opt.ckpt_path, opt.task_name), exist_ok=True)  # 创建存储模型检查点的文件夹路径,opt.ckpt_path和opt.task_name是命令行参数opt中的属性，通过os.path.join()函数将它们拼接在一起形成文件夹路径。exist_ok=True表示如果文件夹已存在，则不会引发异常。
    os.makedirs(os.path.join(opt.log_path, opt.task_name, "predictions"), exist_ok=True)    # 创建存储预测结果的文件夹路径。与上一行类似，但在该路径下创建名为"predictions"的子文件夹。
    set_logger(os.path.join(opt.log_path, opt.task_name, "log.log"))    # 调用set_logger函数，设置日志记录器并指定日志文件的路径。创建名为"log.log"的日志文件作为日志记录器的输出。它并没有创建子文件夹，而是直接创建了一个文件。
    writer = SummaryWriter(os.path.join(opt.log_path, opt.task_name))   # 创建SummaryWriter对象，用于记录训练过程中的事件和指标。SummaryWriter的参数是日志文件夹的路径。
    best_model_name_val = os.path.join(opt.ckpt_path, opt.task_name, "best_model_val.pth.tar")  # 指定保存在验证集上表现最好的模型的文件名及路径。文件名为"best_model_val.pth.tar"
    best_model_name_test = os.path.join(opt.ckpt_path, opt.task_name, "best_model_test.pth.tar")    # 指定保存在测试集上表现最好的模型的文件名及路径。文件名为"best_model_test.pth.tar"
    ckpt_model_name = os.path.join(opt.ckpt_path, opt.task_name, "latest_model.pth.tar")    # 指定保存最新模型的文件名及路径。文件名为"latest_model.pth.tar"
    return writer, best_model_name_val, best_model_name_test, ckpt_model_name   # 返回写入器、最佳模型在验证集上的保存路径、最佳模型在测试集上的保存路径和检查点模型的保存路径。这些变量可能会在训练过程中使用，用于保存模型和记录日志


def other_model_operations(model, opt): # 定义模型中其他操作函数
    for name, param in model.named_parameters():    # model.named_parameters()迭代模型中的所有参数，其中name是参数的名称，param是参数的值
        if opt.bert_freeze=='part' and "bertmodel.encoder.layer" in name:   # 如果opt.bert_freeze是part表示选择性冻结BERT模型的部分参数，且当前参数属于BERT模型中的编码器层，那么：
            layer_num = int(name.split("encoder.layer.")[-1].split(".")[0]) # 通过对参数名称进行分割和提取的方式来获取BERT模型encoder.layer后面的编号。
            if layer_num <= (8):
                param.requires_grad = False # 只有编码器层的编号小于等于8的参数会被设置为param.requires_grad = False，即不进行梯度更新，从而实现参数冻结，也就是说参数冻结只冻结编号小于等于8的参数
        elif opt.bert_freeze=='all' and "bert" in name: # 冻结全部参数
            param.requires_grad = False # 将所有的BERT参数的param.requires_grad属性设置为False，从而禁止对它们进行梯度更新。

        if 'weight_hh' in name: # weight_hh是LSTM模型中的隐藏状态到隐藏状态的权重参数。也就是说若使用LSTM，那么要使用init.orthogonal_初始化函数对参数进行正交初始化。
            torch.nn.init.orthogonal_(param)    # 通过对隐藏状态到隐藏状态的权重参数进行正交初始化，可以帮助LSTM模型更好地学习和表示输入序列中的信息。这可以提高模型的性能和训练稳定性，并有助于防止梯度消失或梯度爆炸等训练过程中的问题。
        if opt.print_param:
            print('\t' + name, param.requires_grad) # 打印参数的名称和requires_grad属性，查看参数的状态，检查参数是否被正确设置进行梯度计算或冻结。


def get_optimizer(opt, model): # 获取优化器和学习率调度器（lr_schedule）
    if opt.bert_lr_rate <= 0:   # 判断语句用于确定是否对BERT参数使用不同的学习率。若小于等于0，意味着不需要对BERT使用不同学习率，则使用模型中所有需要梯度更新的参数作为优化器的参数。
        params = filter(lambda p: p.requires_grad, model.parameters())  # filter() 函数结合 lambda 表达式来过滤出所有 model 中需要梯度更新的参数。 lambda 函数，用于判断参数 p 是否需要梯度更新。filter() 函数将 lambda 表达式应用于每个参数，并过滤出满足条件（需要梯度更新）的参数。最终将满足条件的参数赋值给 params 变量。
    else:   # 如果参数需要梯度更新
        def get_berts_params(model):    # 获取模型中属于BERT的参数，并且这些参数需要梯度更新。
            results = []    # 用于储存满足条件的参数
            for p in model.named_parameters():  # 遍历模型参数
                if 'bert' in p[0] and p[1].requires_grad:   # 判断当前参数的名称是否包含 'bert'，并且参数需要梯度更新。
                    results.append(p[1])    # 如果满足条件那么将参数添加到列表中
            return results  # 返回储存列表，这样后续的代码可以使用这个函数获取BERT参数并设置它们的学习率。
        def get_none_berts_params(model):   # 同理，获取模型中不属于BERT的参数，并且这些参数需要梯度更新。
            results = []
            for p in model.named_parameters():
                if 'bert' not in p[0] and p[1].requires_grad:
                    results.append(p[1])
            return results
        params = [  # 建立params列表，包含两个字典元素
            {'params': get_berts_params(model), 'lr': float(opt.learning_rate) * opt.bert_lr_rate},
            # 第一个字典是BERT参数字典，包含参数params和对应的学习率lr。参数由 get_berts_params(model) 函数返回，学习率是全局学习率 opt.learning_rate 乘以 opt.bert_lr_rate 的结果，因为BERT的参数量大，因此训练过程需要较小学习率来避免过度更新导致性能下降，因此要用 opt.bert_lr_rate 参数来调整BERT参数的学习率，可以实现对BERT参数更细粒度的控制。
            {'params': get_none_berts_params(model), 'lr': float(opt.learning_rate)},
            # 第二个元素是非BERT参数的字典，同样包含参数和学习率。参数由 get_none_berts_params(model) 函数返回，学习率是全局学习率 opt.learning_rate。
        ]
    if opt.optm == "Adam":  # Adam优化，使用 torch.optim.Adam 优化器
        optimizer = torch.optim.Adam(params, lr=float(opt.learning_rate), weight_decay=opt.weight_decay)
    elif opt.optm == "SGD": # SGD优化，使用 torch.optim.SGD 优化器
        optimizer = torch.optim.SGD(params, lr=float(opt.learning_rate), weight_decay=opt.weight_decay, momentum=0.9 )
    elif opt.optm == "SAM": # Utils里自定义的SAM优化器，采用了类似于 Adam 的优化器。参数 params 是模型的参数列表，内部优化器选择 torch.optim.Adam
        optimizer = SAM(params, torch.optim.Adam, lr=float(opt.learning_rate), weight_decay=opt.weight_decay,)
    else:
        raise NotImplementedError

    # 通过使用 LR 学习率调度器，可以在训练过程中按照指定的步数和衰减率自动调整优化器的学习率。这有助于在训练的不同阶段逐渐降低学习率，以提高模型的收敛性和性能。
    if opt.lr_decrease == 'step':   # 使用步长衰减方式调整学习率
        opt.lr_decrease_iter = int(opt.lr_decrease_iter)    # lr_decrease_iter是一个整数，表示学习率调整的步数，将其转换为整数类型，确保步数是一个整数值。
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decrease_iter, opt.lr_decrease_rate)    # 创建了一个 StepLR 学习率调度器对象。它的参数包括优化器对象 optimizer，学习率调整的步数 opt.lr_decrease_iter，以及学习率的衰减率 opt.lr_decrease_rate（浮点数）。
    elif opt.lr_decrease == 'multi_step':   # 使用多步衰减方式调整学习率
        opt.lr_decrease_iter = list((map(int, opt.lr_decrease_iter.split('-'))))    # opt.lr_decrease_iter 是一个字符串，表示学习率调整的步数，将字符串按照 "-" 进行分割，得到一个由字符串组成的列表。将列表中的每个字符串元素转换为整数类型，得到一个整数类型的列表。
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_decrease_iter, opt.lr_decrease_rate)   # 创建了一个 MultiStepLR 学习率调度器对象。它的参数包括优化器对象 optimizer，学习率调整的步数列表 opt.lr_decrease_iter，以及学习率的衰减率 opt.lr_decrease_rate（浮点数）。
    elif opt.lr_decrease == 'exp':  # 使用指数衰减方式调整学习率
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_decrease_rate)   # 创建了一个 ExponentialLR 学习率调度器对象。它的参数包括优化器对象 optimizer 和学习率的衰减率 opt.lr_decrease_rate（浮点数）。
    elif opt.lr_decrease == 'plateau':  # 使用基于指标变化的方式调整学习率
        mode = 'min' # if opt.task == 'regression' else 'max' mode表示评估指标的模式。在这里，它被设置为 'min'，这个模式可根据回归任务或分类任务的不同而有所变化，若是回归任务则设置为min，其他任务则设置为max。
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=int(opt.lr_decrease_iter), factor=opt.lr_decrease_rate,)    # 创建了一个 ReduceLROnPlateau 学习率调度器对象。它的参数包括优化器对象 optimizer、评估指标的模式 mode、等待迭代次数 patience（即opt.lr_decrease_iter的整数型） 和学习率的衰减率 factor（即opt.lr_decrease_rate 浮点数）。
    else:
        raise NotImplementedError
    return optimizer, lr_schedule   # 返回优化器对象和学习率调度器对象


def get_loss(opt):  # 选择相应损失函数
    if opt.loss == 'RMSE':
        loss_func = rmse
    elif opt.loss == 'MAE':
        loss_func = torch.nn.L1Loss()
    elif opt.loss == 'MSE':
        loss_func = torch.nn.MSELoss(reduction='mean')  # 均方根误差，对所有样本的均方误差进行求和并取平均。
    elif opt.loss == 'SIMSE':
        loss_func = SIMSE()
    else :
        raise NotImplementedError
    return [loss_func]  # 返回一个包含所选择的损失函数的列表。返回一个列表的好处是可以方便地对多个项进行处理和扩展。在这种情况下，由于只有一个损失函数，返回的列表只包含一个元素。


def train(train_loader, model, optimizer, loss_func, opt):  # 对模型进行训练，并返回训练过程中的损失和训练指标。参数为训练集的数据加载器，训练模型，优化器，损失函数和参数选项
    model.train()   # 将模型调整到训练模式，模式内有一些特定的训练相关操作，例如Batch Normalization 和 Dropout
    running_loss, predictions_corr, targets_corr = 0.0, [], []  # 初始化积累每个批次的损失值，积累模型的预测结果和目标标签的列表

    for _, datas in enumerate(train_loader):    # 遍历训练集的数据加载器，每次迭代得到一个批次的数据
        t_data, a_data, v_data = datas[0], datas[1].cuda().float(), datas[2].cuda().float() # 从数据中获取文本数据、音频数据和视频数据，并将它们移动到GPU上进行计算。音频和视频需要移动到GPU进行计算，并转换为浮点型。
        labels = get_labels_from_datas(datas, opt) # Get multiple labels    从datas中获取多个标签
        targets = get_loss_label_from_labels(labels, opt).cuda() # Get target from labels   将提取的标签labels转换为用于计算损失的目标张量targets，并移动到GPU进行计算。
        outputs = get_outputs_from_datas(model, t_data, a_data, v_data, opt) # Get multiple outputs 传入模型 model、文本数据 t_data、音频数据 a_data、视频数据 v_data 和命令行参数 opt，以获取模型的多个输出结果。这些输出结果通常包括预测结果和特征。
        loss = get_loss_from_loss_func(outputs, targets, loss_func, opt)  # Get loss 将获取到的输出结果 outputs 和标签数据 targets 作为参数，调用 get_loss_from_loss_func 函数，再传入损失函数 loss_func 和命令行参数 opt，以计算损失值。

        # 训练过程中的优化步骤，使用反向传播更新模型参数。
        optimizer.zero_grad()   # 将模型参数的梯度归零，清除之前的累积梯度。
        loss.backward() # 根据损失函数计算得到的损失值，进行反向传播，计算模型参数的梯度。
        if opt.gradient_clip > 0:   # 如果命令行参数 opt.gradient_clip 大于 0，即设置了梯度剪裁的阈值
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], opt.gradient_clip)  # 使用 torch.nn.utils.clip_grad_value_ 方法对模型参数的梯度进行剪裁，将梯度值限制在指定的范围内，以防止梯度爆炸的问题。
        optimizer.step()    # 根据计算得到的梯度更新模型参数。
        running_loss += loss.item() # 将当前批次的损失值 loss.item() 加到 running_loss 中，用于后续计算平均损失值。

        # 收集模型在训练集上的预测结果和真实标签，以便后续评估模型性能。
        with torch.no_grad():   # torch.no_grad() 上下文管理器，将以下代码块中的计算过程设置为不需要梯度，以减少内存消耗和加速计算。
            predictions_corr += outputs[0].cpu().numpy().tolist()   # 将模型预测结果outputs[0]转移到cpu上，并转化为NumPy数组，然后使用tolist方法将数组转换为Python列表，将转换后的预测结果列表添加到predictions_corr模型的预测结果列表中
            targets_corr += targets.cpu().numpy().tolist()  # 将目标标签targets转移到CPU上，并将其转换为NumPy数组，然后使用tolist方法将数组转换为Python列表。将转换后的目标标签列表添加到targets_corr目标标签结果列表中。

    predictions_corr, targets_corr = np.array(predictions_corr), np.array(targets_corr) # 将预测结果 predictions_corr 和目标标签 targets_corr 转换为 NumPy 数组
    train_score = get_score_from_result(predictions_corr, targets_corr, opt) # return is a dict 调用 get_score_from_result 函数，传入预测结果和目标标签，以及一些其他的选项参数 opt。计算并返回一组评估指标，如准确率、F1 分数、相关系数等。

    return running_loss/len(train_loader), train_score  #将计算得到的损失函数的平均值running_loss/len(train_loader)和训练集的评估指标train_score作为返回结果。损失函数的平均值表示模型在训练集上的平均损失，训练集的评估指标用于评估模型在训练集上的性能。


def evaluate(val_loader, model, loss_func, opt):    # 对模型在验证集上进行评估，并返回评估结果。参数为验证集的数据加载器，训练模型，损失函数和参数选项
    model.eval()    # 将模式设置为评估模式（用于训练后对于验证集和测试集的评估）
    running_loss, predictions_corr, targets_corr = 0.0, [], []  # 初始化积累每个批次的损失值，积累模型的预测结果和目标标签的列表
    with torch.no_grad():   # 禁用梯度计算，节省内存
        for _, datas in enumerate(val_loader):  # 遍历验证集的数据加载器，每次迭代得到一个批次的数据。
            t_data, a_data, v_data = datas[0], datas[1].cuda().float(), datas[2].cuda().float() # 从数据中获取文本数据、音频数据和视频数据，并将它们移动到GPU上进行计算。音频和视频需要移动到GPU进行计算，并转换为浮点型。
            labels = get_labels_from_datas(datas, opt) # Get multiple labels    从datas中获取多个标签
            targets = get_loss_label_from_labels(labels, opt).cuda() # Get target from labels   将提取的标签labels转换为用于计算损失的目标张量targets，并移动到GPU进行计算。
            outputs = get_outputs_from_datas(model, t_data, a_data, v_data, opt) # Get multiple outputs 传入模型 model、文本数据 t_data、音频数据 a_data、视频数据 v_data 和命令行参数 opt，以获取模型的多个输出结果。这些输出结果通常包括预测结果和特征。
            loss = get_loss_from_loss_func(outputs, targets, loss_func, opt)  # Get loss 将获取到的输出结果 outputs 和标签数据 targets 作为参数，调用 get_loss_from_loss_func 函数，再传入损失函数 loss_func 和命令行参数 opt，以计算损失值。
            running_loss += loss.item() # 将当前批次的损失值 loss.item() 加到 running_loss 中，用于后续计算平均损失值。

            predictions_corr += outputs[0].cpu().numpy().tolist()   # 将模型预测结果outputs[0]转移到cpu上，并转化为NumPy数组，然后使用tolist方法将数组转换为Python列表，将转换后的预测结果列表添加到predictions_corr模型的预测结果列表中
            targets_corr += targets.cpu().numpy().tolist()  # 将目标标签targets转移到CPU上，并将其转换为NumPy数组，然后使用tolist方法将数组转换为Python列表。将转换后的目标标签列表添加到targets_corr目标标签结果列表中。

    predictions_corr, targets_corr = np.array(predictions_corr), np.array(targets_corr) # 将预测结果 predictions_corr 和目标标签 targets_corr 转换为 NumPy 数组
    valid_score = get_score_from_result(predictions_corr, targets_corr, opt) # return is a dict 调用 get_score_from_result 函数，传入预测结果和目标标签，以及一些其他的选项参数 opt。计算并返回一组评估指标，如准确率、F1 分数、相关系数等。

    return running_loss/len(val_loader), valid_score, predictions_corr, targets_corr    # 将损失的平均值running_loss/len(val_loader)，评估指标valid_score，预测结果和真实标签作为返回结果。


def main(): # 主函数
    opt = parse_args()  # 解析命令行参数并进行相应的设置调给opt

    set_cuda(opt)   # 设置cuda环境
    set_random_seed(opt)    # 随机种子
    writer, best_model_name_val, best_model_name_test, _ = prepare_ckpt_log(opt)    # 准备训练过程中写入器，生成最佳模型在验证集上的保存路径，生成最佳模型在测试集上的保存路径（最后一个返回是ckpt_model_name检查点模型的保存路径，暂时不需要）

    logging.log(msg=str(opt), level=logging.DEBUG)  # 将命令行参数 opt 对象的配置信息记录到日志中，以便在训练过程中能够查看和追踪配置的细节，级别为 DEBUG 的日志通常用于记录调试信息和详细的运行日志。

    logging.log(msg="Making dataset, model, loss and optimizer...", level=logging.DEBUG)    # 打印日志，正在进行数据集，模型，损失函数和优化器的创建。
    train_loader, valid_loader, test_loader = get_data_loader(opt)  # 获取训练集，验证集和测试集的数据加载器
    model = Model(opt)  # 使用Model创建模型对象。
    other_model_operations(model, opt)  # 函数对模型进行其他操作，和对模型其他操作进行参数设置
    optimizer, lr_schedule = get_optimizer(opt, model)  # 获取模型的优化器和学习率调度器（lr_schedule）
    loss_func = get_loss(opt)   # 获取损失函数
    if opt.parallel:    # 如果设置了GPU并行训练
        logging.log(msg="Model paralleling...", level=logging.DEBUG)    # 打印消息，说明正在模型并行处理
        model = torch.nn.DataParallel(model, device_ids=list(map(int, CUDA.split(','))))    # 使用DataParallel将模型包装成多GPU模型。device_ids：设备列表，指定要在哪些GPU上并行执行模型。可以通过CUDA.split(',')将CUDA环境变量的值分割成设备列表，使其可以在指定的多个GPU上并行计算。
    model = model.cuda()    # 将模型移至GPU计算

    logging.log(msg="Start training...", level=logging.DEBUG)   # 打印日志开始训练
    best_score_val, best_score_test, best_score_test_in_valid  = None, None, None   # 初始化，用于存储最佳的验证集得分、测试集得分和验证集中的测试集得分。
    best_val_predictions, best_test_predictions, best_test_predictions_in_valid = None, None, None  # 初始化，用于存储在最佳验证集得分下的验证集预测结果、测试集预测结果和在验证集中的测试集预测结果。
    for epoch in range(opt.epochs_num): # 进行循环轮次迭代
        # Do Train and Evaluate
        train_loss, train_score = train(train_loader, model, optimizer, loss_func, opt) # 传入训练数据集train_loader、模型model、优化器optimizer、损失函数loss_func和选项opt。该函数返回训练损失train_loss和训练得分train_score
        val_loss, val_score, val_predictions, val_targets = evaluate(valid_loader, model, loss_func, opt)   # 传入验证数据集valid_loader、模型model、损失函数loss_func和选项opt。该函数返回验证损失val_loss、验证得分val_score、验证集的预测结果val_predictions和验证集的真实标签val_targets。
        test_loss, test_score, test_predictions, test_targets = evaluate(test_loader, model, loss_func, opt)    # 传入测试数据集test_loader、模型model、损失函数loss_func和选项opt。该函数返回测试损失test_loss、测试得分test_score、测试集的预测结果test_predictions和测试集的真实标签test_targets。
        if opt.lr_decrease == 'plateau':
            lr_schedule.step(test_loss) # 如果学习率调整策略opt.lr_decrease为'plateau'，则调用学习率调度器lr_schedule的step方法，传入测试损失test_loss作为参数，以根据测试损失动态调整学习率。
        else:
            lr_schedule.step()  # 如果不为'plateau'，则调用学习率调度器lr_schedule的stepLR方法，不传入任何参数，以按照预定义的学习率调整策略调整学习率。

        # Updata metrics, results and features
        if current_result_better(best_score_val, val_score, opt):   # 判断当前验证集评估结果是否优于历史最佳验证集评估结果，若是
            best_model_state_val = {    # 创建一个字典，其中包含当前轮次、模型的状态字典和优化器的状态字典。
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
            }
            best_score_val, best_val_predictions = val_score, val_predictions   # 更新best_score_val为当前验证集评估结果，更新best_val_predictions为当前验证集的预测结果。
            best_score_test_in_valid, best_test_predictions_in_valid = test_score, test_predictions # 更新best_score_test_in_valid为当前测试集在验证集上的评估结果,更新best_test_predictions_in_valid为当前测试集在验证集上的预测结果。
            logging.log(msg='Better Valid score found...', level=logging.DEBUG) # 打印一条日志消息，表示发现了更好的验证集评估结果。
            calc_metrics(val_targets, val_predictions)  # 调用calc_metrics函数计算验证集的指标，传入并记录验证集的真实标签val_targets和预测结果val_predictions。
            logging.log(msg='Test in Better Valid score found...', level=logging.DEBUG) # 打印一条日志消息，表示在验证集上发现了更好的测试集评估结果。
            calc_metrics(test_targets, test_predictions)    # 调用calc_metrics函数计算测试集在验证集上的指标，传入测试集的真实标签test_targets和预测结果test_predictions。

        # Log the epoch result 在每个训练轮次结束后，都要记录当前轮次的结果。
        msg = build_message(epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score) # 调用build_message函数构建包含当前轮次结果的日志消息，并将其赋值给变量msg。
        logging.log(msg=msg, level=logging.DEBUG)   # 使用logging模块将日志消息msg记录到日志中。
        log_tf_board(writer, epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score, lr_schedule)   # 将训练和验证过程中的损失值和指标值写入TensorBoard，以便进行可视化和分析。传入参数分别是：TensorBoard的写入器，当前epoch，训练集损失，训练集指标（字典），验证集损失，验证集指标（字典），测试集损失，测试集指标（字典），学习率调度器的当前学习率。

    # Log the best 不同于上边在每个训练轮次结束后都要记录当前轮次的结果，这里创建单条消息只记录最佳结果。
    logging.log(msg=build_single_message(best_score_val, 'Best Valid Score \t\t'), level=logging.DEBUG) # 创建第一条消息是最佳验证集得分（best_score_val）的日志消息，前缀为"Best Valid Score"，使用logging.log函数将这条消息记录到日志中，日志级别为DEBUG。
    logging.log(msg=build_single_message(best_score_test_in_valid, 'Test Score at Best Valid \t'), level=logging.DEBUG) # 第二条消息是验证集中的测试集得分（best_score_test_in_valid）的日志消息，前缀为"Test Score at Best Valid"，使用logging.log函数将这条消息记录到日志中，日志级别为DEBUG。
    writer.close()  # 关闭TensorBoard的写入器，确保日志写入完整并保存

    # Save predictions 保存预测结果
    np.save(os.path.join(opt.log_path, opt.task_name, "predictions", "val.npy"), best_val_predictions)  # np.save函数将最佳验证集预测结果（best_val_predictions）保存为Numpy数组文件（.npy格式），文件路径为opt.log_path/opt.task_name/predictions/val.npy
    np.save(os.path.join(opt.log_path, opt.task_name, "predictions", "test_for_valid.npy"), best_test_predictions_in_valid) # 保存了在验证集上的测试集最佳预测结果（best_test_predictions_in_valid），文件路径为opt.log_path/opt.task_name/predictions/test_for_valid.npy。

    # Save model 保存最佳验证集得分时的模型状态
    torch.save(best_model_state_val, best_model_name_val)   # 使用save函数把最佳模型的状态保存在best_model_name_val路径中（上边已经定义和初始化好了：指定保存在验证集上表现最好的模型的文件名及路径。文件名为"best_model_val.pth.tar"）


def current_result_better(best_score, current_score, opt):  # 判断当前的评估结果是否优于最佳评估结果，输入参数为最佳评估结果的字典，包含不同指标的分数，当前评估结果的字典，包含不同指标的分数，和配置参数
    if best_score is None:  # best_score为空（即第一次评估），则认为当前评估结果是最佳结果，返回True。
        return True
    if opt.task == 'classification':    # 如果任务类型是分类任务，则通过比较当前评估结果的指定类别准确度和最佳评估结果的指定类别准确度来判断当前结果是否更好。如果当前准确度高于最佳准确度，则返回True，否则返回False。
        return current_score[str(opt.num_class)+'-class_acc'] > best_score[str(opt.num_class)+'-class_acc']
    elif opt.task == 'regression':  # 如果任务类型是回归任务，则通过比较当前评估结果的平均绝对误差（MAE）和最佳评估结果的平均绝对误差来判断当前结果是否更好。如果当前MAE更小（更接近真实值），则返回True，否则返回False。
        return current_score['mae'] < best_score['mae']
    else:
        raise NotImplementedError


def get_labels_from_datas(datas, opt):  # 从数据中获取标签。根据opt.dataset的取值，决定返回哪些数据作为标签
    if 'SDK' in opt.dataset:    # opt.dataset中包含字符串"SDK"，表示使用的数据集是SDK数据集（CMUSDK或MOSI_SDK），则返回datas中从第4个元素到倒数第2个元素。
        return datas[3:-1]
    else:   # 否则，表示使用的数据集是其他数据集（如MOSI或CMU-MOSEI），则返回datas中从第4个元素开始的所有元素作为标签。
        return datas[3:]


def get_loss_label_from_labels(labels, opt):    # 根据数据集和任务的取值从标签中获取用于计算损失的目标数据
    if opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50']:  # 表示使用的是SDK数据集（CMUSDK或MOSI_SDK），则根据opt.task的取值进行如下操作
        if opt.task == 'regression':    # 若是回归任务
            labels = labels[0]  # 直接返回第一个元素
        elif opt.task == 'classification' and opt.num_class==2: # 若是分类任务且分类数为2
            labels = labels[1]  # 返回第二个元素
        elif opt.task == 'classification' and opt.num_class==7: # 若是分类任务且分类数为7
            labels = labels[2]  # 返回第三个元素
        else:
            raise NotImplementedError
    else:   # 使用的是其他数据集（如MOSI或CMU-MOSEI），目前不支持该数据集
        raise NotImplementedError
    return labels   # 返回labels


def get_outputs_from_datas(model, t_data, a_data, v_data, opt): # 根据输入数据获得模型的输出，输入为模型对象，三种数据和参数配置。
    if a_data.shape[1] > opt.time_len:  # 检查音频长度是否超过设定的时间长度阈值，如果超过
        a_data = a_data[:, :opt.time_len, :]    # 将音频和视频阶段，使其长度等于阈值
        v_data = v_data[:, :opt.time_len, :]    # 切片操作，取所有行和列的前opt.time_len个元素，opt.time_len是通过opt对象获取的时间长度阈值，最后的:表示保留所有维度。通过切片操作，保留前 opt.time_len 个时间步的音频数据。
        t_data = [sample[:opt.time_len] for sample in t_data]   # 相应截断，保留样本的前opt.time_len个单词，确保数据长度和模型输入要求一致。
    sentences = [" ".join(sample) for sample in t_data] # 将文本数据转换为句子列表，每个句子是将单词连接成的字符串
    bert_details = bert_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)   # 使用BERT tokenizer对句子进行批量编码，输入包含多个句子的列表sentences，在编码中添加特殊令牌（如起始令牌和终止令牌），并对编码进行填充使其具有相同长度。返回一个字典bert_details，包含编码后的句子信息，如 input ids、token type ids 和 attention mask。
    # print(encoded_bert_sent)
    bert_sentences = to_gpu(torch.LongTensor(bert_details["input_ids"]))    # 将编码后的句子的输入ID转换为LongTensor类型的张量，并将张量移动到GPU上进行加速计算。
    bert_sentence_types = to_gpu(torch.LongTensor(bert_details["token_type_ids"]))  # 将编码后的句子的令牌类型ID转换为LongTensor类型的张量，并将张量移动到GPU上进行加速计算。
    bert_sentence_att_mask = to_gpu(torch.LongTensor(bert_details["attention_mask"]))   # 将编码后的句子的注意力掩码转换为LongTensor类型的张量，并将张量移动到GPU上进行加速计算。
    outputs = model(bert_sentences, bert_sentence_types, bert_sentence_att_mask, a_data, v_data, return_features=True)  # 调用了model对象，传递了多个输入参数，包括bert_sentences（BERT编码后的句子输入）、bert_sentence_types（句子类型）、bert_sentence_att_mask（句子的注意力掩码）、a_data（音频数据）和v_data（视频数据）。return_features=True返回模型的特征。

    return outputs  # 返回模型输出


def get_loss_from_loss_func(outputs, labels, loss_func, opt):   # 根据模型的输出和标签计算损失函数的值，输入为模型输出（通常包括预测结果），用于计算损失的标签数据，损失函数对象和命令行参数对象
    # Get predictions
    # predictions, T_F_, A_F_, V_F_ = outputs[0], outputs[1], outputs[2] ,outputs[3]
    predictions = outputs[0]    # 预测结果在模型输出outputs的第一个元素中，outputs[0]=output预测结果，[1]=feature特征。
    task_loss = loss_func[0]    # 从损失函数列表loss_func中获取任务的损失函数，通常保存在第一个位置

    # Get loss from predictions
    if opt.loss in ['RMSE', 'MAE', 'MSE', 'SIMSE']: # 损失函数有四种
        loss = task_loss(predictions.reshape(-1, ), labels.reshape(-1, ))   # 通过调用task_loss对象并传入预测结果和标签数据，并使用reshape将预测结果和标签数据平展成一维张量，然后损失函数计算对象计算预测结果和标签之间的损失值，结果返回loss
    else:
        raise NotImplementedError

    # Get loss from features

    return loss # 返回计算得到的损失值


def get_score_from_result(predictions_corr, labels_corr, opt):  # 根据模型在验证集或测试集上的预测结果和真实标签计算模型的评分指标，接收参数是模型的预测结果，真实标签和参数
    if opt.task == 'classification':    # 如果是分类任务
        if opt.num_class == 1:  # 如果类别数量为1，说明只存在一个类别，无法进行多类别分类，则需要将模型的预测结果转换为二进制形式。
            predictions_corr = np.int64(predictions_corr.reshape(-1,) > 0)  # predictions_corr.reshape(-1,) > 0将预测结果进行判断，大于0的部分被认为属于正类，将其转换为布尔型数组。然后，通过np.int64将布尔型数组转换为整数数组，其中正类为1，负类为0。
        else:
            _, predictions_corr = topk_(predictions_corr, 1, 1) # 否则，topk_(predictions_corr, 1, 1)通过调用topk_函数来获取概率最高的类别。函数的参数为预测结果predictions_corr、保留的最高概率个数为1、轴方向为1（按行计算）。_表示丢弃，predictions_corr则表示概率最高的类别。这样处理之后，predictions_corr的形状为(样本数量,)，其中的值表示每个样本的预测类别。
        predictions_corr, labels_corr = predictions_corr.reshape(-1,), labels_corr.reshape(-1,) # 预测结果predictions_corr和标签labels_corr被调整为形状为(样本数量,)的一维数组，方便进行后续的指标计算。
        acc = accuracy_score(labels_corr, predictions_corr) # 计算预测结果predictions_corr和标签labels_corr之间的准确率。
        f1 = f1_score(labels_corr, predictions_corr, average='weighted')    # 计算预测结果predictions_corr和标签labels_corr之间的F1分数,average='weighted'表示采用加权平均的方式计算F1分数，对于多类别分类任务，不同类别的样本数量会影响计算结果。
        return {    # 计算得到的准确率（accuracy）和F1分数（F1 score）以字典的形式返回
            str(opt.num_class)+'-cls_acc': acc, # 这个键的值对应计算得到的准确率
            str(opt.num_class)+'-f1': f1    # 这个键的值对应计算得到的F1分数
        }
    elif opt.task == 'regression':  # 处理回归任务的评估指标计算
        predictions_corr, labels_corr = predictions_corr.reshape(-1,), labels_corr.reshape(-1,) # 将预测值 predictions_corr 和标签值 labels_corr重塑为一维数组形式。
        mae = mean_absolute_error(labels_corr, predictions_corr)    # 使用 mean_absolute_error() 函数来计算预测值和标签值之间的平均绝对误差。将计算得到的 MAE 存储在变量 mae 中。
        corr, _ = pearsonr(predictions_corr, labels_corr )  # 使用 pearsonr() 函数来计算预测值和标签值之间的相关系数。相关系数衡量了两个变量之间的线性关系强度和方向。将计算得到的相关系数存储在变量 corr 中。

        if opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50']:  # 如果使用MOSI和MOSEI
            if 'mosi' in opt.dataset:   # 如果使用MOSI，则将预测值和标签值分别传递给函数 mosi_r2c_7() 进行转换（四舍五入加3后转64位整数，与MOSI数据集对应），并将转换后的结果存储在 predictions_corr_7 和 labels_corr_7 中。
                predictions_corr_7 = [mosi_r2c_7(p) for p in predictions_corr]
                labels_corr_7 = [mosi_r2c_7(p) for p in labels_corr]
            else:   # 如果使用MOSEI，则将预测值和标签值分别传递给函数 r2c_7() 进行转换，并将转换后的结果存储在 predictions_corr_7 和 labels_corr_7 中。
                predictions_corr_7 = [r2c_7(p) for p in predictions_corr]
                labels_corr_7 = [r2c_7(p) for p in labels_corr]

            # 将MOSI和MOSEI两个数据集的预测值和标签值从连续的实数范围转换为离散的 2 类分类标签。转换后的结果存储在 predictions_corr_2 和 labels_corr_2 中。
            predictions_corr_2 = [r2c_2(p) for p in predictions_corr]
            labels_corr_2 = [r2c_2(p) for p in labels_corr]

            # 根据转换后的离散分类标签 labels_corr_7 和 labels_corr_2，以及相应的预测结果 predictions_corr_7 和 predictions_corr_2，计算了准确率（accuracy）和加权 F1 分数（weighted F1 score）。
            acc_7 = accuracy_score(labels_corr_7, predictions_corr_7)
            acc_2 = accuracy_score(labels_corr_2, predictions_corr_2)
            f1_2 = f1_score(labels_corr_2, predictions_corr_2, average='weighted')
            f1_7 = f1_score(labels_corr_7, predictions_corr_7, average='weighted')

            return {    # 根据计算得到的评估指标，构建了一个字典作为返回结果
                'mae': mae,
                'corr': corr,
                '7-cls_acc': acc_7,
                '2-cls_acc': acc_2,
                '7-f1': f1_7,
                '2-f1': f1_2,
            }

        elif opt.dataset in ['pom_SDK', 'pom']: # 对于数据集pom，使用了pom_r2c_7() 对预测结果和真实标签进行转换，并计算了基于 7 类分类的准确率和加权 F1 分数（pom没有二分类）。
            predictions_corr_7 = [pom_r2c_7(p) for p in predictions_corr]   #  分别表示经过转换后的预测结果和真实标签。
            labels_corr_7 = [pom_r2c_7(p) for p in labels_corr]
            acc_7 = accuracy_score(labels_corr_7, predictions_corr_7)
            f1_7 = f1_score(labels_corr_7, predictions_corr_7, average='weighted')
            return {
                'mae': mae,
                'corr': corr,
                '7-cls_acc': acc_7,
                '7-f1': f1_7,
            }

        elif opt.dataset in ['mmmo', 'mmmov2']: # 对于数据集 'mmmo' 和 'mmmov2'，将连续的回归结果转换为二分类结果，以阈值 3.5 为界限进行转换。
            predictions_corr_2 = [int(p>=3.5) for p in predictions_corr]    # 对于预测结果 predictions_corr 和真实标签 labels_corr，应用了阈值判定，将大于等于 3.5 的值转换为 1，小于 3.5 的值转换为 0。这样可以将连续的回归结果转换为二分类结果。
            labels_corr_2 = [int(p>=3.5) for p in labels_corr]  # predictions_corr_2 和 labels_corr_2 分别表示经过转换后的二分类预测结果和真实标签。
            acc_2 = accuracy_score(labels_corr_2, predictions_corr_2)   # acc_2 表示基于二分类的准确率，f1_2 表示基于二分类的加权 F1 分数。
            f1_2 = f1_score(labels_corr_2, predictions_corr_2, average='weighted')
            return {
                'mae': mae,
                'corr': corr,
                '2-cls_acc': acc_2,
                '2-f1': f1_2,
            }
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def build_message(epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score):  # 生成一个包含当前轮次训练集和验证集的损失值和评估结果的日志消息。传入参数为当前轮次，训练集的损失值和评估结果字典，验证集的损失值和评估结果字典，测试集的损失值和评估结果字典。
    msg = "Epoch:[{:3.0f}]".format(epoch + 1)   # 定义一个字符串msg，包含当前轮次的信息，使用format方法将epoch加1后格式化为三位整数。

    msg += " ||"    # 将字符串" ||"添加到msg末尾
    msg += " TrainLoss:[{0:.3f}]".format(train_loss)    # 将训练集的损失值格式化为浮点数，并将其添加到msg末尾。
    for key in train_score.keys():  # 对训练集的评估结果字典中的每个键进行迭代。
        msg += " Train_"+key+":[{0:6.3f}]".format(train_score[key]) # 将每个评估结果的键和对应的值格式化为浮点数，并将它们添加到msg末尾。

    msg += " ||"
    msg += " ValLoss:[{0:.3f}]".format(val_loss)    # 将验证集的损失值格式化为浮点数，并将其添加到msg末尾。
    for key in val_score.keys():    # 对验证集的评估结果字典中的每个键进行迭代。
        msg += " Val_"+key+":[{0:6.3f}]".format(val_score[key]) # 将每个评估结果的键和对应的值格式化为浮点数，并将它们添加到msg末尾。

    return msg  # 返回构建好的日志消息字符串msg


def build_single_message(best_score, mode): # 构建一个包含单个评分结果的日志消息。输入参数为当前最高评分结果的字典，包含不同指标的评分值，和一个模式字符串，用于标识评分结果类型。
    msg = mode  # 将模式字符串赋值给变量msg，初始化日志消息
    for key in best_score.keys():   # 遍历评分结果字典的键。
        msg += " "+key+":[{0:6.3f}]".format(best_score[key])    # 将每个评分结果的键和对应的值格式化为浮点数，并添加到日志消息msg中。
    return msg  # 返回构建好的日志消息字符串。消息的格式为"[表示当前结果的模式字符串] [评分结果键1]:[评分结果值1] [评分结果键2]:[评分结果值2] ..."，其中mode是输入参数指定的模式，用于区分不同的评分结果类型。


def log_tf_board(writer, epoch, train_loss, train_score, val_loss, val_score, test_loss, test_score, lr_schedule):  # 用于将训练和验证过程中的指标写入TensorBoard。传入参数分别是：TensorBoard的写入器，当前epoch，训练集损失，训练集指标（字典），验证集损失，验证集指标（字典），测试集损失，测试集指标（字典），学习率调度器对象用于获取当前学习率。
    writer.add_scalar('Train/Epoch/Loss', train_loss, epoch)    # 将训练集上的损失值写入TensorBoard，使用标签为'Train/Epoch/Loss'，X轴为epoch，Y轴为train_loss。
    for key in train_score.keys():  # 对训练集上的每个指标进行迭代。
        writer.add_scalar('Train/Epoch/'+key, train_score[key], epoch)  # 将训练集上的每个指标值写入TensorBoard，使用对应的标签和epoch，X轴为epoch，Y轴为指标。
    writer.add_scalar('Valid/Epoch/Loss', val_loss, epoch)  # 将验证集上的损失值写入TensorBoard，使用标签为'Valid/Epoch/Loss'，X轴为epoch，Y轴为val_loss。
    for key in val_score.keys():    # 对验证集上的每个指标进行迭代。
        writer.add_scalar('Valid/Epoch/'+key, val_score[key], epoch)    # 将验证集上的每个指标值写入TensorBoard，使用对应的标签和epoch，X轴为epoch，Y轴为指标。
    try:    # 尝试
        writer.add_scalar('Lr',  lr_schedule.get_last_lr()[-1], epoch)  # 将学习率的最后一个值写入TensorBoard，使用标签'Lr'，X轴为epoch，Y轴为lr_schedule.get_last_lr()[-1]。lr_schedule是学习率调度器对象，通过get_last_lr()方法获取当前学习率的最后一个值。
    except: # 如果异常
        pass    # 继续执行后续


if __name__ == "__main__":  # 当脚本作为主模块直接运行时，__name__变量的值为__main__
    import faulthandler # 脚本会导入名为faulthandler的模块，并调用其enable()函数。
    faulthandler.enable()   # faulthandler模块是用于处理Python程序中的崩溃和异常的调试工具。通过调用enable()函数，可以启用faulthandler模块的功能，从而在程序发生崩溃或异常时提供更详细的诊断信息。
    main()  # 打开诊断模块后，进入main函数正式训练模型。
