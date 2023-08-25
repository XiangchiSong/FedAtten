from DataLoaderCMUSDK import *  # 导入DataLoaderCMUSDK的所有内容

def get_data_loader(opt):   # 获取传入opt中的数据集名称
    dataset = opt.dataset
    text, audio, video=opt.text, opt.audio, opt.video # Only for CMUSDK dataset 用于获取传入参数opt中的文本、音频和视频相关参数
    normalize = opt.normalize   # 获取传入参数opt中的归一化参数
    persistent_workers=opt.persistent_workers   # 获取传入参数opt中的持久化工作进程数参数。persistent_workers是一个布尔值，指定是否保持工作进程的持久性，如果设置为True，则工作进程将在迭代器完成之前保持打开状态，以加快下一次迭代的速度；
    batch_size, num_workers, pin_memory, drop_last =opt.batch_size, opt.num_workers, opt.pin_memory, opt.drop_last
    # num_workers指定用于数据加载的工作进程数；pin_memory是一个布尔值，指定是否将张量数据复制到CUDA固定内存中，以加快数据传输速度；drop_last是一个布尔值，指定在数据样本数量不能被批次大小整除时，是否丢弃最后一个不完整的批次。
    assert dataset in ['mosi_SDK', 'mosei_SDK'] # 使用断言（assert）来验证变量dataset的取值是否为指定的两个数据集之一，即'mosi_SDK'和'mosei_SDK'。

    if 'mosi' in dataset:
        dataset_train = CMUSDKDataset(mode='train', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, ) # 创建对于mosi的训练集对象
        dataset_valid = CMUSDKDataset(mode='valid', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, ) # 创建对于mosi的验证集对象
        dataset_test = CMUSDKDataset(mode='test', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, )   # 创建对于mosi的测试集对象er类创建训练数据加载器：使用训练集；指定使用的批次样本数；指定使用DataLoaderCMUSDK中的multi_collate_mosei_mosi函数处理样本合成批次；并使用shuffle在每个训练迭代周期开始对训练数据随机洗牌增加训练随机性；
        data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=True,    # 使用DataLoad
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False,   #同训练集，但是样本不打乱，最后一个不完整批次不丢弃
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False,     #同训练集，但是样本不打乱，最后一个不完整批次不丢弃
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        return data_loader_train, data_loader_valid, data_loader_test


    if 'mosei' in dataset:
        dataset_train = CMUSDKDataset(mode='train', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, )    # 创建对于mosei的训练集对象
        dataset_valid = CMUSDKDataset(mode='valid', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, )    # 创建对于mosei的验证集对象
        dataset_test = CMUSDKDataset(mode='test', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, )  # 创建对于mosei的测试集对象
        data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=True,    #同mosi训练集
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False,   #同mosi验证集
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False,     #同mosi测试集
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        return data_loader_train, data_loader_valid, data_loader_test

    raise NotImplementedError

