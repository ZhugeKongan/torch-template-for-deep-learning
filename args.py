import os

class data_config:
    model_name = "baseline"

    '''***********- dataset and directory-*************'''
    dataset='cifar100'
    input_size = 32
    num_class = 100
    data_path = '/disks/disk2/lishengyan/MyProject/Attention_and_CifarModel/dataset'
    train_file=''
    val_file = ''
    test_file = ''
    MODEL_PATH = '/disks/disk2/lishengyan/MyProject/Attention_and_CifarModel/ckpts/cifar100/resnet50/esebase_fselectmixno/'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    '''***********- Hyper Arguments-*************'''
    autoaug = 0  # Auto enhancement set to 1
    gpus=[0,1]  #[1,2,3]
    WORKERS = 5
    tensorboard= False
    epochs = 200
    batch_size = 128
    delta =0.00001
    rand_seed=40   #Fixed seed greater than 0
    lr=0.1
    warm=1#warm up training phase
    optimizer = "torch.optim.SGD"
    optimizer_parm = {'lr': lr,'momentum':0.9, 'weight_decay':5e-4, 'nesterov':False}
    # optimizer = "torch.optim.AdamW"
    # optimizer_parm = {'lr': 0.001, 'weight_decay': 0.00001}
    #学习率：小的学习率收敛慢，但能将loss值降到更低。当使用平方和误差作为成本函数时，随着数据量的增多，学习率应该被设置为相应更小的值。adam一般0.001，sgd0.1，batchsize增大，学习率一般也要增大根号n倍
    #weight_decay:通常1e-4——1e-5，值越大表示正则化越强。数据集大、复杂，模型简单，调小；数据集小模型越复杂，调大。
    scheduler ="torch.optim.lr_scheduler.MultiStepLR"
    scheduler_parm ={'milestones':[60,120,160], 'gamma':0.2}
    # scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    # scheduler_parm = {'T_max': 200, 'eta_min': 1e-4}
    # scheduler = "torch.optim.lr_scheduler.StepLR"
    # scheduler_parm = {'step_size':1000,'gamma': 0.65}
    # scheduler = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    # scheduler_parm = {'mode': 'min', 'factor': 0.8,'patience':10, 'verbose':True,'threshold':0.0001, 'threshold_mode':'rel', 'cooldown':2, 'min_lr':0, 'eps':1e-08}
    # scheduler = "torch.optim.lr_scheduler.ExponentialLR"
    # scheduler_parm = {'gamma': 0.1}
    loss_f ='torch.nn.CrossEntropyLoss'
    loss_dv = 'torch.nn.KLDivLoss'
    loss_fn = 'torch.nn.BCELoss' # loss_fn = 'torch.nn.BCEWithLogitsLoss'  # loss_fn='torch.nn.MSELoss'
    fn_weight =[3.734438666137167, 1.0, 1.0, 1.0, 3.5203138607843196, 3.664049338245769, 3.734438666137167, 3.6917943287286734, 1.0, 3.7058695139403963, 1.0, 2.193419513003608, 3.720083373160097, 3.6917943287286734, 3.734438666137167, 1.0, 2.6778551377707998]


