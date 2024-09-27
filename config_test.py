


#global:
useSigma = False # use Sigmadb or Sigma for noise
sigma = 0.4   #only normal distribution
sigmadb = 2   # for SNR
save_model = False
std_lr = 0.003

#datasets
celeba_dir = 'dataset/celeba_dataset'
mnist_dir  = 'dataset/mnist_dataset'
strain_dir = "data/DAS/SIS-rotated_train_50Hz.npy"

#method std. config
methodes = dict(
    n2self1 = dict(
        lr = 0.001,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,
        erweiterung = 'j-invariant',

        radius = 1,
        grid_size = 4,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    n2self2 = dict(
        lr = 0.001,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,
        erweiterung = 'j-invariant',

        radius = 2,
        grid_size = 4,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    n2self3 = dict(
        lr = 0.001,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,
        erweiterung = 'j-invariant',

        radius = 3,
        grid_size = 4,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    n2self4 = dict(
        lr = 0.001,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,
        erweiterung = 'j-invariant',

        radius = 4,
        grid_size = 4,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    n2self5 = dict(
        lr = 0.001,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,
        erweiterung = 'j-invariant',

        radius = 5,
        grid_size = 4,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    n2self6 = dict(
        lr = 0.001,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,
        erweiterung = 'j-invariant',

        radius = 6,
        grid_size = 4,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),

    
)