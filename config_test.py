


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
    n2same = dict(
        lr = 0.0004, #TODO: verringere nachh 5.000 steps um 0.5
        batchNorm = True,
        dropout = 0,
        net = 'U_Net',
        sheduler = False, #war eigentlich True

        skip_first_connection = True,
        featurLayer = 96,
        lambda_inv=2,
        changeLR_steps = 5000,
        changeLR_rate = -0.5,

        augmentation = False,
        dropout_rate=0,
    ),
    n2info = dict(
        lr = 0.0001,
        batchNorm = True,
        dropout = 0,
        net = 'U_Net',
        sheduler = False, #war eigentlich True
        validation_dataset_size = 100,

        featurLayer = 96,
        predictions = 100, #kmc
        changeLR_steps = 5000,
        changeLR_rate = -0.5,

        augmentation = False,
        lambda_inv=2,
        dropout_rate=0,
    ),

    
)