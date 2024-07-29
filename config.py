


#global:
useSigma = True # use Sigmadb or Sigma for noise
sigma = 0.4   #only normal distribution
sigmadb = 2   # for SNR
save_model = False
std_lr = 0.003

#datasets
celeba_dir = 'dataset/celeba_dataset'
mnist_dir  = 'dataset/mnist_dataset'

#method std. config
methodes = dict(
    #in experiment there are two samples with the same noise level but independent noise
    n2noise_2_input = dict(
        lr = 0.003,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    n2noise_2_input_test = dict(#original with diffrent noise level
        lr = 0.003,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,

        secoundSigma = sigmadb+0.4 if useSigma else sigmadb+2,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    #in experiment there is only ONE sample
    n2noise_1_input = dict(
        lr = 0.003,
        batchNorm = False,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,

        secoundSigma = sigmadb+0.4 if useSigma else sigmadb+2,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    
    n2self = dict(
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
    n2score = dict(
        lr = 0.003, #TODO: check
        batchNorm = True,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    
    n2same = dict(
        lr = 0.0004, #TODO: verringere nachh 5.000 steps um 0.5
        batchNorm = True,
        dropout = 0,
        net = 'U_Net',
        sheduler = True,

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
        sheduler = True,
        validation_dataset_size = 100,

        featurLayer = 96,
        predictions = 100, #kmc
        changeLR_steps = 5000,
        changeLR_rate = -0.5,

        augmentation = False,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    n2void = dict(
        lr = 0.0004,
        batchNorm = True,
        dropout = 0,
        net = 'U_Net',
        sheduler = False,

        augmentation = True,
        lambda_inv=-1,
        dropout_rate=0,
    ),
    s2self = dict(
        lr = 0.003,
        batchNorm = False,
        dropout = 0.3,
        net = 'P_U_Net',
        sheduler = False,
        augmentation = True,

        lambda_inv=-1,
        dropout_rate=0, #for masking (optional)
    ),
)