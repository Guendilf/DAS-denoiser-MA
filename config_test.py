


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
    )
)