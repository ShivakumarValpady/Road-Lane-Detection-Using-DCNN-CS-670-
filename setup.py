import torch
############################initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device==torch.device("cuda"):
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("Device:", device)
####################configuration
#'''
training_dir = '/Users/vignesh/Documents/Computer vision/project_ld/lane-detection/dataset/train_set/clips/'
training_sub = ['0601', '0531', '0313-1', '0313-2']
training_labels = ['/Users/vignesh/Documents/Computer vision/project_ld/lane-detection/dataset/train_set/label_data_0601.json',
           '/Users/vignesh/Documents/Computer vision/project_ld/lane-detection/dataset/train_set/label_data_0531.json',
           '/Users/vignesh/Documents/Computer vision/project_ld/lane-detection/dataset/train_set/label_data_0313.json']

test_dir = '/Users/vignesh/Documents/Computer vision/project_ld/lane-detection/dataset/test_set/clips/'
test_sub = ['0601', '0531', '0530']
test_labels = ['/Users/vignesh/Documents/Computer vision/project_ld/lane-detection/dataset/test_set/test_labels.json']

#'''
#####################################
'''
training_dir = '/home/ajawalimalli/cv_project/lane-detection/datasets/train_set/clips'
training_sub = ['0601', '0531', '0313-1', '0313-2']
training_labels = ['/home/ajawalimalli/cv_project/lane-detection/datasets/train_set/label_data_0601.json',
           '/home/ajawalimalli/cv_project/lane-detection/datasets/train_set/label_data_0531.json',
           '/home/ajawalimalli/cv_project/lane-detection/datasets/train_set/label_data_0313.json']

test_dir = '/home/ajawalimalli/cv_project/lane-detection/datasets/test_set/clips/'
test_sub = ['0601', '0531', '0530']
test_labels = ['/home/ajawalimalli/cv_project/lane-detection/datasets/test_set/test_label.json']
'''
##################################
output_dir = 'visual-results/'

class Configs:
   
    def __init__(self):
        # hyperparameters
        self.epochs = 2
        self.init_lr = 0.001
        self.batch_size = 12
        self.test_batch = 100
        self.workers = 4
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.loss_weights = [0.02, 1.02]
        self.hidden_dims = [512, 512]        
        self.decoder_config = [512, 512, 256, 128, 64]
        self.load_model = False

