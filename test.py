import torch
from evaluation import ThreeDEvaluator
# from dig.threedgraph.method import run
from run import run
from PaiNN import PainnModel
import logging
import time

if  __name__ == '__main__':
    logger = logging.getLogger()
    time = time.asctime()
    logger.setLevel(logging.INFO)   # 设置打印级别
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(f"debug "+time+".log", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.info('Start print log......')

torch.manual_seed(2048)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load the dataset and split
# train_dataset = torch.load('/mnt/data/ai4phys/xuhan/quantile_evidential_deep_learning/dataset/newliquid_shifted.pt')[:1000]
train_dataset = torch.load('/root/xuhan/quantile_evidential_deep_learning/dataset/1k+1k.pt')[:6000] #2200
for i in range(len(train_dataset)):
    train_dataset[i].cell = train_dataset[i].cell.unsqueeze(0)

valid_dataset = torch.load('/root/xuhan/quantile_evidential_deep_learning/dataset/5th.pt')[:200]
for i in range(len(valid_dataset)):
    valid_dataset[i].cell = valid_dataset[i].cell.unsqueeze(0)

test_dataset = torch.load('/root/xuhan/quantile_evidential_deep_learning/dataset/liquid_aimd.pt')[500:550] #/mnt/workspace/tangchenyu/dataset/Water/uncertainty/water_bias_2k_shifted.pt
for i in range(len(test_dataset)):
    test_dataset[i].cell = test_dataset[i].cell.unsqueeze(0)

device = 'cuda:0'

# Define model, loss, and evaluation
model = PainnModel(num_interactions=3,hidden_state_size=128,cutoff=5.0,pdb=True)

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          mol_name = '6th', energy_trans=[0, 0, 0], convert=1,
          epochs=2000, batch_size=2, vt_batch_size=8, lr=0.0001, lr_decay_factor=0.5, lr_decay_step_size=300,
          energy_and_force=True,
          save_dir='/root/xuhan/quantile_evidential_deep_learning/model/painn/evidence/newwater_uncertain/altest2/'
          )