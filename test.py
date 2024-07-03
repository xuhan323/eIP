import torch
from MD17 import MD17
from evaluation import ThreeDEvaluator
from PaiNN import PainnModel
from run import run
import logging
import time

if  __name__ == '__main__':
    logger = logging.getLogger()
    time = time.asctime()
    logger.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(f"debug "+time+".log", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.info('Start print log......')


# Load the dataset and split
dataset_md17 = MD17(root='dataset', name='aspirin')
split_idx_md17 = dataset_md17.get_idx_split(len(dataset_md17.data.y), train_size=1000, valid_size=1000, seed=42)
train_dataset, valid_dataset = dataset_md17[split_idx_md17['train']], dataset_md17[split_idx_md17['valid']]
train_energy_trans = torch.mean(torch.tensor([data.y for data in dataset_md17])).item()


dataset_md17_salicylic = MD17(root='dataset', name='salicylic')
split_idx_md17_asl = dataset_md17.get_idx_split(len(dataset_md17.data.y), train_size=1000, valid_size=1000, seed=42)
test_dataset = dataset_md17_salicylic[split_idx_md17_asl['test']][:1000]
test_energy_trans = torch.mean(torch.tensor([data.y for data in dataset_md17_salicylic])).item()



device = 'cuda:0'

# Define model, loss, and evaluation
# model = SchNet(energy_and_force=True, cutoff=5.0, num_layers=6, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50)
model = PainnModel(num_interactions=3,hidden_state_size=128,cutoff=5.0,pdb=False)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          mol_name = 'small molecule', energy_trans=[train_energy_trans, train_energy_trans, test_energy_trans], convert=1, LAMBDA = 1, THETA = 0.1, q = 0.4,
          epochs=30, batch_size=2, vt_batch_size=4, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=200,
          energy_and_force=True,
          save_dir='./'
          )