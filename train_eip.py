import torch

from dig.threedgraph.evaluation import ThreeDEvaluator
from run import run
import pdb

# Load the dataset and split
train_dataset = torch.load('./dataset/silica_train.pt')[:1200]
valid_dataset = torch.load('./dataset/silica_train.pt')[1200:]
test_dataset = torch.load('./dataset/silica_test.pt')
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))
device = 'cuda:1'

# Define model, loss, and evaluation

from PaiNN import PainnModel
model = PainnModel(num_interactions=3,hidden_state_size=128,cutoff=6.0,pdb=True)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          epochs=1000, batch_size=2, vt_batch_size=2, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=200,energy_and_force=True)