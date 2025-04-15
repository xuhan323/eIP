
import time
import os
import torch
from torch.optim import Adam, AdamW
from torch import nn
from torch_geometric.loader import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import time
import logging
from torch.cuda.amp import autocast, GradScaler
import math
import pytorch_warmup as warmup
import logging
import time
scaler = GradScaler()
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

def mse(y, y_pred, reduce=True):
    ax = list(range(1, len(y.shape)))
    mse = torch.mean((y - y_pred) ** 2, dim=ax)
    return torch.mean(mse) if reduce else mse

def rmse(y, y_):
    rmse = torch.sqrt(torch.mean((y - y_) ** 2))
    return rmse

def gaussian_nll(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))
    logprob = -torch.log(sigma) - 0.5 * torch.log(2 * np.pi) - 0.5 * ((y - mu) / sigma) ** 2
    loss = torch.mean(-logprob, dim=ax)
    return torch.mean(loss) if reduce else loss

def gaussian_nll_logvar(y, mu, logvar, reduce=True):
    ax = list(range(1, len(y.shape)))
    log_liklihood = 0.5 * (
            -torch.exp(-logvar) * (mu - y) ** 2 - torch.log(2 * torch.tensor(np.pi, dtype=logvar.dtype)) - logvar
    )
    loss = torch.mean(-log_liklihood, axis=ax)
    return torch.mean(loss) if reduce else loss

def nig_nll(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)
    return torch.mean(nll) if reduce else nll

def kl_nig(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * torch.square(mu2 - mu1)) \
         + 0.5 * v2 / v1 \
         - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
         - 0.5 + a2 * torch.log(b1 / b2) \
         - (torch.lgamma(a1) - torch.lgamma(a2)) \
         + (a1 - a2) * torch.digamma(a1) \
         - (b1 - b2) * a1 / b1
    return KL

def nig_reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = torch.abs(y - gamma)
    if kl:
        kl = kl_nig(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + (alpha)
        reg = error * evi

    return torch.mean(reg) if reduce else reg

def evidential_regression_loss(coeff=1e-2):
    def func(y_true, evidential_output):
        gamma, v, alpha, beta = evidential_output
        loss_nll = nig_nll(y_true, gamma, v, alpha, beta)
        loss_reg = nig_reg(y_true, gamma, v, alpha, beta)
        return  loss_nll + coeff*loss_reg
    return func

class run():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        pass
        
    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs_p=500,epochs_u=500, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
        energy_and_force=False, p=100, save_dir='', log_dir='', energy_trans=[0, 0, 0], convert=1, mol_name=None):
        r"""
        The run script for training and validation.
        
        Args:
            device (torch.device): Device for computation.
            train_dataset: Training data.
            valid_dataset: Validation data.
            test_dataset: Test data.
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            loss_func (function): The used loss funtion for training.
            evaluation (function): The evaluation function. 
            epochs (int, optinal): Number of total training epochs. (default: :obj:`500`)
            batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
            vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0`)
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
            log_dir (str, optinal): The path to save log files. If set to :obj:`''`, will not save the log files. (default: :obj:`''`)
        
        """        

        model = model.to(device)
        self.convert = convert
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        best_valid = float('inf')
        best_test = float('inf')
        num_params = sum(p.numel() for p in model.parameters())
        epochs = epochs_p+epochs_u
        logging.info(f'#Params: {num_params}')
        logging.info(f'#modle-setting: task_mol:{mol_name},  epochs_p:{epochs_p}, epochs_u:{epochs_u}, train_bs:{batch_size}, infer_bs:{vt_batch_size}, initial_lr:{lr}, convert:{self.convert}, trans:{energy_trans} ' )
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        num_steps = len(train_loader) * epochs
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor) 
            
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
        
        train_mae_npy, validation_mae_npy, test_mae_npy, force_mae_npy, energy_mae_npy, time_npy = np.ones(epochs) , np.ones(epochs) , np.ones(epochs) , np.ones(epochs) , np.ones(epochs) , np.ones(epochs)
        start = time.time()
        debug_para = []
        for epoch in range(1, (epochs+1)):
            logging.info("\n=====Epoch {}".format(epoch))
        
            if epoch < (epochs_p):
                logging.info(" --------in potential function loss process-------")
                train_mae, e_aleatoric_uncertainty, e_epistemic_uncertainty, node_para = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device, trans=energy_trans[0])
            else:
                if epoch == epochs_p: 
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr*0.01
                logging.info(" --------in uncertainty loss process--------")
                train_mae, e_aleatoric_uncertainty, e_epistemic_uncertainty, node_para = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device, trans=0)
                
            valid_mae, energy_mae_val, force_mae_val, e_aleatoric_uncertainty_val, e_epistemic_uncertainty_val = self.val(model, valid_loader, energy_and_force, p, evaluation, device,trans=0)
            test_mae, energy_mae, force_mae, e_aleatoric_uncertainty_test, e_epistemic_uncertainty_test = self.test(model, test_loader, energy_and_force, p, evaluation, device, trans=0)

            end = time.time()
            
            logging.info({
                'Train loss': train_mae, 
                'Validation mae': valid_mae, 
                'Test mae': test_mae
                })
            logging.info(f'time cost: {end-start}')
            
            logging.info({
                'Force mae in validation': force_mae_val, 
                'Energy mae in validation': energy_mae_val
                })
            logging.info({
                'Force mae in test': force_mae, 
                'Energy mae in test': energy_mae
                })
                # logging.info({'e_aleatoric_uncertainty in train': e_aleatoric_uncertainty.mean().item(), 'e_epistemic_uncertainty in train': e_epistemic_uncertainty.mean().item()})
            logging.info({
                    #'e_aleatoric_uncertainty of ID-data': e_aleatoric_uncertainty_val.mean().item(), 
                    'graph_epistemic_uncertainty of ID-data': e_epistemic_uncertainty_val.mean().item()
                    })
            logging.info({
                    #'e_aleatoric_uncertainty of OOD-data': e_aleatoric_uncertainty_test.mean().item(), 
                    'graph_epistemic_uncertainty in OOD-data': e_epistemic_uncertainty_test.mean().item()
                    })

                # logging.info({'f_aleatoric_uncertainty in val': f_aleatoric_uncertainty_val.mean().item(), 'f_epistemic_uncertainty in val': f_epistemic_uncertainty_val.mean().item()})
                # logging.info({'f_aleatoric_uncertainty in test': f_aleatoric_uncertainty_test.mean().item(), 'f_epistemic_uncertainty in test': f_epistemic_uncertainty_test.mean().item()})



            train_mae_npy[epoch-1], validation_mae_npy[epoch-1], test_mae_npy[epoch-1], force_mae_npy[epoch-1], energy_mae_npy[epoch-1], time_npy[epoch-1] = train_mae, valid_mae, test_mae, energy_mae, force_mae, (end-start)

            if log_dir != '':
                writer.add_scalar('train_mae', train_mae, epoch)
                writer.add_scalar('valid_mae', valid_mae, epoch)
                writer.add_scalar('test_mae', test_mae, epoch)
            
            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                if save_dir != '':
                    logging.info('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'lr_scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{mol_name}.pt'))
                    torch.save(model, os.path.join(save_dir, f'model_{mol_name}.pth'))
            
            lr_epoch = optimizer.state_dict()['param_groups'][0]['lr']
            logging.info(f'learning_rate in epoch {epoch}: {lr_epoch}')

            scheduler.step()
            # lr_scheduler.step(valid_mae)

        logging.info(f'Best validation MAE so far: {best_valid}')
        logging.info(f'Test MAE when got best validation result: {best_test}')
        #np.save(f'debug_{mol_name}.npy', torch.tensor(debug_para).cpu().numpy())

        if save_mae:
            logging.info('----saving traing mae----')      
            np.save(f'train_mae_{SI}_{mol_name}.npy', train_mae_npy)
            np.save(f'validation_mae_{SI}_{mol_name}.npy', validation_mae_npy)
            np.save(f'test_mae_{SI}_{mol_name}.npy', test_mae_npy)
            np.save(f'force_mae_{SI}_{mol_name}.npy', force_mae_npy)
            np.save(f'energy_mae_{SI}_{mol_name}.npy', energy_mae_npy)
            np.save(f'time_{SI}_{mol_name}.npy', time_npy)
            logging.info('----successfully saved----')
        
        
        if log_dir != '':
            writer.close()

        return best_valid

    def train(self, model, optimizer, train_loader, energy_and_force, p, loss_func, device, trans):
        r"""
        The script for training.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function): The used loss funtion for training. 
            device (torch.device): The device where the model is deployed.

        :rtype: Traning loss. ( :obj:`mae`)
        
        """   
        model.train()
        loss_accum = 0
        uncert_loss = evidential_regression_loss()
        for step, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            if energy_and_force:
                
                e, force, node_para, aleatoric_uncertainty, epistemic_uncertainty = model(batch_data)
              
                force_norm = torch.norm(force, dim=1)
                e_loss = loss_func(e, (batch_data.y.unsqueeze(1)+trans)/(self.convert))
                f_loss = loss_func(force, (batch_data.force/self.convert))
                    
                batch_data.force = batch_data.force/self.convert
                node_evidential_loss = uncert_loss(force_norm, node_para)

                loss =  (p*f_loss +e_loss) + 0.0005 * node_evidential_loss

            else:
                loss = loss_func(out, batch_data.y.unsqueeze(1))
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            loss_accum += loss.detach().cpu().item()

        return loss_accum / (step + 1), aleatoric_uncertainty, epistemic_uncertainty, node_para


    def test(self, model, data_loader, energy_and_force, p, evaluation, device, trans=0):
        r"""
        The script for validation/test.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.

        :rtype: Evaluation result. ( :obj:`mae`)
        
        """   
        model.eval()

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        if energy_and_force:
            preds_force = torch.Tensor([]).to(device).to(torch.long)
            targets_force = torch.Tensor([]).to(device).to(torch.long)
        
        for step, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(device)
            
            if energy_and_force:

                e, force_hat, node_para, e_aleatoric_uncertainty, e_epistemic_uncertainty = model(batch_data)
                preds_force = torch.cat([preds_force.float(),force_hat.float().detach()], dim=0)
                targets_force = torch.cat([targets_force.float(),(batch_data.force).float()], dim=0)
                    
            preds = torch.cat([preds, e.detach()], dim=0)
            targets = torch.cat([targets, ((batch_data.y.unsqueeze(1)+trans)/(self.convert)).long()], dim=0)

        input_dict = {"y_true": targets.to(torch.float), "y_pred": preds.to(torch.float)}

        if energy_and_force:
            input_dict_force = {"y_true": targets_force.to(torch.float), "y_pred": preds_force.to(torch.float)}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']

            return energy_mae + p * force_mae, energy_mae, force_mae, e_aleatoric_uncertainty, e_epistemic_uncertainty 

        return evaluation.eval(input_dict)['mae']


    def val(self, model, data_loader, energy_and_force, p, evaluation, device, trans=0):
        r"""
        The script for validation/test.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.

        :rtype: Evaluation result. ( :obj:`mae`)
        
        """   
        model.eval()

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        if energy_and_force:
            preds_force = torch.Tensor([]).to(device).to(torch.long)
            targets_force = torch.Tensor([]).to(device).to(torch.long)
        
        for step, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(device)
            
            if energy_and_force:
                e, force_hat, node_para, e_aleatoric_uncertainty, e_epistemic_uncertainty = model(batch_data)
                preds_force = torch.cat([preds_force.float(),force_hat.float().detach()], dim=0)
                targets_force = torch.cat([targets_force.float(),(batch_data.force).float()], dim=0)
                    
            preds = torch.cat([preds, e.detach()], dim=0)
            targets = torch.cat([targets, ((batch_data.y.unsqueeze(1)+trans)/(self.convert)).long()], dim=0)


        input_dict = {"y_true": targets.to(torch.float), "y_pred": preds.to(torch.float)}
        

        if energy_and_force:
            input_dict_force = {"y_true": targets_force.to(torch.float), "y_pred": preds_force.to(torch.float)}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
                
            return energy_mae + p * force_mae, energy_mae, force_mae, e_aleatoric_uncertainty, e_epistemic_uncertainty 
            
