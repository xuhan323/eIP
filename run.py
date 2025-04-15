import time
import os
import torch
from torch.optim import Adam, AdamW, Optimizer
from torch import nn
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import numpy as np
import time
import logging
from torch.cuda.amp import autocast, GradScaler
import math
import pytorch_warmup as warmup
import logging
import time
import torch.distributions as dist
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim.swa_utils as swa_utils 

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

def NIG_NLL(y, gamma, v, alpha, beta, w_i_dis, quantile, reduce=True):
    tau_two = 2.0 / (quantile * (1.0 - quantile))
    twoBlambda = 2.0 * 2.0 * beta * (1.0 + tau_two * w_i_dis.mean * v)

    nll = 0.5 * torch.log(torch.tensor(np.pi / v)) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1) ** 2) \
        + 0.5 * v2 / v1 \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 + a2 * torch.log(b1 / b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2) * torch.digamma(a1) \
        - (b1 - b2) * a1 / b1
    return KL

def tilted_loss(q, e):
    return torch.max(q * e, (q - 1) * e)

def NIG_Reg(y, gamma, v, alpha, beta, w_i_dis, quantile, omega=0.01, reduce=True, kl=False):
    tau_two = 2.0 / (quantile * (1.0 - quantile))
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    error = tilted_loss(quantile, y - gamma)
    w = torch.abs(torch.tensor(quantile - 0.5))

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + alpha + 1 / beta
        reg = error * evi

    return torch.mean(reg) if reduce else reg

def NIG_Reg_Improved(y, gamma, v, alpha, beta, w_i_dis, quantile, reduce=True):
    tau_two = 2.0 / (quantile * (1.0 - quantile))
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    error = tilted_loss(quantile, torch.abs(y - gamma))
    diff = torch.abs(y - gamma)
    eps = 1e-6 
    reg = -diff * torch.log(torch.exp(alpha - 1) - 1 + eps)
    
    return torch.mean(reg) if reduce else reg

def quant_evi_loss(y_true, gamma, v, alpha, beta, quantile=0.5, coeff=0.1, reduce=True):
    theta = (1.0 - 2.0 * quantile) / (quantile * (1.0 - quantile))
    mean_ = beta / (alpha - 1)
    w_i_dis = dist.Exponential(rate=1 / mean_)
    mu = gamma + theta * w_i_dis.mean
    loss_nll = NIG_NLL(y_true, mu, v, alpha, beta, w_i_dis, quantile, reduce=reduce)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, w_i_dis, quantile, reduce=reduce)
    loss_reg_in = NIG_Reg_Improved(y_true, gamma, v, alpha, beta, w_i_dis, quantile, reduce=reduce)
    total_loss = loss_nll + coeff * loss_reg
    
    return total_loss


class run():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        pass
        
        
    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=500, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
        energy_and_force=False, p=100, save_dir='', log_dir='', energy_trans=[0, 0, 0], convert=1, mol_name=None):

        model = model.to(device)
        self.convert = convert
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        best_valid = float('inf')
        best_test = float('inf')
        num_params = sum(p.numel() for p in model.parameters())
        
        logging.info(f'#Params: {num_params}')
        logging.info(f'#modle-setting: task_mol:{mol_name}, epochs:{epochs}, train_bs:{batch_size}, infer_bs:{vt_batch_size}, initial_lr:{lr}, convert:{self.convert}, trans:{energy_trans}')

        # 1. 优化器设置：区分权重衰减
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if 'bias' in name or 'ln' in name or 'bn' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

        # 2. 学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # 每次将学习率降低为原来的一半
            patience=30,  # 10个epoch没有改善就降低学习率
            verbose=True,
            min_lr=1e-6  # 最小学习率限制
        )

        # 3. SWA设置
        swa_model = None
        swa_scheduler = None
        swa_start = int(epochs * 0.9)  # 在训练后期开始使用SWA
        if epochs > 50:  # 只在epochs足够多时使用SWA
            swa_model = torch.optim.swa_utils.AveragedModel(model)
            # 设置SWA的学习率为初始学习率的1/10
            swa_scheduler = torch.optim.swa_utils.SWALR(
                optimizer, 
                swa_lr=lr * 0.1,
                anneal_epochs=5,
                anneal_strategy='cos'
            )

        # 4. 训练相关参数
        max_grad_norm = 1.0
        patience_early_stopping = 700
        patience_counter = 0

        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
        
        train_mae_npy = np.ones(epochs)
        validation_mae_npy = np.ones(epochs)
        test_mae_npy = np.ones(epochs)
        force_mae_npy = np.ones(epochs)
        energy_mae_npy = np.ones(epochs)
        time_npy = np.ones(epochs)
        
        start = time.time()
        debug_para = []

        for epoch in range(1, epochs + 1):
            logging.info(f"\n=====Epoch {epoch}")
            
            # 训练阶段
            train_mae, node_para = self.train(model, optimizer, train_loader, 
                                            energy_and_force, p, loss_func, device, 
                                            trans=energy_trans[0])

            # 验证阶段
            valid_mae, energy_mae_val, force_mae_val = self.val(
                model, valid_loader, energy_and_force, p, evaluation, device, trans=0)

            # 测试阶段
            test_mae, energy_mae, force_mae = self.test(
                model, test_loader, energy_and_force, p, evaluation, device, trans=0)

            end = time.time()
            
            # 记录训练信息
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

            if log_dir != '':
                writer.add_scalar('train_mae', train_mae, epoch)
                writer.add_scalar('valid_mae', valid_mae, epoch)
                writer.add_scalar('test_mae', test_mae, epoch)
            
            # 保存最佳模型
            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                patience_counter = 0
                if save_dir != '':
                    logging.info('Saving checkpoint...')
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_valid_mae': best_valid,
                        'num_params': num_params
                    }
                    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{mol_name}.pt'))
                    #torch.save(model, os.path.join(save_dir, f'model_{mol_name}.pth'))
            else:
                patience_counter += 1

            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Current learning rate: {current_lr}')

            # SWA更新
            if swa_model is not None and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                # 常规学习率调度
                scheduler.step(valid_mae)

            # 记录最佳结果
            logging.info(f'Best validation MAE so far: {best_valid}')
            logging.info(f'Best validation MAE of energy and force so far: force:{force_mae_val}, energy{energy_mae_val}')
            logging.info(f'Test MAE when got best validation result: {best_test}')
            logging.info(f'Best test MAE of energy and force so far: {force_mae}, {energy_mae}')

            # 检查早停
            if patience_counter >= patience_early_stopping:
                logging.info(f'Early stopping triggered after {epoch} epochs')
                break

            # 记录训练指标
            train_mae_npy[epoch-1] = train_mae
            validation_mae_npy[epoch-1] = valid_mae
            test_mae_npy[epoch-1] = test_mae
            force_mae_npy[epoch-1] = force_mae
            energy_mae_npy[epoch-1] = energy_mae
            time_npy[epoch-1] = end - start

        # 在训练结束时保存SWA模型
        if swa_model is not None:
            swa_utils.update_bn(train_loader, swa_model)
            torch.save(swa_model.state_dict(), os.path.join(save_dir, f'swa_model_{mol_name}.pt'))

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
        # uncert_loss = evidential_regression_loss()
        for step, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            if energy_and_force:
                # print(batch_data)
                # input()
                e, force, node_para = model(batch_data)
                e += (1921.62)
                # print(batch_data)
                # input()

                e_loss = loss_func(e, batch_data.energy.unsqueeze(1))
                loss_uncer = quant_evi_loss(batch_data.force[:,0], node_para[0][:,0],node_para[1][:,0],node_para[2][:,0],node_para[3][:,0])+quant_evi_loss(batch_data.force[:,1], node_para[0][:,1],node_para[1][:,1],node_para[2][:,1],node_para[3][:,1])+quant_evi_loss(batch_data.force[:,2], node_para[0][:,2],node_para[1][:,2],node_para[2][:,2],node_para[3][:,2])

                loss = 10000*loss_uncer + 0.1*e_loss

            else:
                loss = loss_func(out, batch_data.y.unsqueeze(1))
                

            optimizer.zero_grad()  # 直接调用 AdamW 的 zero_grad 方法
            loss.backward()  # 反向传播
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # 更新参数
            loss_accum += loss.detach().cpu().item()

        return loss_accum / (step + 1), node_para


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

                e, force_hat, node_para= model(batch_data)
                e += (1921.62)
                preds_force = torch.cat([preds_force.float(),force_hat.float().detach()], dim=0)
                targets_force = torch.cat([targets_force.float(),(batch_data.force).float()], dim=0)
                    
            preds = torch.cat([preds, e.detach()], dim=0)
            targets = torch.cat([targets, ((batch_data.energy.unsqueeze(1)))], dim=0)

        input_dict = {"y_true": targets.to(torch.float), "y_pred": preds.to(torch.float)}

        if energy_and_force:
            input_dict_force = {"y_true": targets_force.to(torch.float), "y_pred": preds_force.to(torch.float)}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
            return energy_mae + p * force_mae, energy_mae, force_mae

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
                e, force_hat, node_para= model(batch_data)
                e += (1921.62)
                preds_force = torch.cat([preds_force.float(),force_hat.float().detach()], dim=0)
                targets_force = torch.cat([targets_force.float(),(batch_data.force).float()], dim=0)
                    
            preds = torch.cat([preds, e.detach()], dim=0)
            targets = torch.cat([targets, (batch_data.energy.unsqueeze(1))], dim=0)


        input_dict = {"y_true": targets.to(torch.float), "y_pred": preds.to(torch.float)}
        

        if energy_and_force:
            input_dict_force = {"y_true": targets_force.to(torch.float), "y_pred": preds_force.to(torch.float)}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
                
            return energy_mae + p * force_mae, energy_mae, force_mae
            
