from src.baselines.cgnn.cgnn import CGNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from typing import Dict, Any, Type


# MinNormSolver: Finds minimum-norm element in convex hull for MGDA loss balancing 
class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        dmin = 1e8
        sol = None
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu().item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu().item()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu().item()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))
        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element_FW(vecs):
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]
        if n < 3:
            return sol_vec, init_sol[2]
        iter_count = 0
        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]
        while iter_count < MinNormSolver.MAX_ITER:
            iter_count += 1
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))
            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
        return sol_vec, nd

    @staticmethod
    def gradient_normalizers(grads, losses, normalization_type):
        gn = {}
        if normalization_type == 'l2':
            for t in grads:
                gn[t] = torch.tensor(np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]])))
        elif normalization_type == 'loss':
            for t in grads:
                gn[t] = losses[t]
        elif normalization_type == 'loss+':
            for t in grads:
                gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
        elif normalization_type == 'none':
            for t in grads:
                gn[t] = torch.tensor(1.0)
        return gn


class ICL(CGNN):
    "ICL: https://github.com/haibin65535/ICL"
    def __init__(
        self,
        gnn_backbone: Type[nn.Module],
        in_channels: int,
        in_channels_e: int,
        num_classes: int,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any]
    ):
        super().__init__(
            gnn_backbone,
            in_channels,
            in_channels_e,
            num_classes,
            optimizer_hparams
        )
        self.cgnn_config.update(model_hparams)
        # Disable automatic optimization for MGDA (manual backward calls)
        self.automatic_optimization = False
    
    # Shared backbone parameters (for MGDA gradient extraction)
    def get_backbone_grads(self):
        """Extract gradients from the shared GNN backbone parameters."""
        grads = []
        for param in self.gnn.parameters():
            if param.grad is not None:
                grads.append(Variable(param.grad.data.clone(), requires_grad=False))
        if len(grads) == 0:
            # Fallback: return zero gradient tensors for all parameters
            grads = [Variable(torch.zeros_like(param.data), requires_grad=False) 
                    for param in self.gnn.parameters()]
        return grads
    
    # ICL Loss
    def compute_loss(self, c_logits, s_logits, combined_logits, labels):
        # Log_probs
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        combined_log_probs = F.log_softmax(combined_logits, dim=-1)

        # Compute losses
        uniform_target = torch.ones_like(s_log_probs, dtype=torch.float, device=s_log_probs.device) / self.num_classes
        c_loss = F.nll_loss(c_log_probs, labels)
        s_loss = F.kl_div(s_log_probs, uniform_target, reduction='batchmean')
        combined_loss = F.nll_loss(combined_log_probs, labels)
        return c_loss, s_loss, combined_loss

    # Training Step (with MGDA)
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        # Forward
        # Logits
        _, _, c_logits, s_logits, combined_logits = self(
            data=batch, 
            eval_random=self.cgnn_config['with_random'],
            c_i_f=False
        )
        
        # Labels
        labels = batch.y.long()

        # Loss
        c_loss, s_loss, combined_loss = self.compute_loss(
            c_logits=c_logits,
            s_logits=s_logits, 
            combined_logits=combined_logits, 
            labels=labels
        )

        # MGDA: compute per-loss gradients on backbone
        mgda_model = self.cgnn_config.get('mgda_model', 'loss+')
        loss_data = {}
        grads = {}

        # Causal loss gradient
        loss_data['c'] = c_loss.data
        self.manual_backward(c_loss, retain_graph=True)
        grads['c'] = self.get_backbone_grads()
        self.zero_grad()

        # Spurious loss gradient
        loss_data['s'] = s_loss.data
        self.manual_backward(s_loss, retain_graph=True)
        grads['s'] = self.get_backbone_grads()
        self.zero_grad()

        # Combined loss gradient (not used in MGDA, always weight=1)
        loss_data['combined'] = combined_loss.data
        self.manual_backward(combined_loss, retain_graph=True)
        grads['combined'] = self.get_backbone_grads()
        self.zero_grad()

        # MGDA solve for weights on spurious and causal losses
        loss_names = ['c', 's']
        gn = MinNormSolver.gradient_normalizers(grads, loss_data, mgda_model)
        for name in loss_names:
            if gn[name] < 1e-3:
                gn[name] = torch.tensor(1e-3)
        for t in loss_data:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
        
        #print("grads:", {t: [g.shape for g in grads[t]] for t in loss_names})
        try:
            sol, _ = MinNormSolver.find_min_norm_element_FW([grads[t] for t in loss_names])
        except TypeError:
            return None
        sol = {k: sol[i] for i, k in enumerate(loss_names)}

        # Final weighted loss
        total_loss = sol['c'] * c_loss + sol['s'] * s_loss + combined_loss

        self.manual_backward(total_loss)
        opt.step()

        # Logging
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return total_loss

    # Validation Step
    def validation_step(self, batch, batch_idx):
        # Logits
        h_c, h_s, c_logits, s_logits, combined_logits = self(
            data=batch, 
            eval_random=self.cgnn_config['eval_random'],
            c_i_f=False
        )
        
        # Log_probs
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        combined_log_probs = F.log_softmax(combined_logits, dim=-1)

        # Labels
        labels = batch.y.long()

        # Validation loss computation
        uniform_target = torch.ones_like(s_log_probs, dtype=torch.float, device=s_log_probs.device) / self.num_classes
        c_loss = F.nll_loss(c_log_probs, labels)
        s_loss = F.kl_div(s_log_probs, uniform_target, reduction='batchmean')
        combined_loss = F.nll_loss(combined_log_probs, labels)
        val_loss = c_loss + s_loss + combined_loss

        # Logging
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        # Predictions
        probs = self.predict_probabilities(logits=combined_logits)

        # Classification metrics
        self.val_metrics.update(probs, labels)

        return val_loss
    