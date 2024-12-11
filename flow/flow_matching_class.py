
# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
from einops import rearrange
import traceback
import numpy as np
import torch
import torch as th
from tqdm import tqdm
from diffusion.nn import sum_flat
from data_loaders.humanml.scripts import motion_process
import torch
import torch.nn as nn
import torch.nn.functional as F

# from zuko.utils import odeint
from torchdiffeq import odeint_adjoint as odeint
import wandb

from utils.dist_util import is_rank_zero
from .truncated_guassian import TruncatedNormal
from torchmetrics import CatMetric

import torch
import time

def skew_symmetric_matrix_batch(v):
    """将3D向量v批量转换为对应的反对称矩阵"""
    batch_size, time_steps, channels, num_vectors, _ = v.shape
    
    # 初始化反对称矩阵
    omega_hat = torch.zeros((batch_size, time_steps, channels, num_vectors, 3, 3), device=v.device)
    
    omega_hat[..., 0, 1] = -v[..., 2]
    omega_hat[..., 0, 2] = v[..., 1]
    omega_hat[..., 1, 0] = v[..., 2]
    omega_hat[..., 1, 2] = -v[..., 0]
    omega_hat[..., 2, 0] = -v[..., 1]
    omega_hat[..., 2, 1] = v[..., 0]
    
    return omega_hat

# 使用近似算法计算矩阵的指数
def matrix_exp_approximation(matrix, order=10):
    I = torch.eye(matrix.size(0), device=matrix.device)
    exp_matrix = I + matrix
    fact = 1
    power = matrix
    for i in range(2, order + 1):
        fact *= i
        power = torch.mm(power, matrix)
        exp_matrix += power / fact
    return exp_matrix

def lie_algebra_to_lie_group_batch(omega):
    """将李代数向量批量转换为李群矩阵"""
    omega_hat = skew_symmetric_matrix_batch(omega)  # 批量转换为反对称矩阵
    # rotation_matrix = torch.matrix_exp(omega_hat)  # 使用PyTorch的matrix_exp批量计算矩阵指数
    rotation_matrix = torch.matrix_exp(omega_hat)
    return rotation_matrix

def process_input_tensor(input_tensor):
    """
    将输入的维度为 [64, 251, 1, 196] 的张量处理，转换第四个维度中的前66个关节数据
    """
    device = input_tensor.device
    
    # 提取张量的形状
    batch_size, time_steps, channels, feature_dim = input_tensor.shape
    
    # 确保输入数据符合预期的形状
    assert feature_dim >= 66, "输入张量的第四个维度必须至少为66"
    
    # 提取需要处理的66个李代数向量
    joint_data = input_tensor[..., :66].reshape(batch_size, time_steps, channels, 22, 3)  # 形状 [64, 251, 1, 22, 3]

    # 将22个李代数向量批量转换为旋转矩阵
    rotation_matrices = lie_algebra_to_lie_group_batch(joint_data)  # 形状 [64, 251, 1, 22, 3, 3]
    
    # 将旋转矩阵展平为9维向量，并拼接为198维向量
    flattened_matrices = rotation_matrices.reshape(batch_size, time_steps, channels, 198)  # 形状 [64, 251, 1, 198]
    
    # 如果只需要保留原来的196维
    # cropped_flattened_matrices = flattened_matrices[..., :196]  # 形状 [64, 251, 1, 196]
    
    # 或者只用部分替换前66个数据
    final_output = torch.cat((flattened_matrices, input_tensor[..., 66:]), dim=-1)  # 形状 [64, 251, 1, 196]
    del joint_data
    del rotation_matrices
    del flattened_matrices
    torch.cuda.empty_cache()

    return final_output

def compute_smooth_loss(rotation_matrices,mask):
    """
    计算旋转矩阵的时序平滑损失，基于相邻时间步的旋转矩阵之间的差异。
    
    :param rotation_matrices: 旋转矩阵，形状为 [batch_size, time_steps, 1, 66]
    :return: 平滑损失
    """
    # 获取批次大小和时间步数
    batch_size, time_steps, _, num_joints = rotation_matrices.shape
    # rotation_matrices = rotation_matrices * mask.float()
    # 计算相邻旋转矩阵之间的差异
    loss = 0.0
    for t in range(time_steps - 1):
        # 获取相邻时间步的旋转矩阵（前66维）
        R_t = rotation_matrices[:, t, :, :66]  # 形状为 [batch_size, 66]
        R_t_next = rotation_matrices[:, t + 1, :, :66]  # 形状为 [batch_size, 66]
        
        # 计算旋转矩阵之间的差异（使用 Frobenius 范数）
        diff = R_t_next - R_t
        loss += torch.norm(diff, p='fro', dim=-1)  # 计算 Frobenius 范数并求和

    loss = loss / (time_steps - 1)
    # 对所有时间步的损失求平均
    return loss.mean()




class FlowMatching:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42


    """

    def __init__(
        self,
        *,
        lambda_rcxyz=0.0,
        lambda_vel=0.0,
        data_rep="rot6d",
        lambda_root_vel=0.0,
        lambda_vel_rcxyz=0.0,
        lambda_fc=0.0,
    ):
        self.data_rep = data_rep
        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc

        self.l2_loss = (
            lambda a, b: (a - b) ** 2
        )  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(
            loss * mask.float()
        )  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val

    @torch.no_grad()
    def sample_euler_raw(self, model, z_orig, N, model_kwargs, ode_kwargs):
        dt = 1.0 / N
        traj = []  # to store the trajectory

        z = z_orig.detach().clone()
        bs = len(z)

        est = []
        return_x_est = ode_kwargs["return_x_est"]
        if return_x_est:
            return_x_est_num = ode_kwargs["return_x_est_num"]
            est_ids = [int(i * N / return_x_est_num) for i in range(return_x_est_num)]

        traj.append(z.detach().clone())
        for i in range(0, N, 1):
            t = torch.ones(bs, device=z_orig.device) * i / N
            pred = model(z, t, **model_kwargs)

            _est_now = z + (1 - i * 1.0 / N) * pred
            est.append(_est_now.detach().clone())

            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        if return_x_est:
            est = [est[i].unsqueeze(0) for i in est_ids]
            est = torch.cat(est, dim=0)
            est = rearrange(est, "t b w h c -> (t b) w h c")
            return traj[-1], est
        else:
            return traj[-1]

    @torch.no_grad()
    def sample_euler_replacement_edit_till(
        self, model, z_orig, N, edit_till, model_kwargs=None, ode_kwargs=None
    ):
        inpainting_mask, inpainted_motion = (
            model_kwargs["y"]["inpainting_mask"],
            model_kwargs["y"]["inpainted_motion"],
        )

        dt = 1.0 / N
        traj = []  # to store the trajectory
        z = z_orig.detach().clone()
        batchsize = len(z)

        est = []
        return_x_est = ode_kwargs["return_x_est"]
        if return_x_est:
            return_x_est_num = ode_kwargs["return_x_est_num"]
            est_ids = [int(i * N / return_x_est_num) for i in range(return_x_est_num)]

        traj.append(z.detach().clone())
        for i in range(0, N, 1):
            t = torch.ones((batchsize), device=z_orig.device) * i / N

            _inpainted_motion = (z_orig * (N - i) + inpainted_motion * i) / N
            if i * 1.0 / N <= edit_till:
                z = (z * ~inpainting_mask) + (_inpainted_motion * inpainting_mask)

            pred = model(z, t, **model_kwargs)

            _est_now = z + (1 - i * 1.0 / N) * pred
            est.append(_est_now.detach().clone())

            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        if return_x_est:
            est = [est[i].unsqueeze(0) for i in est_ids]
            est = torch.cat(est, dim=0)
            est = rearrange(est, "t b w h c -> (t b) w h c")
            return traj[-1], est
        else:
            return traj[-1]

    @torch.no_grad()
    def cal_curveness(self, model, z_orig, N, model_kwargs):
        print(f"cal_curveness, N={N}")
        dt = 1.0 / N
        traj = []  # to store the trajectory
        preds = []
        z = z_orig.detach().clone()
        bs = len(z)

        func = lambda t, x: model(x, t, **model_kwargs)
        target = (
            odeint(
                func,
                z,
                # 0.0,
                torch.tensor([0.0, 1.0], device=z_orig.device, dtype=z_orig.dtype),
                # phi=self.parameters(),
                rtol=1e-5,
                atol=1e-5,
                method="dopri5",
                adjoint_params=(),
                # **ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]
            .detach()
            .clone()
        )
        # pre-compute the target, as it's too memory-consuming to save the intermediate results

        traj.append(z.detach().clone())
        for i in tqdm(range(0, N, 1), desc="cal_curveness", total=N):
            t = torch.ones(bs, device=z_orig.device) * i / N
            pred = model(z, t, **model_kwargs)
            pred = pred.detach().clone()
            preds.append((pred - target).pow(2).mean().item())
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        result = sum(preds) / len(preds)
        print("curveness: ", result)
        return result



    def p_sample_loop(
        self,
        model,
        y,
        shape,
        noise=None,
        ode_kwargs=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device="cuda",
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        sample_steps=None,  # backward compatibility, never use it
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        """
        带有调用位置追踪的 Cost Function。
        """

        y = y.to("cuda")
        
        #***************************************************************************************************************************************
        tg_gaussian = TruncatedNormal(torch.Tensor([0]), torch.Tensor([1]), -2, 2)
        # if noise is None:
        #     # print("noise is None, use randn instead")
            
        #     # noise = tg_gaussian.rsample(shape).squeeze(-1).to("cuda")
        #     # 假设 x_start 是输入的张量，形状为 [64, 251, 1, 196]
            

        #     first_66_features = tg_gaussian.rsample(shape).squeeze(-1).to("cuda")[:, :, :, :66]

        #     # 生成后 130 个特征，使用标准高斯分布
        #     last_130_features = torch.randn(shape[0], shape[1], shape[2], 130).to("cuda")

        #     # 将前 66 个和后 130 个特征拼接起来
        #     noise = torch.cat((first_66_features, last_130_features), dim=-1)


            # first_66_features = x_start2.to("cuda")[:, :, :, :66]

            # # 生成后 130 个特征，使用标准高斯分布
            # last_130_features = torch.randn(shape[0], shape[1], shape[2], 130).to("cuda")

            # # 将前 66 个和后 130 个特征拼接起来
            # noise = torch.cat((first_66_features, last_130_features), dim=-1)
            
        init_x = torch.randn(*shape, device=device,requires_grad=True) 
        

        func = lambda t, x: model(x, t, **model_kwargs)
        print(ode_kwargs["method"])

        if ode_kwargs["method"] in ["euler", "dopri5"]:
            assert not ("return_x_est" in ode_kwargs and ode_kwargs["return_x_est"])
            if ode_kwargs["method"] == "euler":
                ode_kwargs = dict(
                    rtol=ode_kwargs["rtol"],
                    atol=ode_kwargs["atol"],
                    method="euler",
                    options=dict(step_size=ode_kwargs["step_size"]),
                )
            elif ode_kwargs["method"] == "dopri5":
                ode_kwargs = dict(
                    rtol=ode_kwargs["rtol"],
                    atol=ode_kwargs["atol"],
                    method="dopri5",
                )
                
            #*********************************************************************************************************************************
            x1_trajectory = []

            print("We use the ode integrator")
 
            max_iter = 1
            lr = 0.01 #训练越成型，允许的学习率越低
            init_x.requires_grad_(True)
            # 定义需要优化的部分维度
            # partial_x = init_x[:, :, :, 66:196].clone().detach().requires_grad_(True)  # 前闭后开区间  # 优化后 130 个维度
            # print(partial_x.shape)
            
            optimizer_type = 'SGD'
            if optimizer_type == 'LBFGS':
                optimizer = torch.optim.LBFGS([init_x], max_iter=max_iter, lr=lr, line_search_fn='strong_wolfe')
            elif optimizer_type == 'SGD':
                optimizer = torch.optim.Adam([init_x], lr=lr)   #SO3 行列式为1 乘法就是旋转 #四元数 [cos轴 sin轴(a,b,c)] 
            else:
                raise ValueError(f"Uknown optimizer_type {optimizer_type}")

            loss = 0
            
            
            
            target_cost = None
            time_limit = None
            start_time = time.time()
            regularizer = "chi_d"
            reg_lam = 1e-4
            optim_steps = 15
            # loss_func = self.masked_l2()
            y = y.to(init_x)
            init_x.requires_grad_(True)
            loss_fn = nn.MSELoss(reduction='mean')
            metrics = {'loss': CatMetric(), 'reg': CatMetric(), 'norm_x0': CatMetric(), 'std_x0': CatMetric(), 'mean_x0': CatMetric()}

            for step in range(optim_steps):
                
                optimizer.zero_grad()
                
                reg_loss = torch.tensor(0.).to(init_x)
                
                # init_x_clone = init_x.clone()
                # init_x_clone[:, :, :, 66:196] = partial_x
                ### solve for x1
                x1 = odeint(
                    func,
                    init_x,
                    # 0.0,
                    torch.tensor([0.0, 1.0], device=device, dtype=init_x.dtype),
                        # phi=self.parameters(),
                        # method="euler", # "dopri5",
                        # rtol=1e-5,
                        # atol=1e-5,
                    adjoint_params=(init_x),
                    **ode_kwargs
                        # options=dict(step_size=1/100),
                    )
                x1 = x1[-1]
            
                mask = model_kwargs["y"]["mask"]
                mask = mask.to("cuda")
                
                ### set or compute regularization loss
                if regularizer == 'chi_d':
                    dim_x = torch.tensor(init_x[0].numel()).to(init_x)
                    init_x_norm = init_x.norm()
                    reg_loss = -((dim_x-1)*torch.log(init_x_norm) - init_x_norm.pow(2)/2)
                    
                loss =  reg_lam * reg_loss + compute_smooth_loss(x1,mask)
                #SO3的修改就是用SO3变换后的数据作为MSE的标准
                
                # y_masked = y * mask.float()
                # x1_masked = x1 * mask.float()
                # y_SO3 = process_input_tensor(y_masked)
                # x1_SO3 = process_input_tensor(x1_masked)
                # y_SO3_ = y_SO3[:,:,:,:66]
                # x1_SO3_ = x1_SO3[:,:,:,:66]
                # #196 = 66+ 130  = 198 + 130 
                # loss = self.masked_l2(y,x1,mask).mean() + 10 * loss_fn(y_SO3_,x1_SO3_) + reg_lam * reg_loss 
                
                # fid退化 ： 基于语义轨迹
                # print(loss.shape)
                print("第%d轮： mse 是 %f"%(step,torch.norm(loss)))
                
                norm_x0 = init_x.norm()
                std_x0 = init_x.std()
                mean_x0 = init_x.mean()

                metrics['norm_x0'].update(norm_x0.item())
                metrics['std_x0'].update(std_x0.item())
                metrics['mean_x0'].update(mean_x0.item())
                # metrics['cost'].update(cost.item())
                metrics['reg'].update(reg_loss.item())
                metrics['loss'].update(torch.norm(loss).item())
                
                loss.backward()
                optimizer.step()

                x1_trajectory.append(x1.detach().cpu().numpy())


                elapsed = 0
                # metrics['time'].update(elapsed)

                if step % 5 == 0:
                    elapsed = elapsed/60
                    print(f"[Step {step}] Loss {loss.item()}"
                    + f"| x_init min {init_x.detach().min()} max {init_x.detach().max()}" + f"| time: {elapsed} mins")
                
                if loss <= 48 and step>=3:
                    break

                if target_cost is not None:
                    mets_cost = metrics['cost'].compute()
                    if mets_cost.dim() > 0:
                        mets_cost = mets_cost[-1]
                    last_cost = mets_cost.item()
                    if last_cost <= target_cost:
                        print(f'reached cost of {last_cost}')
                        break

                if time_limit is not None:
                    elapsed = (time.time() - start_time)/60 # time in minutes
                    if elapsed > time_limit:
                        print(f'reached time limit of {time_limit} mins')
                        break
                    
            step_time = time.time() - start_time
            print("本次训练过去了%d s"%step_time)  
            x1_nograd = x1.detach()
            data = x1_nograd
            print(data.shape)
        elif ode_kwargs["method"] == "euler_replacement_edit_till":
            data = self.sample_euler_replacement_edit_till(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                edit_till=ode_kwargs["edit_till"],
                model_kwargs=model_kwargs,
                ode_kwargs=ode_kwargs,
            )

        elif ode_kwargs["method"] == "odenoise_euler_replacement":
            inpainting_mask, inpainted_motion = (
                model_kwargs["y"]["inpainting_mask"],
                model_kwargs["y"]["inpainted_motion"],
            )
            partial_data = (0 * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            _ode_kwargs = dict(
                rtol=ode_kwargs["rtol"],
                atol=ode_kwargs["atol"],
                method="dopri5",
            )
            
            #***************************************************************************************************************************
            noise = odeint(
                func,
                partial_data,
                # 0.0,
                torch.tensor([1.0, 0.0], device=device, dtype=noise.dtype),
                # phi=self.parameters(),
                # method="euler", # "dopri5",
                # rtol=1e-5,
                # atol=1e-5,
                adjoint_params=(),
                **_ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]

            data = self.sample_euler_replacement_edit_till(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                edit_till=ode_kwargs["edit_till"],
                model_kwargs=model_kwargs,
                ode_kwargs=ode_kwargs,
            )
        elif ode_kwargs["method"] == "variation_euler_replacement":
            inpainting_mask, inpainted_motion = (
                model_kwargs["y"]["inpainting_mask"],
                model_kwargs["y"]["inpainted_motion"],
            )
            partial_data = inpainted_motion  # (0 * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            _ode_kwargs = dict(
                rtol=ode_kwargs["rtol"],
                atol=ode_kwargs["atol"],
                method="dopri5",
            )
            
            
            #**********************************************************************************************************************************
            tg_gaussian = TruncatedNormal(torch.Tensor([0]), torch.Tensor([1]), -2, 2)
            noise = odeint(
                func,
                partial_data,
                # 0.0,
                torch.tensor([1.0, 0.0], device=device, dtype=noise.dtype),
                # phi=self.parameters(),
                # method="euler", # "dopri5",
                # rtol=1e-5,
                # atol=1e-5,
                adjoint_params=(),
                **_ode_kwargs
                # options=dict(step_size=1/100),
            )[-1]
            
            
            first_66_features = tg_gaussian.rsample(shape).squeeze(-1).to("cuda")[:, :, :, :66]

            # 生成后 130 个特征，使用标准高斯分布
            last_130_features = torch.randn(shape[0], shape[1], shape[2], 130).to("cuda")

            # 将前 66 个和后 130 个特征拼接起来
            noise_origin = torch.cat((first_66_features, last_130_features), dim=-1)


            _noise_masked = (noise_origin * ~inpainting_mask) + (
                noise * inpainting_mask
            )

            data = self.sample_euler_replacement(
                model,
                z_orig=_noise_masked,
                N=int(1 / ode_kwargs["step_size"]),
                model_kwargs=model_kwargs,
            )
        elif ode_kwargs["method"] == "euler_raw":
            data = self.sample_euler_raw(
                model,
                z_orig=noise,
                N=int(1 / ode_kwargs["step_size"]),
                ode_kwargs=ode_kwargs,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError

        is_return_est = isinstance(data, tuple)
        if is_return_est:
            data, x0_est = data
            assert model.training is not True, "x0_est is only for inference"

        data_range_dict = dict()
        # 0th-joint as an obversation
        data_range_dict["data_range_j0/gen_mean"] = data[:, 0].mean()
        data_range_dict["data_range_j0/gen_std"] = data[:, 0].std()
        data_range_dict["data_range_j0/gen_min"] = data[:, 0].min()
        data_range_dict["data_range_j0/gen_max"] = data[:, 0].max()
        if is_rank_zero() and model.training:
            wandb.log(data_range_dict)
        if is_return_est:
            return data, x0_est
        else:
            return data

    def training_losses(
        self,
        model,
        x_start,
        # x_start2,   # x_start2作为噪声的起点
        t,
        model_kwargs=None,
        noise=None,
        dataset=None,
        dataset2=None,
        sigma_min=1e-4,
    ):
    #出现dataset2
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        
        #***********************************************************************************************************************
        tg_gaussian = TruncatedNormal(torch.Tensor([0]), torch.Tensor([1]), -2, 2)
        #实验的时候这个部分没有改
        if noise is None:

            # 有界数据部分使用有界高斯
            shape = x_start.shape
            noise = tg_gaussian.rsample(x_start.shape).squeeze(-1).to("cuda")


            first_66_features = noise.to("cuda")[:, :, :, :66]               

            # 生成后 130 个特征，使用标准高斯分布
            last_130_features = torch.randn(shape[0], shape[1], shape[2], 130).to("cuda")

            # 将前 66 个和后 130 个特征拼接起来
            noise = torch.cat((first_66_features, last_130_features), dim=-1)
            
            
        

        mask = model_kwargs["y"]["mask"]
        get_xyz = lambda sample: model.module.rot2xyz(
            sample,
            mask=None,
            pose_rep=model.module.pose_rep,
            translation=model.module.translation,
            glob=model.module.glob,
            # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
            jointstype="smpl",  # 3.4 iter/sec
            vertstrans=False,
        )

        if model_kwargs is None:
            model_kwargs = {}

        
            
        
            

        assert t is None
        t = torch.rand(len(x_start), device=x_start.device, dtype=x_start.dtype)
        t_1d = t[:,]  # [B, 1, 1, 1]
        t = t[:, None, None, None]  # [B, 1, 1, 1]
        x_t = t * x_start + (1 - (1 - sigma_min) * t) * noise
        target = x_start - (1 - sigma_min) * noise

        terms = {}
        model_output = model(x_t, t_1d, **model_kwargs)

        terms["rot_mse"] = self.masked_l2(
            target, model_output, mask
        )  # mean_flat(rot_mse)
        print("\n rot_mse is %f"%(torch.norm(terms["rot_mse"])))

        target_xyz, model_output_xyz = None, None

        if self.lambda_rcxyz > 0.0:
            target_xyz = get_xyz(
                target
            )  # [bs, nvertices(vertices)/njoints(smpl), 3, nframes]
            model_output_xyz = get_xyz(model_output)  # [bs, nvertices, 3, nframes]
            terms["rcxyz_mse"] = self.masked_l2(
                target_xyz, model_output_xyz, mask
            )  # mean_flat((target_xyz - model_output_xyz) ** 2)
            print("rcxyz_mse is %f",torch.norm(terms["rcxyz_mse"]))
        if self.lambda_vel_rcxyz > 0.0:
            if self.data_rep == "rot6d" and dataset.dataname in [
                "humanact12",
                "uestc",
            ]:
                target_xyz = get_xyz(target) if target_xyz is None else target_xyz
                model_output_xyz = (
                    get_xyz(model_output)
                    if model_output_xyz is None
                    else model_output_xyz
                )
                target_xyz_vel = target_xyz[:, :, :, 1:] - target_xyz[:, :, :, :-1]
                model_output_xyz_vel = (
                    model_output_xyz[:, :, :, 1:] - model_output_xyz[:, :, :, :-1]
                )
                terms["vel_xyz_mse"] = self.masked_l2(
                    target_xyz_vel, model_output_xyz_vel, mask[:, :, :, 1:]
                )  # not used in the loss
                print("vel_xyz_mse is %f",torch.norm(terms["vel_xyz_mse"]))

        if self.lambda_fc > 0.0:
            torch.autograd.set_detect_anomaly(True)
            if self.data_rep == "rot6d" and dataset.dataname in [
                "humanact12",
                "uestc",
            ]:
                target_xyz = get_xyz(target) if target_xyz is None else target_xyz
                model_output_xyz = (
                    get_xyz(model_output)
                    if model_output_xyz is None
                    else model_output_xyz
                )
                # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11
                l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
                relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
                gt_joint_xyz = target_xyz[
                    :, relevant_joints, :, :
                ]  # [BatchSize, 4, 3, Frames]
                gt_joint_vel = torch.linalg.norm(
                    gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2
                )  # [BatchSize, 4, Frames]
                fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=2).repeat(
                    1, 1, 3, 1
                )
                pred_joint_xyz = model_output_xyz[
                    :, relevant_joints, :, :
                ]  # [BatchSize, 4, 3, Frames]
                pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
                pred_vel[~fc_mask] = 0
                terms["fc"] = self.masked_l2(
                    pred_vel,
                    torch.zeros(pred_vel.shape, device=pred_vel.device),
                    mask[:, :, :, 1:],
                )
        if self.lambda_vel > 0.0:
            target_vel = target[..., 1:] - target[..., :-1]
            model_output_vel = model_output[..., 1:] - model_output[..., :-1]
            terms["vel_mse"] = self.masked_l2(
                target_vel[:, :-1, :, :],  # Remove last joint, is the root location!
                model_output_vel[:, :-1, :, :],
                mask[:, :, :, 1:],
            )  # mean_flat((target_vel - model_output_vel) ** 2)

        terms["loss"] = (
            terms["rot_mse"]
            + (self.lambda_vel * terms.get("vel_mse", 0.0))
            + (self.lambda_rcxyz * terms.get("rcxyz_mse", 0.0))
            + (self.lambda_fc * terms.get("fc", 0.0))
        )
        return terms

    def fc_loss_rot_repr(self, gt_xyz, pred_xyz, mask):
        def to_np_cpu(x):
            return x.detach().cpu().numpy()

        """
        pose_xyz: SMPL batch tensor of shape: [BatchSize, 24, 3, Frames]
        """
        # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11

        l_ankle_idx, r_ankle_idx = 7, 8
        l_foot_idx, r_foot_idx = 10, 11
        """ Contact calculated by 'Kfir Method' Commented code)"""
        # contact_signal = torch.zeros((pose_xyz.shape[0], pose_xyz.shape[3], 2), device=pose_xyz.device) # [BatchSize, Frames, 2]
        # left_xyz = 0.5 * (pose_xyz[:, l_ankle_idx, :, :] + pose_xyz[:, l_foot_idx, :, :]) # [BatchSize, 3, Frames]
        # right_xyz = 0.5 * (pose_xyz[:, r_ankle_idx, :, :] + pose_xyz[:, r_foot_idx, :, :])
        # left_z, right_z = left_xyz[:, 2, :], right_xyz[:, 2, :] # [BatchSize, Frames]
        # left_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)  # [BatchSize, Frames]
        # right_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)
        #
        # left_z_mask = left_z <= torch.mean(torch.sort(left_z)[0][:, :left_z.shape[1] // 5], axis=-1)
        # left_z_mask = torch.stack([left_z_mask, left_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # left_z_mask[:, :, 1] = False  # Blank right side
        # contact_signal[left_z_mask] = 0.4
        #
        # right_z_mask = right_z <= torch.mean(torch.sort(right_z)[0][:, :right_z.shape[1] // 5], axis=-1)
        # right_z_mask = torch.stack([right_z_mask, right_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # right_z_mask[:, :, 0] = False  # Blank left side
        # contact_signal[right_z_mask] = 0.4
        # contact_signal[left_z <= (torch.mean(torch.sort(left_z)[:left_z.shape[0] // 5]) + 20), 0] = 1
        # contact_signal[right_z <= (torch.mean(torch.sort(right_z)[:right_z.shape[0] // 5]) + 20), 1] = 1

        # plt.plot(to_np_cpu(left_z[0]), label='left_z')
        # plt.plot(to_np_cpu(left_velocity[0]), label='left_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 0]), label='left_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.plot(to_np_cpu(right_z[0]), label='right_z')
        # plt.plot(to_np_cpu(right_velocity[0]), label='right_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 1]), label='right_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()

        gt_joint_xyz = gt_xyz[
            :, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :
        ]  # [BatchSize, 4, 3, Frames]
        gt_joint_vel = torch.linalg.norm(
            gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2
        )  # [BatchSize, 4, Frames]
        fc_mask = gt_joint_vel <= 0.01
        pred_joint_xyz = pred_xyz[
            :, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :
        ]  # [BatchSize, 4, 3, Frames]
        pred_joint_vel = torch.linalg.norm(
            pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1], axis=2
        )  # [BatchSize, 4, Frames]
        pred_joint_vel[
            ~fc_mask
        ] = 0  # Blank non-contact velocities frames. [BS,4,FRAMES]
        pred_joint_vel = torch.unsqueeze(pred_joint_vel, dim=2)

        """DEBUG CODE"""
        # print(f'mask: {mask.shape}')
        # print(f'pred_joint_vel: {pred_joint_vel.shape}')
        # plt.title(f'Joint: {joint_idx}')
        # plt.plot(to_np_cpu(gt_joint_vel[0]), label='velocity')
        # plt.plot(to_np_cpu(fc_mask[0]), label='fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        return self.masked_l2(
            pred_joint_vel,
            torch.zeros(pred_joint_vel.shape, device=pred_joint_vel.device),
            mask[:, :, :, 1:],
        )

    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def foot_contact_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259
        contact_mask_gt = (
            contact > 0.5
        )  # contact mask order for indexes are fid_l [7, 10], fid_r [8, 11]
        vel_lf_7 = local_velocity[:, 7 * 3 : 8 * 3, :, :]
        vel_rf_8 = local_velocity[:, 8 * 3 : 9 * 3, :, :]
        vel_lf_10 = local_velocity[:, 10 * 3 : 11 * 3, :, :]
        vel_rf_11 = local_velocity[:, 11 * 3 : 12 * 3, :, :]

        calc_vel_lf_7 = (
            ric_data[:, 6 * 3 : 7 * 3, :, 1:] - ric_data[:, 6 * 3 : 7 * 3, :, :-1]
        )
        calc_vel_rf_8 = (
            ric_data[:, 7 * 3 : 8 * 3, :, 1:] - ric_data[:, 7 * 3 : 8 * 3, :, :-1]
        )
        calc_vel_lf_10 = (
            ric_data[:, 9 * 3 : 10 * 3, :, 1:] - ric_data[:, 9 * 3 : 10 * 3, :, :-1]
        )
        calc_vel_rf_11 = (
            ric_data[:, 10 * 3 : 11 * 3, :, 1:] - ric_data[:, 10 * 3 : 11 * 3, :, :-1]
        )

        # vel_foots = torch.stack([vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11], dim=1)
        for chosen_vel_foot_calc, chosen_vel_foot, joint_idx, contact_mask_idx in zip(
            [calc_vel_lf_7, calc_vel_rf_8, calc_vel_lf_10, calc_vel_rf_11],
            [vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11],
            [7, 10, 8, 11],
            [0, 1, 2, 3],
        ):
            tmp_mask_gt = (
                contact_mask_gt[:, contact_mask_idx, :, :]
                .cpu()
                .detach()
                .numpy()
                .reshape(-1)
                .astype(int)
            )
            chosen_vel_norm = np.linalg.norm(
                chosen_vel_foot.cpu().detach().numpy().reshape((3, -1)), axis=0
            )
            chosen_vel_calc_norm = np.linalg.norm(
                chosen_vel_foot_calc.cpu().detach().numpy().reshape((3, -1)), axis=0
            )

            print(tmp_mask_gt.shape)
            print(chosen_vel_foot.shape)
            print(chosen_vel_calc_norm.shape)
            import matplotlib.pyplot as plt

            plt.plot(tmp_mask_gt, label="FC mask")
            plt.plot(chosen_vel_norm, label="Vel. XYZ norm (from vector)")
            plt.plot(chosen_vel_calc_norm, label="Vel. XYZ norm (calculated diff XYZ)")

            plt.title(f"FC idx {contact_mask_idx}, Joint Index {joint_idx}")
            plt.legend()
            plt.show()
        # print(vel_foots.shape)
        return 0

    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def velocity_consistency_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259

        calc_vel_from_xyz = ric_data[:, :, :, 1:] - ric_data[:, :, :, :-1]
        velocity_from_vector = local_velocity[:, 3:, :, 1:]  # Slicing out root
        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(
            target.permute(0, 2, 3, 1).type(th.FloatTensor)
        )
        print(f"r_rot_quat: {r_rot_quat.shape}")
        print(f"calc_vel_from_xyz: {calc_vel_from_xyz.shape}")
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 2, 3, 1)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21, 3)).type(
            th.FloatTensor
        )
        r_rot_quat_adapted = (
            r_rot_quat[..., :-1, None, :]
            .repeat((1, 1, 1, 21, 1))
            .to(calc_vel_from_xyz.device)
        )
        print(
            f"calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}"
        )
        print(
            f"r_rot_quat_adapted: {r_rot_quat_adapted.shape}, {r_rot_quat_adapted.device}"
        )

        calc_vel_from_xyz = motion_process.qrot(r_rot_quat_adapted, calc_vel_from_xyz)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21 * 3))
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 3, 1, 2)
        print(
            f"calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}"
        )

        import matplotlib.pyplot as plt

        for i in range(21):
            plt.plot(
                np.linalg.norm(
                    calc_vel_from_xyz[:, i * 3 : (i + 1) * 3, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape((3, -1)),
                    axis=0,
                ),
                label="Calc Vel",
            )
            plt.plot(
                np.linalg.norm(
                    velocity_from_vector[:, i * 3 : (i + 1) * 3, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape((3, -1)),
                    axis=0,
                ),
                label="Vector Vel",
            )
            plt.title(f"Joint idx: {i}")
            plt.legend()
            plt.show()
        print(calc_vel_from_xyz.shape)
        print(velocity_from_vector.shape)
        diff = calc_vel_from_xyz - velocity_from_vector
        print(np.linalg.norm(diff.cpu().detach().numpy().reshape((63, -1)), axis=0))

        return 0
