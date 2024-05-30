# Copyright (c) 2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""IndustReal: class for peg insertion task.

Inherits IndustReal pegs environment class and Factory abstract task class (not enforced).

Trains a peg insertion policy with Simulation-Aware Policy Update (SAPU), SDF-Based Reward, and Sampling-Based Curriculum (SBC).

Can be executed with python train.py task=IndustRealTaskPegsInsert test=True headless=True task.env.numEnvs=256
"""


import hydra
import numpy as np
import omegaconf
import os
import torch
import warp as wp

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
import sys
sys.path.append('../../using_planner/')
repulsor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../using_planner'))
sys.path.append(repulsor_dir)
from repulsor import *
sys.path.append('..')
sys.path.append('../..')
sys.path.append('/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/')
sys.path.append('../../isaacgym/python/isaacgym')

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
from isaacgymenvs.tasks.industreal.industreal_env_pegs import IndustRealEnvPegs
from isaacgymenvs.utils import torch_jit_utils
import torch.nn as nn
import random

import isaacgymenvs.tasks.factory.factory_control as fc

"""
ReadMe

In the init, select the policy to load in. If you select self.potential_field=True then the potential field will be used as the policy in place of the actor loaded in.
In the init if potential_field==True then create pegs and sockets [0,0,0,0,0,0,1] to be used as template for parallel changes later

Within prephysics step actions are chosen:
Actions are passed into function of RL games network. This can be used for residual RL.
In prephysics step the peg and socket poses and quats have noise added to them. 
Then either potential field action or policy action is used and passed to the DG controller.
For the potential field, in parallel put the points in the proper locations. Then choose the actions with the act() function and pass to DG controller.
"""

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action
        self._initialize_weights

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use 'relu' for nonlinearity as it's a common choice for ELU as well
                init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        #action = self.max_action * self.net(state)
        return self.net(state)
        peg_pos = state[:,0:3]
        peg_quat = state[:,3:7]
        hole_pos = state[:,7:10]
        hole_quat = state[:,10:14]

        state = torch.cat((peg_pos, peg_quat, hole_pos, hole_quat), dim=1)
        
        # Check outputs of each layer
        x = state#.clone()

        for i, layer in enumerate(self.net):
            x = layer(x)
            #print(f"Output after layer {i} ({layer.__class__.__name__}):", x)
            if torch.isnan(x).any():
                print(f"NaN detected after layer {i} ({layer.__class__.__name__})")
                break  # Optional: break the loop if NaN is detected

        return x
        #action = self.net(state)
        #action[0:3] *= self.max_action
        #return action

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


#"""
from torch.distributions import Normal
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        return mean
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)
    
    def return_mean(self, obs: torch.Tensor):
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()
        
class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 1024, n_hidden: int = 5):
        super().__init__()
        """
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)
        """
        #self.n_hidden = 5
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        layers = []
        #Add input layer
        layers.append(nn.Linear(state_dim, self.hidden_dim))
        #layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.ELU())
        
        #Add hidden layers
        for i in range(1, self.n_hidden):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            #layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ELU())
        
        # Add output layer
        layers.append(nn.Linear(self.hidden_dim, 1))
        #layers.append(nn.ReLU())
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)


    def forward(self, obs: torch.Tensor) -> torch.Tensor: #state is peg and hole position
        x = obs.clone()

        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        for i in range(1, self.n_hidden):
          x_res = x.clone()
          x = self.net[(i*3)+0](x)
          x = self.net[(i*3)+1](x)
          x = self.net[(i*3)+2](x)
          #print("this layer", i, "num activated", (x <= -0.99).sum())
          x += x_res
          #if i == self.n_hidden-1:
            #print(x)
        
        x = self.net[self.n_hidden*3](x)
        
        return x
        
class DeterministicFourierPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 1024,
        n_hidden: int = 5,
        mapping_size: int = 256,
        scale_down: float = 6e4,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.mapping_size = mapping_size
        layers = []
        #Add input layer
        layers.append(nn.Linear(self.mapping_size*2, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.ELU())
        
        #Add hidden layers
        for i in range(1, self.n_hidden):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ELU())
        
        # Add output layer
        layers.append(nn.Linear(self.hidden_dim, act_dim))
        layers.append(nn.Tanh())
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)
        self.max_action = max_action
        
        #self.B = np.random.normal(rand_key, (self.mapping_size, 2))
        self.B = torch.randn((self.mapping_size, state_dim))
        scale=10.0 #1.0, 100.0 other options there
        self.B = self.B * scale
        self.scale_down = scale_down

    #def input_mapping(x):
    #    x_proj = (2.*np.pi*x) @ self.B.T #This is v=[batch, state14] @ BT=[state14, mapping] => vBT=[batch,mapping]
    #    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)
    
    def input_mapping(self, x):
        x_proj = (2.*np.pi*x) @ self.B.T #This is v=[batch, state14] @ BT=[state14, mapping] => vBT=[batch,mapping]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1) #[batch, mapping*2]

    def forward(self, obs: torch.Tensor) -> torch.Tensor: #state is peg and hole position
        x = obs.clone()
        x = self.input_mapping(x)
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        for i in range(1, self.n_hidden):
          x_res = x.clone()
          x = self.net[(i*3)+0](x)
          x = self.net[(i*3)+1](x)
          x = self.net[(i*3)+2](x)
          x += x_res
        
        x = self.net[self.n_hidden*3](x)
        x = self.net[self.n_hidden*3 + 1](x) #here we are doing just tanh
        x = x / self.scale_down
        
        return x

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )
        
class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 1024,
        n_hidden: int = 5,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        #self.net = MLP(
        #    [state_dim, *([hidden_dim] * n_hidden), act_dim],
        #    output_activation_fn=nn.Tanh,
        #    dropout=dropout,
        #)
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        layers = []
        #Add input layer
        layers.append(nn.Linear(state_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.ELU())
        
        #Add hidden layers
        for i in range(1, self.n_hidden):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ELU())
        
        # Add output layer
        layers.append(nn.Linear(self.hidden_dim, act_dim))
        layers.append(nn.Tanh())
        
        # Create the sequential model
        self.net = nn.Sequential(*layers)
                 
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor: #state is peg and hole position
        x = obs.clone()

        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        for i in range(1, self.n_hidden):
          x_res = x.clone()
          x = self.net[(i*3)+0](x)
          x = self.net[(i*3)+1](x)
          x = self.net[(i*3)+2](x)
          x += x_res
        
        x = self.net[self.n_hidden*3](x)
        x = self.net[self.n_hidden*3 + 1](x) #here we are doing just tanh
        
        return x
        #return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )
        
class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
#"""

class IndustRealTaskPegsInsert(IndustRealEnvPegs, FactoryABCTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize task superclass."""

        self.cfg = cfg
        self._get_task_yaml_params()

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self._acquire_task_tensors()
        self.parse_controller_spec()

        # Get Warp mesh objects for SAPU and SDF-based reward
        wp.init()
        self.wp_device = wp.get_preferred_device()
        (
            self.wp_plug_meshes,
            self.wp_plug_meshes_sampled_points,
            self.wp_socket_meshes,
        ) = algo_utils.load_asset_meshes_in_warp(
            plug_files=self.plug_files,
            socket_files=self.socket_files,
            num_samples=self.cfg_task.rl.sdf_reward_num_samples,
            device=self.wp_device,
        )

        if self.viewer != None:
            self._set_viewer_params()
        

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------POTENTIAL FIELD------------#
        self.potential_field = True
        resolution=10
        if self.potential_field == True:
          self.socket_points_template = create_socket(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to(self.device), resolution=resolution).unsqueeze(0)
          for jj in range(1, self.num_envs):
            self.socket_points_template = torch.cat([self.socket_points_template, create_socket(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to(self.device), resolution=resolution).unsqueeze(0)], dim=0)
          self.socket_points_template.to(self.device)
          print("socket points shape", self.socket_points_template.shape)
          #self.trajectory=define_trajectory(resolution=100)
          #always create new pegs
          self.plug_points_template = create_peg(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to(self.device), resolution=resolution).unsqueeze(0)
          for jj in range(1, self.num_envs):
            self.plug_points_template = torch.cat([self.plug_points_template, create_peg(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to(self.device), resolution=resolution).unsqueeze(0)], dim=0)
          self.plug_points_template.to(self.device)
          print("peg points shape", self.plug_points_template.shape)
          
        self.entire_action_residual = self.cfg_task.env.entire_action_residual#False
        self.potential_field_scale_residual = self.cfg_task.env.potential_field_scale_residual
        self.self_scaling_residual = self.cfg_task.env.self_scaling_residual
        self.simple_self_scaling_residual = self.cfg_task.env.simple_self_scaling_residual
        self.residual_scale = self.cfg_task.env.residual_scale
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------POTENTIAL FIELD------------#
        
        #Noise
        self.socket_pos_obs_noise = self.cfg_task.env.socket_pos_obs_noise
        self.socket_rot_obs_noise = self.cfg_task.env.socket_rot_obs_noise
        self.plug_pos_obs_noise = self.cfg_task.env.plug_pos_obs_noise
        self.plug_rot_obs_noise = self.cfg_task.env.plug_rot_obs_noise
        if self.cfg_task.env.correlated_noise == True:
          self.correlated_peg_obs_pos_noise = torch.zeros([self.num_envs, 3], device=self.device)
       
        self.pose_tensor = torch.zeros((1, 7), dtype=torch.float32).to(device)
        self.iterations_correct=0
        self.last_z=torch.zeros([self.num_envs])
        self.run_actions = False #ISIDOROS
        self.iterations_since_last_reset = 0 #ISIDOROS
        normal=False
        iql=False#True
        iql_new = True
        fourier_policy = False
        self.pos_only_policy = False
        gaussian_policy = False
        if normal == True:
          #Added---------------------------------------------------------------------------------------------------------------------------#
          self.policy = Actor(14, 6, 1.0)
          self.policy = self.policy.to('cuda')
          policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/properly_trained_results/BC_300000lr0.0003_size512_checkpoint.pt"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/max_trained_TD3_BC_Working_999999"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/100000lr0.0003_size512_checkpoint_BC.pt"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/checkpoint_589999.pt"
          print(f"Loading model from {policy_file}")
          #print(torch.load(policy_file))
          actor_weights=torch.load(policy_file)#['actor']
          self.policy.load_state_dict(actor_weights)
          #Added---------------------------------------------------------------------------------------------------------------------------#
        if iql_new == True:
          self.policy = (DeterministicPolicy(14, 6, 1.0, n_hidden=5, dropout=None))
          self.policy = self.policy.to('cuda')
          policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/LowerActionQFixed_1000times_IQL_5_layer_laynorm_larger_data_sparse/chp_IQL_actor_action_trained_only_749999.pth"#"/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/IQL_5_layer_vfqfpolicy_laynorm_dense/chp_IQL_actor_399999.pth"
          print(f"Loading model from {policy_file}")
          actor_weights=torch.load(policy_file)#['actor']
          self.policy.load_state_dict(actor_weights)
          self.policy.eval()
          
          self.vf = ValueFunction(14, hidden_dim=1024, n_hidden=5)
          self.vf = self.vf.to('cuda')
          vf_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/Q_fixed_IQL_5_layer_laynorm_no_noise_sparse/chp_IQL_vf_999999.pth"
          vf_weights=torch.load(vf_file)#['actor']
          self.vf.load_state_dict(vf_weights)
          #task.env.numEnvs=1
        if fourier_policy == True:
          self.policy = DeterministicFourierPolicy(14, 3, 1.0, hidden_dim=1024, n_hidden=5, scale_down=7258.98046875)
          self.policy = self.policy.to('cuda')
          policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/FourierBC_BC_pos_3_only_5_layer/chp_FBC_actor_scale_7258.98046875_step_499999.pth"
          print(f"Loading model from {policy_file}")
          actor_weights=torch.load(policy_file)
          self.policy.load_state_dict(actor_weights)
          B_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/FourierBC_BC_pos_3_only_5_layer/chp_FBC_7258.98046875_B.pth"
          B_matrix=torch.load(B_file).to('cuda')
          self.policy.B = B_matrix
        if self.pos_only_policy == True:
          self.policy = (DeterministicPolicy(14, 3, 1.0, n_hidden=5, dropout=None))
          self.policy = self.policy.to('cuda')
          policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/DotProduct_BC_BIGDATA_5_layer/chp_BC_actor_999999.pth"
          print(f"Loading model from {policy_file}")
          actor_weights=torch.load(policy_file)#['actor']
          self.policy.load_state_dict(actor_weights)
        if gaussian_policy == True:
          #Added---------------------------------------------------------------------------------------------------------------------------#
          self.policy = GaussianPolicy(14, 6, 1.0)
          self.policy = self.policy.to('cuda')
          policy_file = "/common/home/jhd79/robotics/IsaacGymEnvs/isaacgymenvs/checkpoints/checkpoint_IQL_sparse_999999.pt"
          print(f"Loading model from {policy_file}")
          #print(torch.load(policy_file))
          actor_weights=torch.load(policy_file)['actor']
          self.policy.load_state_dict(actor_weights)
          #Added---------------------------------------------------------------------------------------------------------------------------#

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        ppo_path = os.path.join(
            "train/IndustRealTaskPegsInsertPPO.yaml"
        )  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_pos_local = torch.tensor(
            [
                [
                    0.0,
                    0.0,
                    (self.cfg_task.env.socket_base_height + self.plug_grasp_offsets[i]),
                ]
                for i in range(self.num_envs)
            ],
            device=self.device,
        )
        self.gripper_goal_quat_local = self.identity_quat.clone()

        self.socket_top_pos_local = torch.tensor(
            [[0.0, 0.0, self.socket_heights[i]] for i in range(self.num_envs)],
            device=self.device,
        )
        self.socket_quat_local = self.identity_quat.clone()

        # Define keypoint tensors
        self.keypoint_offsets = (
            algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device)
            * self.cfg_task.rl.keypoint_scale
        )
        self.keypoints_plug = torch.zeros(
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.keypoints_socket = torch.zeros_like(
            self.keypoints_plug, device=self.device
        )

        self.actions = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions_RL), device=self.device
        )

        self.curr_max_disp = self.cfg_task.rl.initial_max_disp

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pose of gripper goal and top of socket in global frame
        self.gripper_goal_quat, self.gripper_goal_pos = torch_jit_utils.tf_combine(
            self.socket_quat,
            self.socket_pos,
            self.gripper_goal_quat_local,
            self.gripper_goal_pos_local,
        )
    
        self.socket_top_quat, self.socket_top_pos = torch_jit_utils.tf_combine(
            self.socket_quat,
            self.socket_pos,
            self.socket_quat_local,
            self.socket_top_pos_local,
        )
        #print("socket top pos", self.socket_top_pos)

        # Add observation noise to socket pos
        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )
        socket_obs_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        # Add observation noise to socket rot
        socket_rot_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        socket_obs_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.env.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        self.noisy_socket_quat = torch_utils.quat_from_euler_xyz(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )

        # Compute observation noise on socket
        (
            self.noisy_gripper_goal_quat,
            self.noisy_gripper_goal_pos,
        ) = torch_jit_utils.tf_combine(
            self.noisy_socket_quat,
            self.noisy_socket_pos,
            self.gripper_goal_quat_local,
            self.gripper_goal_pos_local,
        )

        # Compute pos of keypoints on plug and socket in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(
                self.plug_quat,
                self.plug_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

            self.keypoints_socket[:, idx] = torch_jit_utils.tf_combine(
                self.socket_quat,
                self.socket_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

    def prepare_state(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        
        peg_pos = state[:,0:3]
        peg_quat = state[:,3:7]
        hole_pos = state[:,7:10]
        hole_quat = state[:,10:14]

        peg_pos_ = self.pose_world_to_robot_base(peg_pos, peg_quat)[0],  # 3
        peg_quat_ = self.pose_world_to_robot_base(peg_pos, peg_quat)[1],  # 4
        hole_pos_ = self.pose_world_to_robot_base(hole_pos, hole_quat)[0],  # 3
        hole_quat_ = self.pose_world_to_robot_base(hole_pos, hole_quat)[1],  # 4

        peg_pos = peg_pos_[0].clone()
        hole_pos = hole_pos_[0].clone()
        peg_quat = peg_quat_[0].clone()
        hole_quat = hole_quat_[0].clone()

        state = torch.cat((peg_pos, peg_quat, hole_pos, hole_quat), dim=1)
        
        # Normalize States
        #state = normalize_states(state, self.state_mean, self.state_std)

        state_input = state
        
        return torch.tensor(state_input, dtype=torch.float32)

    def create_rotational_randomization_axis_angle(self, plug_rot_quat=None, target_position=None):
        if plug_rot_quat == None:
          plug_rot_noise = 2 * (
              torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
              - 0.5
          )
          plug_rot_noise = plug_rot_noise @ torch.diag(
              torch.tensor(
                  self.cfg_task.randomize.plug_rot_noise,
                  dtype=torch.float32,
                  device=self.device,
              )
          )
          plug_rot_euler = (
              torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
              + plug_rot_noise
          )
          plug_rot_quat = torch_utils.quat_from_euler_xyz(
              plug_rot_euler[:, 0], plug_rot_euler[:, 1], plug_rot_euler[:, 2]
          )
          #print("plug rot quat was zero", plug_rot_quat)
        quat_total_rotation=torch_utils.quat_mul(plug_rot_quat, self.identity_quat.clone())
        action_towards = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        if target_position != None:
          action_towards[:, 0:3] = target_position - self.plug_pos
        action_towards[:, 3:6] = self.quaternion_towards(quat_total_rotation, self.plug_quat)
        #print("action towards in creating rotational randomization axis angle", action_towards)
        return action_towards, plug_rot_quat

    def quaternion_towards(self, a: torch.Tensor, b: torch.Tensor): #b to a, quaternion relative rotation to move
        #b is q1 and a is q2. This is b to a rotation
        b_conj = torch_jit_utils.quat_conjugate(b)
        rotation_towards = torch_jit_utils.quat_mul(b_conj, a)
        #rotation_towards = torch_jit_utils.quat_mul(a, b_conj)
        rotation_towards = rotation_towards / torch.norm(rotation_towards, dim=-1, keepdim=True)
        angle, axis = torch_jit_utils.quat_to_angle_axis(rotation_towards)
        rot_actions = angle.unsqueeze(1) * axis
        #print("from quaternion towards, rot_actions", rot_actions)
        return rot_actions

    def residual_servoing(self, actions):
        new_actions = torch.zeros_like(actions)
        movement = -1 * (self.noisy_plug_pos[:, 0] - self.noisy_socket_pos[:, 0])
        #print("x diff", movement*100)
        movement = torch.clip(movement * 100.0, -1.0, 1.0)
        new_actions[:, 0] += movement
        
        movement = -1 * (self.noisy_plug_pos[:, 1] - self.noisy_socket_pos[:, 1])
        #print("y diff", movement*100)
        movement = torch.clip(movement * 100.0, -1.0, 1.0)
        new_actions[:, 1] += movement
        
        movement = -1 * (self.noisy_plug_pos[:, 2] - self.noisy_socket_pos[:, 2])
        #print("z diff", movement*100)
        movement = torch.clip(movement * 100.0, -1.0, 1.0)
        new_actions[:, 2] += movement
        
        new_actions[:, 3:6] = torch.clip(self.quaternion_towards(self.noisy_socket_quat, self.noisy_plug_quat)*10.0, -1.0, 1.0)
 
        #print("TO SHOW THE NOISE OBSERVATIONS: self.noisy_socket_pos", self.noisy_socket_pos[0], "self.socket_pos", self.socket_pos[0])
 
        use_z = torch.where( #true first, in the condition the tensor used where it this is true the value becaomes the first part
          torch.abs(new_actions[:, 1]) + torch.abs(new_actions[:, 0]) < 0.18, #0.08
          torch.full((self.num_envs,), 1.0, device=self.device),
          torch.full((self.num_envs,), 0.0, device=self.device))
        new_actions[:, 2] *= use_z
        self.iterations_correct+=1
        #This is to make sure we don't accidently just go downwards
        if self.iterations_correct < 100:#(self.max_episode_length / 2):
          new_actions[:, 2] += (torch.ones_like(actions) * 0.2)[:, 0]
          
        self.actions=new_actions.clone().to(self.device)
        #Then with rotational randomness, then with even more noise

        #print("actions:", self.actions[0])
        self._apply_actions_as_ctrl_targets_noisy(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

    def residual_servoing_attractor(self, actions_residual):
        #input is peg: [envs, points, 3], socket: [envs, points, 3], peg_pose: [envs, 7], trajectory: [points, 3]
        #print("noisy vs no noisy", self.noisy_plug_pos[0], self.plug_pos[0])
        if self.potential_field_scale_residual == True:
          #2 dim actions of pos scale and rot scale
          actions_residual += 1.0
          actions_residual /= 2.0 #between 0 and 1 now as the original output is tanh
          #print(actions_residual, "higher means more repulsion")
          start_index=0
          if self.entire_action_residual == True:
            start_index=6
          if self.self_scaling_residual == True:
            start_index=8
          new_actions = act(self.plug_points, self.socket_points, torch.cat([self.noisy_plug_pos, self.noisy_plug_quat], dim=1), self.trajectory, 0.001, b_pos=actions_residual[:, start_index].unsqueeze(-1), b_rot=actions_residual[:, start_index+1])
        else:
          new_actions = act(self.plug_points, self.socket_points, torch.cat([self.noisy_plug_pos, self.noisy_plug_quat], dim=1), self.trajectory, 0.001)
          #print("new actions start", new_actions)
        #print("noise socket quat", self.noisy_socket_quat, "noisy plug quat", self.noisy_plug_quat) 
        #new_actions[:, 3:6] = torch.clip(self.quaternion_towards(self.noisy_socket_quat, self.noisy_plug_quat)*10.0, -1.0, 1.0)
        #if self.progress_buf[0] >= 10:
        #  print("now only rotation")
        #new_actions[:, 0:3] *= 0.0
        #Apply residual:
        if self.entire_action_residual == True:
          #Is in -1,1 so scale down, add, then scale
          actions_residual *= self.residual_scale
          new_actions += actions_residual
          max_val, _ = torch.max(torch.abs(new_actions), dim=1, keepdims=True)
          max_val[max_val==0.0] = 1.0
          new_actions /= max_val
        
        #Verified this system works with the scaling
        if self.self_scaling_residual == True:
          lowest_scale, _ = torch.min(torch.abs(actions_residual[:, 6:8]), dim=1, keepdims=True) #last 2 values are scaling for the RL
          actions_residual[:, 0:6] *= lowest_scale
          #print(actions_residual, "new actions before", new_actions)
          new_actions += actions_residual[:, 0:6]
          max_val, _ = torch.max(torch.abs(new_actions), dim=1, keepdims=True)
          max_val[max_val==0.0] = 1.0
          new_actions /= max_val
          #print("new actions after", new_actions)
          
        if self.simple_self_scaling_residual == True:
          lowest_scale = torch.abs(actions_residual[:, 6]).unsqueeze(-1)
          actions_residual[:, 0:6] *= lowest_scale
          #print(actions_residual, "new actions before", new_actions)
          new_actions += actions_residual[:, 0:6]
          max_val, _ = torch.max(torch.abs(new_actions), dim=1, keepdims=True)
          max_val[max_val==0.0] = 1.0
          new_actions /= max_val
          #print("new actions after", new_actions)
        
        #print("new actions after", new_actions)
        self.actions=new_actions.clone().to(self.device)
        #print("residual servoing actions", self.actions)
        self._apply_actions_as_ctrl_targets_noisy(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

    def add_noise_to_pos_quats(self):
        #--------------------------------------------------------------------------Adding noise
        # Add observation noise to socket pos
        self.noisy_socket_pos = torch.zeros_like(
            self.socket_pos, dtype=torch.float32, device=self.device
        )
        socket_obs_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_pos_noise = socket_obs_pos_noise @ torch.diag(
            torch.tensor(
                self.socket_pos_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.noisy_socket_pos[:, 0] = self.socket_pos[:, 0] + socket_obs_pos_noise[:, 0]
        self.noisy_socket_pos[:, 1] = self.socket_pos[:, 1] + socket_obs_pos_noise[:, 1]
        self.noisy_socket_pos[:, 2] = self.socket_pos[:, 2] + socket_obs_pos_noise[:, 2]

        # Add observation noise to socket rot --------------------------------------------------------------This noise is wrong!
        socket_rot_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        #socket_rot_euler = torch_utils.get_euler_xyz(self.socket_quat)
        socket_obs_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_obs_rot_noise = socket_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.socket_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        socket_obs_rot_euler = socket_rot_euler + socket_obs_rot_noise
        noise_socket_quat = torch_utils.quat_from_euler_xyz(
            socket_obs_rot_euler[:, 0],
            socket_obs_rot_euler[:, 1],
            socket_obs_rot_euler[:, 2],
        )
        
        socket_obs_rot_noise_quat = torch_jit_utils.quat_from_euler_xyz(socket_obs_rot_noise[:, 0], socket_obs_rot_noise[:, 1], socket_obs_rot_noise[:, 2])
        #------------------------------------------------------------------------Still need to apply this
        
        self.noisy_socket_quat = torch_utils.quat_mul(self.socket_quat, noise_socket_quat)
        
        
        #plug noise
        self.noisy_plug_pos = torch.zeros_like(self.plug_pos, dtype=torch.float32, device=self.device)
        plug_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)- 0.5)
        plug_obs_pos_noise = plug_obs_pos_noise @ torch.diag(torch.tensor(self.plug_pos_obs_noise,dtype=torch.float32,device=self.device,))
        self.noisy_plug_pos[:, 0] = self.plug_pos[:, 0] + plug_obs_pos_noise[:, 0]
        self.noisy_plug_pos[:, 1] = self.plug_pos[:, 1] + plug_obs_pos_noise[:, 1]
        self.noisy_plug_pos[:, 2] = self.plug_pos[:, 2] + plug_obs_pos_noise[:, 2]

        if self.cfg_task.env.correlated_noise == True:
          self.noisy_plug_pos[:, :] += self.correlated_peg_obs_pos_noise[:, :]
        
        # Add observation noise to peg rot
        plug_rot_euler = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        #plug_rot_euler = torch_utils.get_euler_xyz(self.plug_quat)
        plug_obs_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        plug_obs_rot_noise = plug_obs_rot_noise @ torch.diag(
            torch.tensor(
                self.plug_rot_obs_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )

        plug_obs_rot_euler = plug_rot_euler + plug_obs_rot_noise
        noise_plug_quat = torch_utils.quat_from_euler_xyz(
            plug_obs_rot_euler[:, 0],
            plug_obs_rot_euler[:, 1],
            plug_obs_rot_euler[:, 2],
        )
        self.noisy_plug_quat = torch_utils.quat_mul(self.plug_quat, noise_plug_quat)
        #--------------------------------------------------------------------------Adding noise

    def pre_physics_step(self, actions_original):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""
        se3_pos = torch.cat([self.plug_pos[0, :], self.plug_quat[0, :]], dim=0).to(device)
        #print("se3_pos_shape", se3_pos.shape)
        self.pose_tensor = torch.cat([self.pose_tensor, se3_pos.unsqueeze(0)], dim=0).to(device)
        #print(self.pose_tensor.shape)
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:
          #np.savetxt('trajectory_pos_noise_{self.cfg_task.env.plug_pos_obs_noise[0]}.txt', self.pose_tensor)
          torch.save(self.pose_tensor, 'trajectory_pos_noise.pt')
          
        self.add_noise_to_pos_quats()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        if self.cfg_task.env.printing == True:
          print("actions", self.actions)

        self.run_actions = False

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        #Residual
        actions_original -= 1.0
        actions_original /= 2.0
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------POTENTIAL FIELD------------#
        #print("noisy vs real", self.noisy_plug_pos[:, :], self.plug_pos[:, :])
        resolution=10
        if self.potential_field == True:
          #convert sockets and pegs then run residual_servoing attractor
          #import time
          #print("updating vertices start")
          #start_time = time.time()
          self.plug_points = update_vertices_parallel(self.plug_points_template.clone(), self.noisy_plug_quat, self.noisy_plug_pos)
          self.socket_points = update_vertices_parallel(self.socket_points_template.clone(), self.noisy_socket_quat, self.noisy_socket_pos)
          self.trajectory = define_trajectory_parallel(self.noisy_socket_pos)
          #print(self.noisy_plug_pos)
          #print(self.trajectory[0])
          self.residual_servoing_attractor(actions_original)
          #print("updating vertices end", time.time()-start_time)
          return
        
          """
          #print("starting point creation")
          if self.progress_buf[0] == 0:
            #create sockets if it is the first step
            print("noisy socket pos vs real pos and quats", self.noisy_socket_pos[:, :].shape, self.noisy_socket_quat[:, :].shape, self.socket_pos[:, :].shape, self.socket_quat[:, :].shape)
            self.socket_points = create_socket(torch.cat([self.noisy_socket_pos[0, :], self.noisy_socket_quat[0, :]], dim=0), resolution=resolution).unsqueeze(0)
            #When I make a new socket, the different noisy offsets means there are different numbers of points meaning I cannot concatenate them, more importantly the straight-down quats
            #So remove the points before rotating?
            #For now everyone will use the same noisy socket and I will just do many batches
            for jj in range(1, self.noisy_socket_pos.shape[0]):
              self.socket_points = torch.cat([self.socket_points, create_socket(torch.cat([self.noisy_socket_pos[jj, :], self.noisy_socket_quat[jj, :]], dim=0), resolution=resolution).unsqueeze(0)], dim=0)
              #self.socket_points = torch.cat([self.socket_points, create_socket(torch.cat([self.noisy_socket_pos[0, :], self.noisy_socket_quat[0, :]], dim=0), resolution=resolution).unsqueeze(0)], dim=0)
            self.socket_points.to(self.device)
            self.trajectory=define_trajectory(resolution=200)
          #always create new pegs
          self.plug_points = create_peg(torch.cat([self.noisy_plug_pos[0, :], self.noisy_plug_quat[0, :]], dim=0), resolution=resolution).unsqueeze(0)
          for jj in range(1, self.noisy_socket_pos.shape[0]):
            self.plug_points = torch.cat([self.plug_points, create_peg(torch.cat([self.noisy_plug_pos[jj, :], self.noisy_plug_quat[jj, :]], dim=0), resolution=resolution).unsqueeze(0)], dim=0)
          self.plug_points.to(self.device)
          #print("ending point creation")
          self.residual_servoing_attractor(actions_original)
          #print("ending calculating actions")
          return
          """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------POTENTIAL FIELD------------#
        
        if self.cfg_task.env.printing == True:
          print("actions", self.actions)
        self.run_actions = False
        #Added---------------------------------------------------------------------------------------------------------------------------#
        current_state = [
                    self.noisy_plug_pos,
                    self.noisy_plug_quat,
                    self.noisy_socket_pos,
                    self.noisy_socket_quat,]
                                                  
        current_state = torch.cat(current_state, dim=-1).to(self.device) 

        prepared_state = self.prepare_state(current_state)

        actions = self.policy(prepared_state) #output axis angles
        if self.pos_only_policy == True:
          actions = torch.cat([actions, torch.zeros_like(actions)], axis=-1)
        #print("actions:", actions[0])   
        if self.cfg_task.env.printing == True:
          values = self.vf(prepared_state)
          print("value", values)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]
        #Added---------------------------------------------------------------------------------------------------------------------------#

        self._apply_actions_as_ctrl_targets_noisy(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        delta_pos = self.gripper_goal_pos - self.fingertip_centered_pos
        noisy_delta_pos = self.noisy_gripper_goal_pos - self.fingertip_centered_pos

        # Define observations (for actor)
        obs_tensors = [
            self.arm_dof_pos,  # 7
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                1
            ],  # 4
            self.pose_world_to_robot_base(
                self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.noisy_gripper_goal_pos, self.noisy_gripper_goal_quat
            )[
                1
            ],  # 4
            noisy_delta_pos,
        ]  # 3
        
        # Define state (for critic)
        state_tensors = [
            self.arm_dof_pos,  # 7
            self.arm_dof_vel,  # 7
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.fingertip_centered_pos, self.fingertip_centered_quat
            )[
                1
            ],  # 4
            self.fingertip_centered_linvel,  # 3
            self.fingertip_centered_angvel,  # 3
            self.pose_world_to_robot_base(
                self.gripper_goal_pos, self.gripper_goal_quat
            )[
                0
            ],  # 3
            self.pose_world_to_robot_base(
                self.gripper_goal_pos, self.gripper_goal_quat
            )[
                1
            ],  # 4
            delta_pos,  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0],  # 3
            self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1],  # 4
            noisy_delta_pos - delta_pos,
        ]  # 3

        #self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        self.states_buf = torch.cat(state_tensors, dim=-1)

        #put in robot frame
        current_state = [
            self.noisy_plug_pos,
            self.noisy_plug_quat,
            self.noisy_socket_pos,
            self.noisy_socket_quat,]
        self.obs_buf = torch.cat(current_state, dim=-1).to(self.device)
        #self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        #print("obs_buf", self.obs_buf)
        return self.obs_buf

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        """
        Compute reward at current timestep.
        They will apply the rewards properly later. This is just the immediate reward for the given timestep as the self.rew_buf values.
        Just use a sparse reward for engagement or insertion that checks each step
        If we high interpenerate zero the reward and if we low interpenetrate then it should stay same reward value
        At the end here we check for if its the final step so I make noise adjustments based on success and shift the correlated noise
        """

        self.prev_rew_buf = self.rew_buf.clone()

        # SAPU: Compute reward scale based on interpenetration distance
        low_interpen_envs, high_interpen_envs = [], []
        (
            low_interpen_envs,
            high_interpen_envs,
            sapu_reward_scale,
        ) = algo_utils.get_sapu_reward_scale(
            asset_indices=self.asset_indices,
            plug_pos=self.plug_pos,
            plug_quat=self.plug_quat,
            socket_pos=self.socket_pos,
            socket_quat=self.socket_quat,
            wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
            wp_socket_meshes=self.wp_socket_meshes,
            interpen_thresh=self.cfg_task.rl.interpen_thresh,
            wp_device=self.wp_device,
            device=self.device,
        )

        #After calculing interpenetration in the simualtor, check engagements:
        # Success bonus: Check which envs have plug engaged (partially inserted) or fully inserted
        is_plug_engaged_w_socket = algo_utils.check_plug_engaged_w_socket(
            plug_pos=self.plug_pos,
            socket_top_pos=self.socket_top_pos,
            keypoints_plug=self.keypoints_plug,
            keypoints_socket=self.keypoints_socket,
            cfg_task=self.cfg_task,
            progress_buf=self.progress_buf,
        )
        is_plug_inserted_in_socket = algo_utils.check_plug_inserted_in_socket(
            plug_pos=self.plug_pos,
            socket_pos=self.socket_pos,
            keypoints_plug=self.keypoints_plug,
            keypoints_socket=self.keypoints_socket,
            cfg_task=self.cfg_task,
            progress_buf=self.progress_buf,
        )
        is_plug_halfway_inserted_in_socket = algo_utils.check_plug_halfway_inserted_in_socket(
            plug_pos=self.plug_pos,
            socket_top_pos=self.socket_top_pos,
            keypoints_plug=self.keypoints_plug,
            keypoints_socket=self.keypoints_socket,
            cfg_task=self.cfg_task,
            progress_buf=self.progress_buf,
        )

        # Success bonus: Apply sparse rewards for engagement and insertion
        #Make sure to do an equals here first because if you grey out the other options of setting = then just adds rewards perpetually and this creates just higher rewards as you go, ruining training
        self.rew_buf[:] = 1.0*is_plug_engaged_w_socket #self.cfg_task.rl.engagement_bonus 
        self.rew_buf[:] += 100.0*is_plug_inserted_in_socket #self.cfg_task.rl.engagement_bonus

        if len(high_interpen_envs) > 0:
            self.rew_buf[high_interpen_envs] = 0.0#0.1 * (self.prev_rew_buf[high_interpen_envs]) #Scale down reward of previous step I guess
        #self.rew_buf[low_interpen_envs] *= 1.0
            
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:
          # Success bonus: Log success rate, ignoring environments with large interpenetration
          if len(high_interpen_envs) > 0:
              is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket[
                  low_interpen_envs
              ]
              #print(low_interpen_envs.shape)
              #print("is inserted low interpen shape", is_plug_inserted_in_socket_low_interpen.shape)
              self.extras["insertion_successes"] = torch.sum(
                  is_plug_inserted_in_socket_low_interpen.float()
              ) / self.num_envs
              if len(low_interpen_envs) == 0:
                self.extras["insertion_successes"] = 0.0
              print("percentage_plug_inserted_in_socket with low interpenetration:", self.extras["insertion_successes"] * 100.0)
              percent_inserted_nointerpen_just_from_there=torch.mean(
                  is_plug_inserted_in_socket_low_interpen.float()
              )
              print("percentage_plug_inserted_in_socket from low interpenetration that inserted:", percent_inserted_nointerpen_just_from_there * 100.0)
          else:
              self.extras["insertion_successes"] = torch.mean(
                  is_plug_inserted_in_socket.float()
              )
              print("percentage_plug_inserted_in_socket with low interpenetration:", self.extras["insertion_successes"] * 100.0)

          # SBC: Log current max downward displacement of plug at beginning of episode
          self.extras["curr_max_disp"] = self.curr_max_disp

          # SBC: Update curriculum difficulty based on success rate
          self.curr_max_disp = algo_utils.get_new_max_disp(
              curr_success=self.extras["insertion_successes"],
              cfg_task=self.cfg_task,
              curr_max_disp=self.curr_max_disp,
          )

          '''
          Calculate and Print the percentage of pegs that were engaged with the socke
          '''
          #print(is_plug_engaged_w_socket)
          #print(is_plug_inserted_in_socket)
          is_plug_engaged_w_socket_episode = torch.sum(is_plug_engaged_w_socket).item()
          print("is_plug_engaged_w_socket_episode:", is_plug_engaged_w_socket_episode, "out of", self.num_envs)
          # Calculate the percentage of pegs that were engaged with the socket
          percentage_plug_engaged_w_socket = (is_plug_engaged_w_socket_episode/self.num_envs)*100
          print("percentage_plug_engaged_w_socket:", percentage_plug_engaged_w_socket)

          is_plug_halfway_inserted_in_socket_episode = torch.sum(is_plug_halfway_inserted_in_socket).item()
          print("is_plug_halfway_inserted_in_socket_episode:", is_plug_halfway_inserted_in_socket_episode, "out of", self.num_envs)
          # Calculate the percentage of pegs that were engaged with the socket
          percentage_is_plug_halfway_inserted_in_socket = (is_plug_halfway_inserted_in_socket_episode/self.num_envs)*100
          print("percentage_is_plug_halfway_inserted_in_socket:", percentage_is_plug_halfway_inserted_in_socket)

          '''
          Calculate and Print the percentage of pegs that were inserted in the socket
          '''
          is_plug_inserted_in_socket_episode = torch.sum(is_plug_inserted_in_socket).item()
          print("is_plug_inserted_in_socket_episode:", is_plug_inserted_in_socket_episode, "out of", self.num_envs)
          # Calculate the percentage of pegs that were inserted in the socket
          percentage_plug_inserted_in_socket = (is_plug_inserted_in_socket_episode/self.num_envs)*100
          self.extras["sapu_adjusted_reward"] = percentage_plug_inserted_in_socket #Just use this so we can see the value on wandb
          print("percentage_plug_inserted_in_socket:", percentage_plug_inserted_in_socket)
          
          #Updates that it is the last step
          #Based on the final success rate adjust the noise parameters
          if self.cfg_task.env.noise_shift_with_success_rate == True:
            self.extras["sdf_reward"] = self.plug_pos_obs_noise[0]#Use this as a proxy for current peg pos noise
            #Scale then print the noise values. So we have it as a scale of percentages. So we choose a max noise value and the percent success times that is the noise we choose
            scaling_percent_inserted_nointerpen = torch.sum(is_plug_inserted_in_socket.float()).item() / self.num_envs
            scaling_percent_inserted_not_considering_interpen = scaling_percent_inserted_nointerpen
            if len(high_interpen_envs) > 0:
              if len(low_interpen_envs) > 0: #If we still have some low_interpen_envs
                is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket[low_interpen_envs]
              else: #All high interpen so just zeros
                is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket * 0.0
              scaling_percent_inserted_nointerpen = torch.sum(is_plug_inserted_in_socket_low_interpen.float()).item() / self.num_envs
            

            step_scale=self.cfg_task.env.step_scale
            improve_limit=self.cfg_task.env.improve_limit
            adding=-1.0
            #if scaling_percent_inserted_nointerpen > improve_limit:
            if scaling_percent_inserted_not_considering_interpen > improve_limit:
              adding=1.0
            print(torch.tensor(self.cfg_task.env.socket_pos_obs_noise, dtype=torch.float32, device=self.device) * step_scale * adding)
            self.socket_pos_obs_noise = \
              torch.clip(
                torch.tensor(self.socket_pos_obs_noise, device=self.device) + (torch.tensor(self.cfg_task.env.socket_pos_obs_noise, dtype=torch.float32, device=self.device) * step_scale * adding), 
                0.0,
                self.cfg_task.env.socket_pos_obs_noise[0]
                ).tolist()
            self.socket_rot_obs_noise = \
              torch.clip(
                torch.tensor(self.socket_rot_obs_noise, device=self.device) + (torch.tensor(self.cfg_task.env.socket_rot_obs_noise, dtype=torch.float32, device=self.device) * step_scale * adding), 
                0.0,
                self.cfg_task.env.socket_rot_obs_noise[0]
                ).tolist()
            self.plug_pos_obs_noise = \
              torch.clip(
                torch.tensor(self.plug_pos_obs_noise, device=self.device) + (torch.tensor(self.cfg_task.env.plug_pos_obs_noise, dtype=torch.float32, device=self.device) * step_scale * adding), 
                0.0,
                self.cfg_task.env.plug_pos_obs_noise[0]
                ).tolist()
            self.plug_rot_obs_noise = \
              torch.clip(
                torch.tensor(self.plug_rot_obs_noise, device=self.device) + (torch.tensor(self.cfg_task.env.plug_rot_obs_noise, dtype=torch.float32, device=self.device) * step_scale * adding), 
                0.0,
                self.cfg_task.env.plug_rot_obs_noise[0]
                ).tolist()
            print("|-------Noises->\nSockets", self.socket_pos_obs_noise, self.socket_rot_obs_noise, "\nPegs:", self.plug_pos_obs_noise, self.plug_rot_obs_noise)
            
          #Also change the base shift for each environment
          if self.cfg_task.env.correlated_noise == True:
            #Not sure how correlated rotational noise will work but positional sounds a bit easier
            peg_obs_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)
            self.correlated_peg_obs_pos_noise = self.cfg_task.env.correlated_noise_relative_scale * peg_obs_pos_noise @ torch.diag(torch.tensor(self.plug_pos_obs_noise, 
              dtype=torch.float32, device=self.device,))

    def _update_rew_buf_dep(self):
        """Compute reward at current timestep."""

        self.prev_rew_buf = self.rew_buf.clone()
 
        # SDF-Based Reward: Compute reward based on SDF distance
        sdf_reward = algo_utils.get_sdf_reward(
            wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
            asset_indices=self.asset_indices,
            plug_pos=self.plug_pos,
            plug_quat=self.plug_quat,
            plug_goal_sdfs=self.plug_goal_sdfs,
            wp_device=self.wp_device,
            device=self.device,
        )

        # SDF-Based Reward: Apply reward
        self.rew_buf[:] = self.cfg_task.rl.sdf_reward_scale * sdf_reward

        # SDF-Based Reward: Log reward
        self.extras["sdf_reward"] = torch.mean(self.rew_buf)

        # SAPU: Compute reward scale based on interpenetration distance
        low_interpen_envs, high_interpen_envs = [], []
        (
            low_interpen_envs,
            high_interpen_envs,
            sapu_reward_scale,
        ) = algo_utils.get_sapu_reward_scale(
            asset_indices=self.asset_indices,
            plug_pos=self.plug_pos,
            plug_quat=self.plug_quat,
            socket_pos=self.socket_pos,
            socket_quat=self.socket_quat,
            wp_plug_meshes_sampled_points=self.wp_plug_meshes_sampled_points,
            wp_socket_meshes=self.wp_socket_meshes,
            interpen_thresh=self.cfg_task.rl.interpen_thresh,
            wp_device=self.wp_device,
            device=self.device,
        )

        # SAPU: For envs with low interpenetration, apply reward scale ("weight" step)
        self.rew_buf[low_interpen_envs] *= sapu_reward_scale

        # SAPU: For envs with high interpenetration, do not update reward ("filter" step)
        if len(high_interpen_envs) > 0:
            self.rew_buf[high_interpen_envs] = self.prev_rew_buf[high_interpen_envs]

        # SAPU: Log reward after scaling and adjustment from SAPU
        self.extras["sapu_adjusted_reward"] = torch.mean(self.rew_buf)
        
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:
            # Success bonus: Check which envs have plug engaged (partially inserted) or fully inserted
            is_plug_engaged_w_socket = algo_utils.check_plug_engaged_w_socket(
                plug_pos=self.plug_pos,
                socket_top_pos=self.socket_top_pos,
                keypoints_plug=self.keypoints_plug,
                keypoints_socket=self.keypoints_socket,
                cfg_task=self.cfg_task,
                progress_buf=self.progress_buf,
            )
            is_plug_inserted_in_socket = algo_utils.check_plug_inserted_in_socket(
                plug_pos=self.plug_pos,
                socket_pos=self.socket_pos,
                keypoints_plug=self.keypoints_plug,
                keypoints_socket=self.keypoints_socket,
                cfg_task=self.cfg_task,
                progress_buf=self.progress_buf,
            )
            is_plug_halfway_inserted_in_socket = algo_utils.check_plug_halfway_inserted_in_socket(
                plug_pos=self.plug_pos,
                socket_top_pos=self.socket_top_pos,
                keypoints_plug=self.keypoints_plug,
                keypoints_socket=self.keypoints_socket,
                cfg_task=self.cfg_task,
                progress_buf=self.progress_buf,
            )

            # Success bonus: Compute reward scale based on whether plug is engaged with socket, as well as closeness to full insertion
            engagement_reward_scale = algo_utils.get_engagement_reward_scale(
                plug_pos=self.plug_pos,
                socket_pos=self.socket_pos,
                is_plug_engaged_w_socket=is_plug_engaged_w_socket,
                success_height_thresh=self.cfg_task.rl.success_height_thresh,
                device=self.device,
            )

            # Success bonus: Apply reward with reward scale
            self.rew_buf[:] += (
                engagement_reward_scale * self.cfg_task.rl.engagement_bonus
            )

            # Success bonus: Log success rate, ignoring environments with large interpenetration
            if len(high_interpen_envs) > 0:
                is_plug_inserted_in_socket_low_interpen = is_plug_inserted_in_socket[
                    low_interpen_envs
                ]
                self.extras["insertion_successes"] = torch.mean(
                    is_plug_inserted_in_socket_low_interpen.float()
                )
                print("insertion_success", self.extras["insertion_successes"])
            else:
                self.extras["insertion_successes"] = torch.mean(
                    is_plug_inserted_in_socket.float()
                )
                print("insertion_success", self.extras["insertion_successes"])

            # SBC: Compute reward scale based on curriculum difficulty
            sbc_rew_scale = algo_utils.get_curriculum_reward_scale(
                cfg_task=self.cfg_task, curr_max_disp=self.curr_max_disp
            )

            # SBC: Apply reward scale (shrink negative rewards, grow positive rewards)
            self.rew_buf[:] = torch.where(
                self.rew_buf[:] < 0.0,
                self.rew_buf[:] / sbc_rew_scale,
                self.rew_buf[:] * sbc_rew_scale,
            )

            # SBC: Log current max downward displacement of plug at beginning of episode
            self.extras["curr_max_disp"] = self.curr_max_disp

            # SBC: Update curriculum difficulty based on success rate
            self.curr_max_disp = algo_utils.get_new_max_disp(
                curr_success=self.extras["insertion_successes"],
                cfg_task=self.cfg_task,
                curr_max_disp=self.curr_max_disp,
            )
            '''
            Calculate and Print the percentage of pegs that were engaged with the socke
            '''
            #print(is_plug_engaged_w_socket)
            #print(is_plug_inserted_in_socket)
            is_plug_engaged_w_socket_episode = torch.sum(is_plug_engaged_w_socket).item()
            print("is_plug_engaged_w_socket_episode:", is_plug_engaged_w_socket_episode, "out of", self.num_envs)
            # Calculate the percentage of pegs that were engaged with the socket
            percentage_plug_engaged_w_socket = (is_plug_engaged_w_socket_episode/self.num_envs)*100
            print("percentage_plug_engaged_w_socket:", percentage_plug_engaged_w_socket)

            is_plug_halfway_inserted_in_socket_episode = torch.sum(is_plug_halfway_inserted_in_socket).item()
            print("is_plug_halfway_inserted_in_socket_episode:", is_plug_halfway_inserted_in_socket_episode, "out of", self.num_envs)
            # Calculate the percentage of pegs that were engaged with the socket
            percentage_is_plug_halfway_inserted_in_socket = (is_plug_halfway_inserted_in_socket_episode/self.num_envs)*100
            print("percentage_is_plug_halfway_inserted_in_socket:", percentage_is_plug_halfway_inserted_in_socket)

            '''
            Calculate and Print the percentage of pegs that were inserted in the socket
            '''
            is_plug_inserted_in_socket_episode = torch.sum(is_plug_inserted_in_socket).item()
            print("is_plug_inserted_in_socket_episode:", is_plug_inserted_in_socket_episode, "out of", self.num_envs)
            # Calculate the percentage of pegs that were inserted in the socket
            percentage_plug_inserted_in_socket = (is_plug_inserted_in_socket_episode/self.num_envs)*100
            print("percentage_plug_inserted_in_socket:", percentage_plug_inserted_in_socket)
            

    def _update_reset_buf(self):
        """Assign environments for reset if maximum episode length has been reached."""

        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka()

        # Close gripper onto plug
        self.disable_gravity()  # to prevent plug from falling
        self._reset_object()
        self._move_gripper_to_grasp_pose(
            sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
        )
        
        self.close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
        
        target_position=self.plug_pos.clone()
        steps_total=50
        plug_rot_quat=None
        for i in range(steps_total):
          fix_actions,plug_rot_quat=self.create_rotational_randomization_axis_angle(plug_rot_quat=plug_rot_quat, target_position=target_position)
          #print("plug rot quat", plug_rot_quat)
          #if i < int(0):
          #if i == steps_total-1:
          #  print("fix actions", fix_actions)
          fix_actions[:, 0:3] *= 10.0
          #else:
          fix_actions[:, 3:6] *= 10.0
          if i > int(steps_total/2):
            fix_actions[:, 0:3] *= 20.0
            fix_actions[:, 3:6] *= 0.0

          self._apply_actions_as_ctrl_targets_noisy(actions=fix_actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True)
          #self._apply_actions_as_ctrl_targets_randomize_rotations(fix_actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True) #Rotation is relative to the bottom of the peg
          self.simulate_and_refresh()
        print("######################FINISHED SETTING##########################################\n")
        self.add_noise_to_pos_quats()
        
        self.enable_gravity()

        # Get plug SDF in goal pose for SDF-based reward
        self.plug_goal_sdfs = algo_utils.get_plug_goal_sdfs(
            wp_plug_meshes=self.wp_plug_meshes,
            asset_indices=self.asset_indices,
            socket_pos=self.socket_pos,
            socket_quat=self.socket_quat,
            wp_device=self.wp_device,
        )

        self._reset_buffers()

    def _reset_franka(self):
        """Reset DOF states, DOF torques, and DOF targets of Franka."""

        # Randomize DOF pos
        self.dof_pos[:] = torch.cat(
            (
                torch.tensor(
                    self.cfg_task.randomize.franka_arm_initial_dof_pos,
                    device=self.device,
                ),
                torch.tensor(
                    [self.asset_info_franka_table.franka_gripper_width_max],
                    device=self.device,
                ),
                torch.tensor(
                    [self.asset_info_franka_table.franka_gripper_width_max],
                    device=self.device,
                ),
            ),
            dim=-1,
        ).unsqueeze(
            0
        )  # shape = (num_envs, num_dofs)

        # Stabilize Franka
        self.dof_vel[:, :] = 0.0  # shape = (num_envs, num_dofs)
        self.dof_torque[:, :] = 0.0
        self.ctrl_target_dof_pos = self.dof_pos.clone()
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

        # Set DOF state
        franka_actor_ids_sim = self.franka_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_torque),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def _reset_object(self):
        """Reset root state of plug and socket."""

        self._reset_socket()
        self._reset_plug(before_move_to_grasp=True)

    def _reset_socket(self):
        """Reset root state of socket."""

        # Randomize socket pos
        socket_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_noise_xy = socket_noise_xy @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.socket_pos_xy_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        socket_noise_z = torch.zeros(
            (self.num_envs), dtype=torch.float32, device=self.device
        )
        socket_noise_z_mag = (
            self.cfg_task.randomize.socket_pos_z_noise_bounds[1]
            - self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
        )
        socket_noise_z = (
            socket_noise_z_mag
            * torch.rand((self.num_envs), dtype=torch.float32, device=self.device)
            + self.cfg_task.randomize.socket_pos_z_noise_bounds[0]
        )

        self.socket_pos[:, 0] = (
            self.robot_base_pos[:, 0]
            + self.cfg_task.randomize.socket_pos_xy_initial[0]
            + socket_noise_xy[:, 0]
        )
        self.socket_pos[:, 1] = (
            self.robot_base_pos[:, 1]
            + self.cfg_task.randomize.socket_pos_xy_initial[1]
            + socket_noise_xy[:, 1]
        )
        self.socket_pos[:, 2] = self.cfg_base.env.table_height + socket_noise_z

        # Randomize socket rot
        socket_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        socket_rot_noise = socket_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.socket_rot_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        socket_rot_euler = (
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            + socket_rot_noise
        )
        socket_rot_quat = torch_utils.quat_from_euler_xyz(
            socket_rot_euler[:, 0], socket_rot_euler[:, 1], socket_rot_euler[:, 2]
        )
        self.socket_quat[:, :] = socket_rot_quat.clone()

        # Stabilize socket
        self.socket_linvel[:, :] = 0.0
        self.socket_angvel[:, :] = 0.0

        # Set socket root state
        socket_actor_ids_sim = self.socket_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(socket_actor_ids_sim),
            len(socket_actor_ids_sim),
        )

        # Simulate one step to apply changes
        self.simulate_and_refresh()

    def _reset_plug(self, before_move_to_grasp):
        """Reset root state of plug."""

        if before_move_to_grasp:
            # Generate randomized downward displacement based on curriculum
            #curr_curriculum_disp_range = (
            #    self.curr_max_disp - self.cfg_task.rl.curriculum_height_bound[0]
            #)
            #self.curriculum_disp = self.cfg_task.rl.curriculum_height_bound[
            #    0
            #] + curr_curriculum_disp_range * (
            #    torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            #)

            self.curriculum_disp = self.cfg_task.rl.curriculum_height_bound[0] #ADDED----------------------------------------------------------------------
            #self.curriculum_disp = self.cfg_task.rl.curriculum_height_bound[1] #To start partially inserted to see if it learns to go down
            # Generate plug pos noise
            self.plug_pos_xy_noise = 2 * (
                torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
                - 0.5
            )
            self.plug_pos_xy_noise = self.plug_pos_xy_noise @ torch.diag(
                torch.tensor(
                    self.cfg_task.randomize.plug_pos_xy_noise,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        self.plug_pos[:, :] = self.socket_pos.clone()
        self.plug_pos[:, 2] += self.socket_heights
        self.plug_pos[:, 2] -= self.curriculum_disp
        
        #------------------------------------------------------------------Added
        #self.plug_pos[:, 1] = 0.01
        #self.plug_pos[:, 2] = 1.07

        # Apply XY noise to plugs not partially inserted into sockets
        socket_top_height = self.socket_pos[:, 2] + self.socket_heights
        plug_partial_insert_idx = np.argwhere(
            self.plug_pos[:, 2].cpu().numpy() > socket_top_height.cpu().numpy()
        ).squeeze()
        self.plug_pos[plug_partial_insert_idx, :2] += self.plug_pos_xy_noise[
            plug_partial_insert_idx
        ]

        self.plug_quat[:, :] = self.identity_quat.clone()
        #print("plug pos before", self.plug_pos)
        # Randomize plug rot -----------------------------------------------------------------------------------------------------
        plug_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
        plug_rot_noise = plug_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.plug_rot_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        plug_rot_euler = (
            torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
            + plug_rot_noise
        )
        plug_rot_quat = torch_utils.quat_from_euler_xyz(
            plug_rot_euler[:, 0], plug_rot_euler[:, 1], plug_rot_euler[:, 2]
        )
        #self.plug_quat[:, :] = plug_rot_quat.clone() #------------------------------------------------------------------------------------Stopping plug rot randomization here
        #--------------------------------------------------------------------------------------------------------------------------

        # Stabilize plug
        self.plug_linvel[:, :] = 0.0
        self.plug_angvel[:, :] = 0.0

        # Set plug root state
        plug_actor_ids_sim = self.plug_actor_ids_sim.clone().to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(plug_actor_ids_sim),
            len(plug_actor_ids_sim),
        )

        # Simulate one step to apply changes
        #print("plug pos before refresh", self.plug_pos)
        self.simulate_and_refresh()
        #print("plug pos after refresh", self.plug_pos, "self.plug_quat", self.plug_quat)

    def _reset_buffers(self):
        """Reset buffers."""

        self.reset_buf[:] = 0
        self.progress_buf[:] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #"""
    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ):
        #Apply actions from policy as position/rotation targets.

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_centered_pos = (
            self.fingertip_centered_pos + pos_actions
        )
        #print("vel and torque", self.dof_vel[:, :], self.dof_torque[:, :])

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_centered_quat
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()
    #"""
    def DG_IndustReal(self, num_envs, device, jacobian_type):

            inv_Pose_Gripper_quat, inv_Pose_Gripper_t = torch_jit_utils.tf_inverse(self.fingertip_centered_quat, self.fingertip_centered_pos)
    
            curr_plug_pos = self.noisy_plug_pos
            curr_plug_quat = self.noisy_plug_quat

            #curr_plug_pos = self.robot_plug_pos
            #curr_plug_quat = self.robot_plug_quat

            #curr_plug_pos = self.noisy_plug_pos
            #curr_plug_quat = self.noisy_plug_quat

            DeltaG_quat, DeltaG_t = torch_jit_utils.tf_combine(inv_Pose_Gripper_quat, inv_Pose_Gripper_t, curr_plug_quat, curr_plug_pos)
            inv_DeltaG_quat, inv_DeltaG_t = torch_jit_utils.tf_inverse(DeltaG_quat, DeltaG_t)

            return inv_DeltaG_t, inv_DeltaG_quat
    
    def apply_invDG_IndustReal_insert(self, num_envs, target_pos, target_quat, inv_DeltaG_t, inv_DeltaG_quat, device):

            target_Pose_Gripper_quat, target_Pose_Gripper_t = torch_jit_utils.tf_combine(target_quat, target_pos, inv_DeltaG_quat, inv_DeltaG_t)
            return target_Pose_Gripper_t, target_Pose_Gripper_quat

    def refresh_and_acquire_tensors(self):
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self._acquire_env_tensors()
    
    def _apply_actions_as_ctrl_targets_noisy(self, actions, ctrl_target_gripper_dof_pos, do_scale):
            
            """
                Apply actions from policy as position/rotation targets.
            """
            #print("actions_ctrl_targets_noisy", actions)
            # Interpret actions as target pos displacements and set pos target
            pos_actions = actions[:, 0:3]
            if do_scale:
                pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
            
            #self.ctrl_target_fingertip_centered_pos = (self.fingertip_centered_pos + pos_actions)

            ############ Control Object Pose ############
            #self.robot_plug_pos = self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0]  # 3
            target_pos = self.noisy_plug_pos + pos_actions
            #target_pos = (self.noisy_plug_pos + pos_actions)
            #target_pos = self.robot_plug_pos + pos_actions
            ############################################

            
            # Interpret actions as target rot (axis-angle) displacements
            rot_actions = actions[:, 3:6]
            if do_scale:
                rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

            # Convert to quat and set rot target
            angle = torch.norm(rot_actions, p=2, dim=-1)
            axis = rot_actions / angle.unsqueeze(-1)
            rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
            if self.cfg_task.rl.clamp_rot:
                rot_actions_quat = torch.where(
                                                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                                rot_actions_quat,
                                                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
                                                                                                                                        )
            
            #self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_centered_quat)
            ####################### Control EEf ########################################
            #self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_centered_quat)
            #self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)
            ###########################################################################
            

            ####################### Control Object Pose ################################
            #self.robot_plug_quat = self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1] # 4
            
            #print("noisy quat in noisy action", self.noisy_plug_quat)
            target_quat = torch_utils.quat_mul(rot_actions_quat, self.noisy_plug_quat)
            #target_quat = torch_utils.quat_mul(rot_actions_quat, self.noisy_plug_quat)
            #target_quat = torch_utils.quat_mul(rot_actions_quat, self.robot_plug_quat)
            ############################################################################

            """
            print("*-------------------------------------------------------------------------------------------*")
            print("NN Output: ")
            print(" ")
            print(" ")
            print(" ")
            print("Current Pos:")
            print("state pos: ", self.plug_pos)
            print("current plug pos in the Robot frame: ", self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[0])
            print(" ")
            print("hole pos: ", self.socket_pos)
            print("robot_base_pos: ", self.robot_base_pos)
            print("pos_actions: ", pos_actions)
            print("next_state_pred_pos: ", target_pos)
            print("next_state pos GT: ", "I don't have one here")
            print(" ")
            print(" ")
            print(" ")
            print("Current_Quat")
            print("state quat: ", self.plug_quat)
            print("current plug quat in the Robot frame: ", self.pose_world_to_robot_base(self.plug_pos, self.plug_quat)[1])
            print(" ")
            print("hole_quat ", self.socket_quat)
            print("robot_base_quat: ", self.robot_base_quat)
            print("rot_actions_quat: ", rot_actions_quat)
            print("next_state_pred_quat: ", target_quat)
            print("next_state quat GT: ", "I don't have one here")
            print(" ")
            print(" ")
            print(" ")
            print("*-------------------------------------------------------------------------------------------*")
            """
            current_state = [
                                self.noisy_plug_pos,
                                self.noisy_plug_quat,
                                self.noisy_socket_pos,
                                self.noisy_socket_quat,
                                                    ]

            

            # Implement the kNN reset logic


            '''
            # Check if the reset condition is met
            if self.run_actions:
                if self.iterations_since_last_reset >= self.reset_threshold:
                    neighbors = find_k_nearest_neighbors(self.replay_buffer, current_state, k=10, batch_size=220000)

                    new_state = weighted_average_neighbors(neighbors)
                    new_state = new_state.to(self.device)
                    target_pos, target_quat = self.reset_to_state(new_state)
                    self.iterations_since_last_reset = 0
                else:
                    self.iterations_since_last_reset += 1
            '''

            #current_state_np = np.array(current_state)
            #current_state_tensor = torch.tensor(current_state_np, device=self.device)  # Replace with your current state
            current_state_tensor = torch.cat(current_state, dim=-1)
                                            
            '''
            self.plug_pos = self.plug_pos.to('cuda:0')
            self.plug_quat = self.plug_quat.to('cuda:0')
            self.socket_pos = self.socket_pos.to('cuda:0')
            self.socket_quat = self.socket_quat.to('cuda:0')

            # Ensure all tensors have the same number of dimensions
            # Assuming that pos tensors are of shape [1, 3] and quat tensors are of shape [1, 4]
            self.plug_pos = self.plug_pos.unsqueeze(0) if self.plug_pos.dim() == 1 else self.plug_pos
            self.socket_pos = self.socket_pos.unsqueeze(0) if self.socket_pos.dim() == 1 else self.socket_pos
            self.plug_quat = self.plug_quat.unsqueeze(0) if self.plug_quat.dim() == 1 else self.plug_quat
            self.socket_quat = self.socket_quat.unsqueeze(0) if self.socket_quat.dim() == 1 else self.socket_quat
            '''
            
            target_pos = target_pos.to('cuda:0')
            target_quat = target_quat.to('cuda:0')
            target_pos = target_pos.unsqueeze(0) if target_pos.dim() == 1 else target_pos
            target_quat = target_quat.unsqueeze(0) if target_quat.dim() == 1 else target_quat

            #self.BC
            if True:
                
                #while ( torch.norm(self.plug_pos[0, 2].double() - target_pos[0, 2].double(), p=2) > .001 ): #or torch.rad2deg(quat_diff_rad(self.plug_quat[0, :].unsqueeze(0), target_quat[0, :].unsqueeze(0))) > 1: 

                for _ in range(1):
                    
                    #print(" ")
                    #print("In the while loop...............................................................................................................")
                    #print("self.plug_pose: ", self.plug_pos, self.plug_quat)
                    #print("target_pose at this step: ", target_pos, target_quat)
                    #print("self.socket_pose: ", self.socket_pos, self.socket_quat)
                    #print(" ")
                    
                    #self.refresh_and_acquire_tensors()

                    ############################## Control Object Pose (DG Transformation ###############################################
                    inv_DeltaG_t, inv_DeltaG_quat = self.DG_IndustReal(self.num_envs, self.device, jacobian_type=None)
            
                    self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)

                    #self.ctrl_target_fingertip_midpoint_pos, self.ctrl_target_fingertip_midpoint_quat = self.apply_invDG_IndustReal_insert(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)
        #####################################################################################################################
                    s_t_pos_error, s_t_axis_angle_error = fc.get_pose_error(
                                                                                fingertip_midpoint_pos = self.fingertip_centered_pos,
                                                                                fingertip_midpoint_quat = self.fingertip_centered_quat,
                                                                                ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_centered_pos,
                                                                                ctrl_target_fingertip_midpoint_quat = self.ctrl_target_fingertip_centered_quat,
                                                                                jacobian_type=self.cfg_ctrl['jacobian_type'],
                                                                                rot_error_type='axis_angle'
                                                                                                                                                                        )
        
                    s_t_delta_hand_pose = torch.cat((s_t_pos_error, s_t_axis_angle_error), dim=-1)
                    s_t_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions_control), device = self.device)
                    s_t_actions[:, :6] = s_t_delta_hand_pose
                
                    s_t_actions_pos = s_t_actions[:, :3]
                    self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + s_t_actions_pos

                    s_t_actions_rot = s_t_actions[:, 3:]# * 20.0
                    #print("s_t_actions_rot", s_t_actions_rot, s_t_actions_pos)
                    s_t_angle = torch.norm(s_t_actions_rot, p=2, dim=-1)
                    s_t_axis = s_t_actions_rot / s_t_angle.unsqueeze(-1)
                    s_t_rot_actions_quat = torch_utils.quat_from_angle_axis(s_t_angle, s_t_axis)
                    self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(s_t_rot_actions_quat, self.fingertip_centered_quat)

                    self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

                    self.generate_ctrl_signals()
                    #self.zero_velocities(self.num_envs)
                    self.refresh_and_acquire_tensors()

                    '''
                    # Capture and write frames from each camera
                    for i_cam in range(self.num_cameras):
                        rgb_tensor = self.rgb_tensors[i_cam]
                        frame = rgb_tensor.cpu().numpy()
                        frame = np.transpose(frame, (1, 2, 0))  # Change tensor layout if necessary
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                        self.video_writer.write(frame)
                    '''

            else:
                inv_DeltaG_t, inv_DeltaG_quat = self.DG_IndustReal(self.num_envs, self.device, jacobian_type=None)

                self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)
            
            
                self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
                self.generate_ctrl_signals()
                
    def _apply_actions_as_ctrl_targets_randomize_rotations(self, actions, ctrl_target_gripper_dof_pos, do_scale):
            
            """
                Apply actions from policy as position/rotation targets.
            """

            # Interpret actions as target pos displacements and set pos target
            pos_actions = actions[:, 0:3]
            if do_scale:
                pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))


            target_pos = self.plug_pos + pos_actions
            
            # Interpret actions as target rot (axis-angle) displacements
            rot_actions = actions[:, 3:6]
            if do_scale:
                rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

            # Convert to quat and set rot target
            angle = torch.norm(rot_actions, p=2, dim=-1)
            axis = rot_actions / angle.unsqueeze(-1)
            rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
            if self.cfg_task.rl.clamp_rot:
                rot_actions_quat = torch.where(
                                                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                                rot_actions_quat,
                                                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),)

            target_quat = torch_utils.quat_mul(rot_actions_quat, self.noisy_plug_quat)

            current_state = [self.plug_pos,
                                self.plug_quat,
                                self.socket_pos,
                                self.socket_quat,
                                               ]
            current_state_tensor = torch.cat(current_state, dim=-1)

            if self.run_actions:
                if self.iterations_since_last_reset >= self.reset_threshold:
                    # Ensure current_state is in the correct format
                    neighbors = find_k_nearest_neighbors(self.replay_buffer, current_state_tensor, k=1000)
                    new_state = weighted_average_neighbors(neighbors)
                    new_state = new_state.to(self.device)
                    target_pos, target_quat = self.reset_to_state(new_state)
                    self.iterations_since_last_reset = 0
                else:
                    self.iterations_since_last_reset += 1
                                            
            
            target_pos = target_pos.to('cuda:0')
            target_quat = target_quat.to('cuda:0')
            target_pos = target_pos.unsqueeze(0) if target_pos.dim() == 1 else target_pos
            target_quat = target_quat.unsqueeze(0) if target_quat.dim() == 1 else target_quat

            #while torch.norm(self.plug_pos - target_pos, dim=1, p=2) logical or quatdiffrad close enough for all environments, continue control until reach all of them #While loop until all of them reach
            for _ in range(1):
              #print("target quat", "target pos", target_quat, target_pos, "current pose", self.plug_pos)
              print("plug quat", self.plug_quat, "noisy plug quat", self.noisy_plug_quat)
              inv_DeltaG_t, inv_DeltaG_quat = self.DG_IndustReal(self.num_envs, self.device, jacobian_type=None)
      
              self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)

              s_t_pos_error, s_t_axis_angle_error = fc.get_pose_error(
                                                                          fingertip_midpoint_pos = self.fingertip_centered_pos,
                                                                          fingertip_midpoint_quat = self.fingertip_centered_quat,
                                                                          ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_centered_pos,
                                                                          ctrl_target_fingertip_midpoint_quat = self.ctrl_target_fingertip_centered_quat,
                                                                          jacobian_type=self.cfg_ctrl['jacobian_type'],
                                                                          rot_error_type='axis_angle'
                                                                                                                                                                  )
  
              s_t_delta_hand_pose = torch.cat((s_t_pos_error, s_t_axis_angle_error), dim=-1)
              s_t_actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions_control), device = self.device)
              s_t_actions[:, :6] = s_t_delta_hand_pose
          
              s_t_actions_pos = s_t_actions[:, :3]
              self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + s_t_actions_pos

              s_t_actions_rot = s_t_actions[:, 3:]# * 20.0
              #print("s_t_actions_rot", s_t_actions_rot, s_t_actions_pos)
              s_t_angle = torch.norm(s_t_actions_rot, p=2, dim=-1)
              s_t_axis = s_t_actions_rot / s_t_angle.unsqueeze(-1)
              s_t_rot_actions_quat = torch_utils.quat_from_angle_axis(s_t_angle, s_t_axis)
              self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(s_t_rot_actions_quat, self.fingertip_centered_quat)

              self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

              self.generate_ctrl_signals()
              #self.zero_velocities(self.num_envs)
              self.refresh_and_acquire_tensors()

              self.ctrl_target_fingertip_centered_pos, self.ctrl_target_fingertip_centered_quat = self.apply_invDG_IndustReal_insert(num_envs = self.num_envs, target_pos = target_pos, target_quat = target_quat, inv_DeltaG_t = inv_DeltaG_t, inv_DeltaG_quat = inv_DeltaG_quat, device = self.device)
          
          
              self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
              self.generate_ctrl_signals()

    def _move_gripper_to_grasp_pose(self, sim_steps):
        """Define grasp pose for plug and move gripper to pose."""

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = self.plug_pos.clone()

        self.ctrl_target_fingertip_midpoint_pos[:, 2] += self.plug_grasp_offsets

        # Set target rot
        ctrl_target_fingertip_centered_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_centered_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_centered_euler[:, 0],
            ctrl_target_fingertip_centered_euler[:, 1],
            ctrl_target_fingertip_centered_euler[:, 2],
        )

        self.move_gripper_to_target_pose(
            gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            sim_steps=sim_steps,
        )

        # Reset plug in case it is knocked away by gripper movement
        self._reset_plug(before_move_to_grasp=False)
