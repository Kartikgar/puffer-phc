from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .policy import PolicyWithDiscriminator, RunningNorm
from .diffusion_policy import DiffusionPolicy
from .network import FeedForwardNN
from pufferlib.pytorch import layer_init


@dataclass
class FpoActionInfo:
    """Store FPO-specific action information for training"""
    x_t_path: torch.Tensor         # (*, flow_steps, action_dim)
    loss_eps: torch.Tensor         # (*, sample_dim, action_dim)
    loss_t: torch.Tensor           # (*, sample_dim, 1)  # Fixed: was missing field name
    initial_cfm_loss: torch.Tensor # (*,)


class FPODistribution:
    """
    A distribution-like wrapper for FPO that provides the interface expected by pufferlib.
    This stores the FPO-specific action information needed for training.
    """
    def __init__(self, action_info: FpoActionInfo, batch_size: int, action_dim: int, device: str):
        self.action_info = action_info
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.device = device
        
    def sample(self):
        # Return the predicted actions from FPO
        # action_info.x_t_path contains the full diffusion path, final action is at the end
        if self.action_info.x_t_path.ndim == 3:  # [batch, steps, action_dim]
            return self.action_info.x_t_path[:, -1, :]  # Take final step
        else:  # [steps, action_dim] - single sample
            return self.action_info.x_t_path[-1:, :].expand(self.batch_size, -1)
    
    def log_prob(self, actions):
        """
        For FPO, we compute log probability differently using the CFM loss.
        This is where the FPO-specific computation happens.
        """
        # This will be computed in the policy's custom loss computation
        # For now, return zeros for compatibility
        return torch.zeros(actions.shape[0], device=self.device)
    
    def entropy(self):
        """
        FPO doesn't have traditional entropy. Return zeros for compatibility.
        """
        return torch.zeros(self.batch_size, device=self.device)


class FPOPolicy(PolicyWithDiscriminator):
    """
    FPO policy that integrates with the pufferlib training system.
    Extends PolicyWithDiscriminator to maintain compatibility with existing infrastructure.
    """
    
    def __init__(self, env, hidden_size=512, **kwargs):
        super().__init__(env, hidden_size)
        
        # FPO-specific parameters
        self.num_fpo_samples = kwargs.get('num_fpo_samples', 100)
        self.positive_advantage = kwargs.get('positive_advantage', False)
        self.num_diffusion_steps = kwargs.get('num_diffusion_steps', 10)
        self.fixed_noise_inference = kwargs.get('fixed_noise_inference', False)
        
        print(f"Initializing FPO policy with {self.num_fpo_samples} samples, "
              f"positive_advantage={self.positive_advantage}")
        
        # Override the actor with a diffusion policy
        # Input dimension: observation + action + time step
        diffusion_input_dim = self.input_size + self.action_size + 1
        self.diffusion_actor = DiffusionPolicy(
            in_dim=diffusion_input_dim,
            out_dim=self.action_size,
            num_steps=self.num_diffusion_steps,
            fixed_noise_inference=self.fixed_noise_inference
        )
        
        # Keep the regular critic from parent class but create it explicitly
        # self.critic_mlp = nn.Sequential(
        #     layer_init(nn.Linear(self.input_size, 1024)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(1024, 512)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(512, 256)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(256, 1), std=0.01),
        # )
        
        # Storage for FPO-specific data during rollout
        self.stored_action_info = None
        self._fpo_mode = True  # Flag to indicate FPO mode
        
    def encode_observations(self, obs):
        """Encode observations through normalization"""
        self.obs_pointer = self.obs_norm(obs)
        return self.obs_pointer, None
    
    def decode_actions(self, hidden, lookup=None):
        """
        Decode actions using FPO diffusion policy.
        Returns FPO distribution and value.
        """
        batch_size = hidden.shape[0]
        
        # Get actions and FPO info from diffusion policy
        actions_list = []
        action_infos = []
        
        for i in range(batch_size):
            obs_single = hidden[i:i+1]  # Keep batch dimension
            
            with torch.no_grad():
                action, x_t_path, eps, t, initial_cfm_loss = self.diffusion_actor.sample_action_with_info(
                    obs_single.squeeze(0), self.num_fpo_samples
                )
            
            actions_list.append(action)
            action_info = FpoActionInfo(
                x_t_path=x_t_path,
                loss_eps=eps,
                loss_t=t,
                initial_cfm_loss=initial_cfm_loss
            )
            action_infos.append(action_info)
        
        # Stack all action infos for batch processing
        if batch_size > 1:
            # Combine action infos into batch format
            combined_info = FpoActionInfo(
                x_t_path=torch.stack([info.x_t_path for info in action_infos]),
                loss_eps=torch.stack([info.loss_eps for info in action_infos]),
                loss_t=torch.stack([info.loss_t for info in action_infos]),
                initial_cfm_loss=torch.stack([info.initial_cfm_loss for info in action_infos])
            )
        else:
            combined_info = action_infos[0]
        
        # Store for later use in custom loss computation
        self.stored_action_info = combined_info
        
        # Create FPO distribution
        fpo_dist = FPODistribution(
            combined_info, batch_size, self.action_size, hidden.device
        )
        
        # Compute value using critic
        value = self.critic_mlp(hidden)
        
        return fpo_dist, value
    
    def compute_fpo_loss(self, obs, actions, advantages, old_values, returns):
        """
        Custom FPO loss computation that should be called from the training loop.
        This replaces the standard PPO loss computation.
        """
        if self.stored_action_info is None:
            raise ValueError("No stored action info found. Make sure forward pass was called first.")
        
        action_info = self.stored_action_info
        batch_size = obs.shape[0]
        
        # Flatten batch and samples for CFM loss computation
        loss_eps = action_info.loss_eps  # [B, N, D]
        loss_t = action_info.loss_t      # [B, N, 1]
        initial_cfm_loss = action_info.initial_cfm_loss  # [B, N]
        
        # Handle different tensor shapes
        if loss_eps.ndim == 3:  # Batch format [B, N, D]
            B, N, D = loss_eps.shape
            flat_obs = obs.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            flat_acts = actions.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            flat_eps = loss_eps.reshape(B * N, D)
            flat_t = loss_t.reshape(B * N, 1)
            flat_init_loss = initial_cfm_loss.reshape(B * N)
        else:  # Single sample format [N, D]
            N, D = loss_eps.shape
            B = batch_size
            flat_obs = obs.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            flat_acts = actions.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            flat_eps = loss_eps.unsqueeze(0).expand(B, -1, -1).reshape(B * N, D)
            flat_t = loss_t.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 1)
            flat_init_loss = initial_cfm_loss.unsqueeze(0).expand(B, -1).reshape(B * N)
        
        # Compute CFM loss
        cfm_loss = self.diffusion_actor.compute_cfm_loss(flat_obs, flat_acts, flat_eps, flat_t)
        cfm_difference = flat_init_loss - cfm_loss
        
        # Convert back to batch format
        cfm_difference = cfm_difference.view(B, N)
        cfm_difference = torch.clamp(cfm_difference, -3, 3)
        
        # Compute importance sampling ratio for FPO
        rho_s = torch.exp(torch.clamp(cfm_difference.mean(dim=1), -3, 3))
        
        # Apply positive advantage transformation if enabled
        if self.positive_advantage:
            advantages = F.softplus(advantages)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # FPO policy loss (similar to PPO clipped objective)
        clip_coef = 0.2  # Can be made configurable
        surr1 = rho_s * advantages
        surr2 = torch.clamp(rho_s, 1 - clip_coef, 1 + clip_coef) * advantages
        policy_loss = -(torch.min(surr1, surr2)).mean()
        
        # Value loss
        values = self.critic_mlp(obs).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # Clear stored action info
        self.stored_action_info = None
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': torch.tensor(0.0, device=obs.device),  # No entropy in FPO
            'fpo_ratio': rho_s.mean(),
            'fpo_ratio_mean': rho_s.mean(),
            'fpo_ratio_min': rho_s.min(), 
            'fpo_ratio_max': rho_s.max(),
            'cfm_difference': cfm_difference.mean(),
        }

    def set_deterministic_action(self, deterministic):
        """Set deterministic action mode"""
        self._deterministic_action = deterministic
        if hasattr(self.diffusion_actor, 'fixed_noise_inference'):
            self.diffusion_actor.fixed_noise_inference = deterministic
