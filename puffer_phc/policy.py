import torch
from torch import nn

from pufferlib.pytorch import layer_init
import pufferlib.models

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

        # Point to the original policy's methods
        self.set_deterministic_action = self.policy.set_deterministic_action
        self.discriminate = self.policy.discriminate
        self.update_obs_rms = self.policy.update_obs_rms
        self.update_amp_obs_rms = self.policy.update_amp_obs_rms

    @property
    def mean_bound_loss(self):
        return self.policy.mean_bound_loss


class PolicyWithDiscriminator(nn.Module):
    def __init__(self, env, hidden_size=512):
        super().__init__()
        self.is_continuous = True
        self._deterministic_action = False

        self.input_size = env.single_observation_space.shape[0]
        self.action_size = env.single_action_space.shape[0]

        # Assume the action space is symmetric (low=-high)
        self.soft_bound = 0.9 * env.single_action_space.high[0]

        self.obs_norm = torch.jit.script(RunningNorm(self.input_size))

        ### Actor
        self.actor_mlp = None
        self.mu = nn.Sequential(
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )

        # NOTE: Original PHC uses a constant std. Something to experiment?
        self.sigma = nn.Parameter(
            torch.zeros(self.action_size, requires_grad=False, dtype=torch.float32),
            requires_grad=False,
        )
        nn.init.constant_(self.sigma, -2.9)

        ### Critic
        self.critic_mlp = None

        ### Discriminator
        self.use_amp_obs = env.amp_observation_space is not None
        self.amp_obs_norm = None

        if self.use_amp_obs:
            amp_obs_size = env.amp_observation_space.shape[0]
            self.amp_obs_norm = torch.jit.script(RunningNorm(amp_obs_size))

            self._disc_mlp = nn.Sequential(
                layer_init(nn.Linear(amp_obs_size, 1024)),
                nn.ReLU(),
                layer_init(nn.Linear(1024, hidden_size)),
                nn.ReLU(),
            )
            self._disc_logits = layer_init(torch.nn.Linear(hidden_size, 1))

        self.obs_pointer = None
        self.mean_bound_loss = None

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, obs):
        raise NotImplementedError

    def decode_actions(self, hidden, lookup=None):
        raise NotImplementedError

    def set_deterministic_action(self, value):
        self._deterministic_action = value

    def discriminate(self, amp_obs):
        if not self.use_amp_obs:
            return None

        norm_amp_obs = self.amp_obs_norm(amp_obs)
        disc_mlp_out = self._disc_mlp(norm_amp_obs)
        disc_logits = self._disc_logits(disc_mlp_out)
        return disc_logits

    # NOTE: Used for network weight regularization
    # def disc_logit_weights(self):
    #     return torch.flatten(self._disc_logits.weight)

    # def disc_weights(self):
    #     weights = []
    #     for m in self._disc_mlp.modules():
    #         if isinstance(m, nn.Linear):
    #             weights.append(torch.flatten(m.weight))

    #     weights.append(torch.flatten(self._disc_logits.weight))
    #     return weights

    def update_obs_rms(self, obs):
        self.obs_norm.update(obs)

    def update_amp_obs_rms(self, amp_obs):
        if not self.use_amp_obs:
            return

        self.amp_obs_norm.update(amp_obs)

    def bound_loss(self, mu):
        mu_loss = torch.zeros_like(mu)
        mu_loss = torch.where(mu > self.soft_bound, (mu - self.soft_bound) ** 2, mu_loss)
        mu_loss = torch.where(mu < -self.soft_bound, (mu + self.soft_bound) ** 2, mu_loss)
        return mu_loss.mean()


# NOTE: The PHC implementation, which has no LSTM. 17.0M params
class PHCPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        # NOTE: Original PHC network + LayerNorm
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        # NOTE: Original PHC network + LayerNorm
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the normed obs to use in the critic
        self.obs_pointer = self.obs_norm(obs)
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value


class LSTMCriticPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        # Actor: Original PHC network
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )
        self.mu = None

        ### Critic with LSTM
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.ReLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the normed obs to use in the actor
        self.obs_pointer = self.obs_norm(obs)

        # NOTE: hidden goes through LSTM, then to the value (critic head)
        return self.critic_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.actor_mlp(self.obs_pointer)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            # mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            # self.mean_bound_loss = mean_violation.mean()
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: hidden from LSTM goes to the critic head
        value = self.value(hidden)
        return probs, value


# NOTE: 13.5M params, Worked for simple motions, but not capable for many, complex motions
class LSTMActorPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.SiLU(),
        )

        self.mu = nn.Sequential(
            nn.SiLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )

        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            # nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            # nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the obs to use in the critic
        self.obs_pointer = self.obs_norm(obs)
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            # mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            # self.mean_bound_loss = mean_violation.mean()
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value


class RunningNorm(nn.Module):
    def __init__(self, shape: int, epsilon=1e-5, clip=10.0):
        super().__init__()

        self.register_buffer("running_mean", torch.zeros((1, shape), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((1, shape), dtype=torch.float32))
        self.register_buffer("count", torch.ones(1, dtype=torch.float32))
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        return torch.clamp(
            (x - self.running_mean.expand_as(x)) / torch.sqrt(self.running_var.expand_as(x) + self.epsilon),
            -self.clip,
            self.clip,
        )

    @torch.jit.ignore
    def update(self, x):
        # NOTE: Separated update from forward to compile the policy
        # update() must be called to update the running mean and var
        with torch.no_grad():
            x = x.float()
            assert x.dim() == 2, "x must be 2D"
            mean = x.mean(0, keepdim=True)
            var = x.var(0, unbiased=False, keepdim=True)
            weight = 1 / self.count
            self.running_mean = self.running_mean * (1 - weight) + mean * weight
            self.running_var = self.running_var * (1 - weight) + var * weight
            self.count += 1

    # NOTE: below are needed to torch.save() the model
    @torch.jit.ignore
    def __getstate__(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    @torch.jit.ignore
    def __setstate__(self, state):
        self.running_mean = state["running_mean"]
        self.running_var = state["running_var"]
        self.count = state["count"]
        self.epsilon = state["epsilon"]
        self.clip = state["clip"]

from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
from .diffusion_policy import DiffusionPolicy
from .network import FeedForwardNN


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
    
    def compute_fpo_loss(self, obs, actions, advantages, old_values, returns, config):
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
        # clip_coef = 0.2  # Can be made configurable
        surr1 = rho_s * advantages
        surr2 = torch.clamp(rho_s, 1 - config.clip_coef, 1 + config.clip_coef) * advantages
        policy_loss = -(torch.min(surr1, surr2)).mean()
        
        newvalue = self.critic_mlp(obs).squeeze()
        # Value loss
        if config.clip_vloss:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = val + torch.clamp(
                            newvalue - val,
                            -config.vf_clip_coef,
                            config.vf_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = ((newvalue - ret) ** 2).mean()
        
        # Clear stored action info
        self.stored_action_info = None
        
        return {
            'policy_loss': policy_loss,
            'value_loss': v_loss,
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
