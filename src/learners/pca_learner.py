import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.pca import PCACritic
from components.action_selectors import multinomial_entropy
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam

class PCALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.log_stats_t_agent = -self.args.learner_log_interval - 1

        self.critic = PCACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(self.mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr)

        self.entropy_coef = args.entropy_coef
        self.each_ig_step_num = args.each_ig_step_num
        self.cal_order_len = args.cal_order_len
        self.init_order()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        actions = batch["actions"][:, :-1]
        onehot_actions = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :]
        adj = batch["adj_mat"][:, :-1]

        entropy_mask = copy.deepcopy(mask).view(-1)
        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        mac_prob_out = []
        mac_out_entropy = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            agent_probs = th.nn.functional.softmax(agent_outs, dim=-1)
            agent_entropy = multinomial_entropy(agent_outs).mean(dim=-1, keepdim=True)
            mac_prob_out.append(agent_probs)
            mac_out_entropy.append(agent_entropy)
        mac_prob_out = th.stack(mac_prob_out, dim=1)
        mac_out_entropy = th.stack(mac_out_entropy, dim=1)
        mac_out_entropy = mac_out_entropy[:, :-1].reshape(-1)

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_prob_out[avail_actions == 0] = 0
        mac_prob_out = mac_prob_out/mac_prob_out.sum(dim=-1, keepdim=True)
        mac_prob_out[avail_actions == 0] = 0

        mac_prob_out = mac_prob_out[:, :-1]


        # Calculate policy grad with mask
        pi = mac_prob_out.reshape(-1, self.n_actions)
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        # Calculate adv with F2 - F1

        if not self.args.timestep_change_order:
            adv = [None] * self.args.n_agents
            tmp = mac_prob_out.clone()
            F1 = self.critic(mac_prob_out, batch["state"][:, :-1])
            for i in self.order:
                tmp = tmp.clone()
                tmp[:,:,i,:] = onehot_actions[:,:,i,:]
                F2 = self.critic(tmp, batch["state"][:, :-1])
                adv[i] = (F2-F1).clone().detach()
                F1 = F2.clone()
            adv = th.cat(adv, dim=-1).view(-1)
        else:
            adv = th.zeros_like(log_pi_taken).view(bs, max_t-1, self.n_agents)
            tmp = mac_prob_out.clone()
            for i in range(self.n_agents):
                tmp1 = tmp.clone()
                neighbor = adj[:, :, i].clone().unsqueeze(-1).repeat(1, 1, 1, self.n_actions)
                onehot_action_adj = th.mul(onehot_actions, neighbor)
                non_zero_index = (onehot_action_adj != 0).any(dim=3)
                tmp1[non_zero_index] = onehot_action_adj[non_zero_index]
                if self.args.critic_q:
                    F1 = self.critic(tmp1, batch["state"][:, :-1])
                else:
                    F1, _ = self.critic(tmp1, batch["state"][:, :-1])
                tmp1[:, :, i:i+1, :] = onehot_actions[:, :, i:i+1, :]
                if self.args.critic_q:
                    F2 = self.critic(tmp1, batch["state"][:, :-1])
                else:
                    F2, _ = self.critic(tmp1, batch["state"][:, :-1])
                adv_tmp = (F2-F1).clone().detach()
                adv[:, :, i:i+1] = adv_tmp
            adv = adv.view(-1)

        mix_loss = log_pi_taken * adv
        mix_loss = (mix_loss * mask).sum() / mask.sum()

        # Adaptive Entropy Regularization
        entropy_loss = (mac_out_entropy * entropy_mask).sum() / entropy_mask.sum()
        entropy_ratio = self.entropy_coef / entropy_loss.item()

        mix_loss = - mix_loss - entropy_ratio * entropy_loss

        # Optimise agents
        self.agent_optimiser.zero_grad()
        mix_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if t_env - self.log_stats_t_agent >= self.args.learner_log_interval:
            self.logger.log_stat("mix_loss", mix_loss.item(), t_env)
            self.logger.log_stat("entropy", entropy_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.log_stats_t_agent = t_env


    def train_critic_td(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"]
        actions = batch["actions_onehot"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, return_logits=True)
            agent_probs = th.nn.functional.softmax(agent_outs, dim=-1)
            mac_out.append(agent_probs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Optimise critic
        if self.args.critic_q:
            target_q_vals = self.target_critic(actions, batch["state"])[:, :]
        else:
            target_Fu, target_b_tmp = self.target_critic(actions, batch["state"])
            target_Fpi, target_b = self.target_critic(mac_out, batch["state"])
            target_q_vals = target_Fu - target_Fpi + target_b

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_q_vals, self.n_agents, self.args.gamma, self.args.td_lambda)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_t_mean": [],
        }

        mask = mask[:, :-1]

        if self.args.critic_q:
            q_t = self.critic(actions[:, :-1], batch["state"][:, :-1])
        else:
            Fu, b_tmp = self.critic(actions[:, :-1], batch["state"][:, :-1])
            Fpi, b = self.target_critic(mac_out[:, :-1], batch["state"][:, :-1])
            q_t = Fu - Fpi + b

        td_error = (q_t - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_t_mean"].append((q_t * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((targets * mask).sum().item() / mask_elems)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(running_log["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_t_mean", "target_mean"]:
                self.logger.log_stat(key, sum(running_log[key])/ts_logged, t_env)
            self.log_stats_t = t_env

        # Update target critic
        if (self.critic_training_steps - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = self.critic_training_steps

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
    
    def init_order(self):
        self.order = [_ for _ in range(self.n_agents)]



