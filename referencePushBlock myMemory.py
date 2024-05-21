# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
# torch.set_printoptions(linewidth=100, precision=4, sci_mode=False, threshold=2000)
import matplotlib.pyplot as plt
from utils import *

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    capture_video: bool = False
    env_id: str = "LunarLander-v2"
    total_timesteps: int = 10000
    buffer_size: int = int(1e4)
    gamma: float = 0.99
    tau: float = 1.0
    batch_size: int = 3
    learning_starts: int = 2e2
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    update_frequency: int = 4
    target_network_frequency: int = 1000
    alpha: float = 0.2
    autotune: bool = True
    target_entropy_scale: float = 0.89


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(210, 512))
        self.fc2 = layer_init(nn.Linear(512, 256))
        self.fc3 = layer_init(nn.Linear(256, 128))
        self.fc_q = layer_init(nn.Linear(128, 7))

    def forward(self, x):
        obs1D, _ = processObservations(x)
        x = F.relu(self.fc1(obs1D))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_vals = self.fc_q(x)
        return q_vals.view(-1, 7)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(210, 512))
        self.fc2 = layer_init(nn.Linear(512, 256))
        self.fc3 = layer_init(nn.Linear(256, 128))
        self.fc_logits = layer_init(nn.Linear(128, 7))


    def forward(self, x):
        obs1D, _ = processObservations(x)
        x = F.relu(self.fc1(obs1D))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x, withLogProbs=True):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        if withLogProbs:
            action_probs = policy_dist.probs
            log_prob = F.log_softmax(logits, dim=-1)
        else:
            log_prob, action_probs = None, None
        # print(f"finalLogprobs:\n{log_prob} of shape {log_prob.shape}")
        # print(f"finalProbs:\n{action_probs} of shape {action_probs.shape}")
        return action, log_prob, action_probs


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = UnityInterface("Builds\\Windows\\PushBlock\\UnityEnvironment", seed=args.seed)   # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step
    behaviorNames = env.getBehaviorNames()

    actor = Actor().to(device)
    qf1 = SoftQNetwork().to(device)
    qf2 = SoftQNetwork().to(device)
    qf1_target = SoftQNetwork().to(device)
    qf2_target = SoftQNetwork().to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(7))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
        # print(f"target entropy: {target_entropy}")
    else:
        alpha = args.alpha

    totalAgentsCounts = 0
    for behavior in behaviorNames:
        totalAgentsCounts += (env.getSpecs(behavior)["AgentsCount"])
    observationBuffer = [None] * totalAgentsCounts
    actionsBuffer = [None] * totalAgentsCounts
    rewards = [0] * totalAgentsCounts

    rb = Memory(5000)
    start_time = time.time()
    finalRewards, qnetsLosses, actorLosses, alphaLosses, alphas, QEvaluations, logProbs = [], [], [], [], [], [], []
    for global_step in range(args.total_timesteps):
        for behavior in behaviorNames:
            decisionSteps, terminalSteps = env.getSteps(behavior)
            observationsThatNeedAction = []
            for agent in decisionSteps:

                observation = decisionSteps[agent].obs
                observationsThatNeedAction.append(observation)
                reward = decisionSteps[agent].reward
                lastObservation = observationBuffer[agent]
                lastActionDiscrete = actionsBuffer[agent]
                if lastObservation != None and lastActionDiscrete != None:
                    rb.push(lastObservation, None, lastActionDiscrete, reward, False, observation)
                observationBuffer[agent] = observation
                rewards[agent] += reward
                
            for agent in terminalSteps:
                observation = terminalSteps[agent].obs
                reward = terminalSteps[agent].reward
                lastObservation = observationBuffer[agent]
                lastActionDiscrete = actionsBuffer[agent]
                # Technically could skip the action None check. If lastObs exist, action does too
                if lastObservation != None and lastActionDiscrete != None:
                    rb.push(lastObservation, None, lastActionDiscrete, reward, True, observation)
                    observationBuffer[agent] = None
                    actionsBuffer[agent] = None
                    actionsBuffer[agent] = None
                    # Save rewards only if we made an action before, otherwise the initial state was terminated state
                    rewards[agent] += reward
                    if rewards[agent] > 4:
                        print(f"Final reward: {rewards[agent]:>.2f}")
                finalRewards.append(rewards[agent])
                rewards[agent] = 0

            
            behaviorActionsForThisStep = {}
            specs = env.getSpecs(behavior)
            nrOfContinuousActions = specs["ContinuousActions"] # TODO: Substitute it with actors[behavior].usingContinuousActions
            nrOfDiscreteActions = len(specs["DiscreteActions"])
            behaviorActionsForThisStep["continuous"] = torch.zeros((len(decisionSteps), nrOfContinuousActions), requires_grad=False, dtype=torch.float32, device=device)
            behaviorActionsForThisStep["discrete"] = torch.zeros((len(decisionSteps), nrOfDiscreteActions), requires_grad=False, dtype=torch.int32, device=device)
            # print(f"Allocated discrete action buffer of shape {behaviorActionsForThisStep['discrete'].shape}")
            # Batched pass to get actions  
            if len(observationsThatNeedAction) > 0:
                # if actor.usingContinuousActions:
                #     behaviorActionsForThisStep["continuous"], _ = actor.getContinuousAction(observationsThatNeedAction, withLogProbs=False)
                # if actor.usingDiscreteActions:
                behaviorActionsForThisStep["discrete"], _, _ = actor.get_action((observationsThatNeedAction), withLogProbs=False)

            # Transcribe the actions to buffer
                for j, agent in enumerate(decisionSteps):
                    # if nrOfContinuousActions > 0:
                    #     actionsBuffer[agent] =  behaviorActionsForThisStep['coninuous'][j]
                    # if nrOfDiscreteActions > 0:
                    actionsBuffer[agent] = behaviorActionsForThisStep['discrete'][j]

            # print(f"Setting Continuous actions: {behaviorActionsForThisStep}, Discrete actions: {behaviorActionsForThisStep}")
            # print(f"Discrete action buffer shape after detachcpunumpy: {behaviorActionsForThisStep['discrete'].detach().cpu().numpy().shape}")
            
            # FIXME: I cannot be copying the original buffer to cpu. REWORK THIS
            behaviorActionsForThisStep['discrete'] = behaviorActionsForThisStep['discrete'].detach().cpu().numpy()
            if behaviorActionsForThisStep['discrete'].ndim == 1:
                behaviorActionsForThisStep['discrete'] = np.expand_dims(behaviorActionsForThisStep['discrete'], -1)
            env.setActions(behavior, behaviorActionsForThisStep['continuous'].detach().cpu().numpy(), behaviorActionsForThisStep['discrete'])
        env.step()
        
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                observationsBatch       =               torch.tensor(np.stack(data.observations), device=device, dtype=torch.float32)
                actionsDiscreteBatch    =               torch.tensor(np.stack(data.actionsDiscrete), device=device, dtype=torch.long)
                rewardsBatch            =               torch.tensor(np.stack(data.rewards), device=device, dtype=torch.float32)
                isThereNextStepBatch    =               torch.logical_not(torch.tensor(np.stack(data.dones), device=device, dtype=torch.float32))
                nextObservationsBatch   =               torch.tensor(np.stack(data.nextObservations), device=device, dtype=torch.float32)
                # print(f"Sampled obs:\n{observationsBatch} of shape {observationsBatch.shape},\nSampled actions:\n{actionsDiscreteBatch} of shape {actionsDiscreteBatch.shape}")
                # CRITIC training
                with torch.no_grad():
                    # print(f"CRITIC OPTIM")
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(nextObservationsBatch)
                    qf1_next_target = qf1_target(nextObservationsBatch)
                    qf2_next_target = qf2_target(nextObservationsBatch)
                    # print(f"output of critic is of shape {qf1_next_target.shape}")
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # print(f"next_state_action_probs:\n{next_state_action_probs} of shape {next_state_action_probs.shape}")
                    # print(f"next_state_log_probs\n{next_state_log_pi} of shape {next_state_log_pi.shape}")
                    # print(f"min evaluation in critic optim\n{torch.min(qf1_next_target, qf2_next_target)} of shape {torch.min(qf1_next_target, qf2_next_target).shape}")

                    # adapt Q-target for discrete Q-function
                    # print(f"minQNextTarget before sum:\n{min_qf_next_target} of shape {min_qf_next_target.shape}")
                    min_qf_next_target = min_qf_next_target.sum(axis=tuple(range(1, min_qf_next_target.ndim)))
                    # print(f"minQNextTarget after sum:\n{min_qf_next_target} of shape {min_qf_next_target.shape}")
                    # print(f"rewardsBatchs of shape {rewardsBatch.shape}, nextStepBatch shape {isThereNextStepBatch.shape}, minqfnexttarget shape {min_qf_next_target.shape}")
                    next_q_value = rewardsBatch.flatten() + isThereNextStepBatch.flatten() * args.gamma * (min_qf_next_target)
                    next_q_value = rewardsBatch.flatten() + isThereNextStepBatch.flatten() * args.gamma * (min_qf_next_target)
                    # print(f"nextQValue:\n{next_q_value} of shape {next_q_value.shape}")

                # use Q-values only for the taken actions
                qf1_values = qf1(observationsBatch)
                qf2_values = qf2(observationsBatch)
                # print(f"output of QFunction1ActionValues before gathering\n{qf1_values} of shape {qf1_values.shape}")
                # print(f"qf1values of shape {qf1_values.shape} gathering with actions of shape {actionsDiscreteBatch.shape}")
                qf1_a_values = qf1_values.gather(1, actionsDiscreteBatch).view(-1)
                qf2_a_values = qf2_values.gather(1, actionsDiscreteBatch).view(-1)
                # print(f"output of QFunction1ActionValues after gathering\n{qf1_a_values} of shape {qf1_a_values.shape}")
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss
                # print(f"mse_loss of QFunction1ActionValues and nextqvalue difference is the critic loss\n\n")

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(observationsBatch)
                # print(f"ACTOR OPTIM")
                with torch.no_grad():
                    qf1_values = qf1(observationsBatch)
                    qf2_values = qf2(observationsBatch)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()
                # print(f"state probs:\n{action_probs} of shape {action_probs.shape}")
                # print(f"state logprobs:\n{log_pi} of shape {log_pi.shape}")
                # print(f"min evaluation in actor optim:\n{min_qf_values} of shape {min_qf_values.shape}")
                # print(f"Actor loss is probs*(alpha*logprobs - minqeval).mean()\n\n")
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()
                    # print(f"ALPHA OPTIM")
                    # print(f"probsD:\n{action_probs} of shape {action_probs.shape}")
                    # print(f"logprobsD:\n{log_pi} of shape {log_pi.shape}")
                    # print(f"alpha:\n{log_alpha.exp()} of shape {log_alpha.exp().shape}")
                    # print(f"target entropy:\n{target_entropy} of shape {target_entropy.shape}")
                    # print(f"alpha loss is: (probsD(-alpha*logprobs + target)).mean()\n\n")

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                qnetsLosses.append(qf_loss)
                actorLosses.append(actor_loss)
                alphaLosses.append(alpha_loss)
                alphas.append(alpha)
                QEvaluations.append(min_qf_values.mean())
                logProbs.append(log_pi.mean())

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 200 == 0:
                print(f"Step {global_step}, Actor loss: {actor_loss:>8.4f}, QF loss: {qf_loss:>8.4f}")

    env.close()

    averagingNr = 10
    beginning = 0
    dif_qnets = (len(qnetsLosses) - beginning) % averagingNr

    # Style
    plt.style.use('seaborn-v0_8-bright')

    # Creating a high-resolution figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), dpi=200)

    # First plot
    # ax1.plot(torch.tensor(qnetsLosses[beginning:-dif_qnets]).view(-1, averagingNr).mean(-1), label="critic loss")
    ax1.plot(torch.tensor(actorLosses[beginning:-dif_qnets]).view(-1, averagingNr).mean(-1), label="actor loss")
    ax1.plot(torch.tensor(alphaLosses[beginning:-dif_qnets]).view(-1, averagingNr).mean(-1), label="alpha loss")
    ax1.plot(torch.tensor(alphas[beginning:-dif_qnets]).view(-1, averagingNr).mean(-1), label="alpha")
    ax1.plot(torch.tensor(QEvaluations[beginning:-dif_qnets]).view(-1, averagingNr).mean(-1), label="batch evaluation")
    ax1.plot(torch.tensor(logProbs[beginning:-dif_qnets]).view(-1, averagingNr).mean(-1), label="logprobs")

    ax1.legend()
    ax1.set_title("SAC multidiscrete CleanRL LunarLander-V2")
    ax1.set_xlabel(f"Iterations / {averagingNr}")
    ax1.set_ylabel("Value")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Second plot
    averageNr = 5
    dif_rewards = len(finalRewards) % averageNr
    ax2.plot(torch.tensor(finalRewards[:-dif_rewards]).view(-1, averageNr).mean(-1))
    ax2.set_title("Final Rewards")
    ax2.set_xlabel(f"Iterations / {averageNr}")
    ax2.set_ylabel("Reward Value")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()