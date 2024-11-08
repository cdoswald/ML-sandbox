import os
import random
import time

import numpy as np

import gymnasium as gym
from gymnasium.experimental.wrappers.rendering import RecordVideoV0 as RecordVideo
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from config import Args
from models import Actor, SoftQNetwork
from utils import save_video

# Modified version of CleanRL SAC implementation
def train_SAC(args):
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = gym.make(args.env_id, render_mode=None)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(env.action_space.high[0])

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    episode_reward = 0
    episode_length = 0
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = env.action_space.sample()
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy() if not (terminations or truncations) else obs
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break
        episode_reward += rewards
        episode_length += 1
        if "episode" in infos:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

        # Reset environment on termination or truncation
        if terminations or truncations:
            obs, _ = env.reset(seed=args.seed)
            # print(f'Global step: {global_step}; episode reward: {episode_reward}; length = {episode_length}')
            episode_reward = 0
            episode_length = 0
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    # Save models
    torch.save(actor.state_dict(), f"runs/{run_name}/actor.pth")
    torch.save(qf1.state_dict(), f"runs/{run_name}/qf1.pth")
    torch.save(qf2.state_dict(), f"runs/{run_name}/qf2.pth")

    # Clean up
    env.close()
    writer.close()


def eval_SAC(args, actor_path):
    """Evaluate trained SAC agent"""
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = gym.make(args.env_id, render_mode="rgb_array")
    videos_dir = f"videos/{run_name}"
    os.makedirs(videos_dir, exist_ok=True)
    # env = RecordVideo(
    #     env,
    #     f"videos/{run_name}",
    #     episode_trigger=lambda episode_id: episode_id % 20 == 0,
    # )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(args.seed)

    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    max_action = float(env.action_space.high[0])

    # Load model
    actor = Actor(env).to(device)
    actor.load_state_dict(torch.load(actor_path, weights_only=True))

    # TRY NOT TO MODIFY: start the game
    episode_reward = 0
    episode_length = 0
    episode_frames = []
    episode_idx = 0
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        _, _, mean_action = actor.get_action(torch.Tensor(obs).to(device))
        actions = mean_action.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # Render and record frame (VideoRecorder not working correctly)
        episode_frames.append(env.render().astype(np.uint8))

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        episode_reward += rewards
        episode_length += 1
        if "episode" in infos:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

        # Reset environment on termination or truncation
        if terminations or truncations:
            obs, _ = env.reset()
            # print(f'Global step: {global_step}; episode reward: {episode_reward}; length = {episode_length}')
            episode_reward = 0
            episode_length = 0
            # Save episode video
            if episode_idx < 10:
                save_path = os.path.join(videos_dir, f"episode_{episode_idx}.mp4")
                save_video(episode_frames, save_path)
            # Increment episode index and reset frames list
            episode_idx += 1
            episode_frames = []
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

    # Clean up
    env.close()
    writer.close()


if __name__ == "__main__":

    # Specify MuJoCo tasks
    env_ids = [
        "Ant-v4",
        "HalfCheetah-v4",
        "Hopper-v4",
        "Humanoid-v4",
        "Walker2d-v4",
    ]

    for env_id in env_ids:
        print(gym.spec(env_id))
        env = gym.make(env_id, render_mode="rgb_array")
        print(env.observation_space)
        print(env.action_space)

        # Train agent
        train_args = Args()
        train_args.exp_name = "SAC_baseline_train"
        train_args.env_id = env_id
        train_args.seed = 42
        train_args.total_timesteps = 100000
        train_args.buffer_size = 100000

        train_SAC(train_args)

        # Evaluate agent
        eval_args = Args()
        eval_args.exp_name = "SAC_baseline_eval"
        eval_args.env_id = env_id
        eval_args.seed = 42
        eval_args.total_timesteps = 10000

        actor_path = f"runs/{train_args.env_id}__{train_args.exp_name}__{train_args.seed}/actor.pth"
        eval_SAC(eval_args, actor_path=actor_path)