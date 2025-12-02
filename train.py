import argparse
import torch as tc
from functools import partial

from rl2.envs.mdp_env import MDPEnv
from rl2.agents.preprocessing.tabular import MDPPreprocessing

from rl2.agents.architectures.snail import SNAIL

from rl2.agents.heads.policy_heads import LinearPolicyHead
from rl2.agents.heads.value_heads import LinearValueHead
from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet
from rl2.algos.ppo import training_loop

from rl2.utils.checkpoint_util import maybe_load_checkpoint, save_checkpoint
from rl2.utils.comm_util import get_comm, sync_state
from rl2.utils.constants import ROOT_RANK
from rl2.utils.optim_util import get_weight_decay_param_groups

def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_states", type=int, default=10)
    parser.add_argument("--num_actions", type=int, default=5)
    parser.add_argument("--max_episode_len", type=int, default=10,
                        help="Timesteps before automatic episode reset.")
    parser.add_argument("--meta_episode_len", type=int, default=100,
                        help="Timesteps per meta-episode.")

    parser.add_argument("--num_features", type=int, default=256)

    parser.add_argument("--max_pol_iters", type=int, default=12000)
    parser.add_argument("--meta_episodes_per_policy_update", type=int, default=-1,
                        help="If -1, quantity is determined using a formula")
    parser.add_argument("--meta_episodes_per_learner_batch", type=int, default=60)
    parser.add_argument("--ppo_opt_epochs", type=int, default=8)
    parser.add_argument("--ppo_clip_param", type=float, default=0.10)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.01)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.3)
    parser.add_argument("--standardize_advs", type=int, choices=[0,1], default=0)
    parser.add_argument("--adam_lr", type=float, default=2e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-5)
    parser.add_argument("--adam_wd", type=float, default=0.01)
    return parser

def create_env(num_states, num_actions, max_episode_len):
    return MDPEnv(
        num_states=num_states,
        num_actions=num_actions,
        max_episode_length=max_episode_len)

def create_preprocessing(num_states, num_actions):
    return MDPPreprocessing(
        num_states=num_states,
        num_actions=num_actions)

def create_architecture(input_dim, num_features, context_size):
    return SNAIL(
        input_dim=input_dim,
        feature_dim=num_features,
        context_size=context_size,
        use_ln=True)

def create_head(head_type, num_features, num_actions):
    if head_type == 'policy':
        return LinearPolicyHead(
            num_features=num_features,
            num_actions=num_actions)
    if head_type == 'value':
        return LinearValueHead(
            num_features=num_features)
    raise NotImplementedError

def create_net(
        net_type, num_states, num_actions,
        num_features, context_size
):
    preprocessing = create_preprocessing(
        num_states=num_states,
        num_actions=num_actions)
    architecture = create_architecture(
        input_dim=preprocessing.output_dim,
        num_features=num_features,
        context_size=context_size)
    head = create_head(
        head_type=net_type,
        num_features=architecture.output_dim,
        num_actions=num_actions)

    if net_type == 'policy':
        return StatefulPolicyNet(
            architecture=architecture,
            preprocessing=preprocessing,
            policy_head=head)
    if net_type == 'value':
        return StatefulValueNet(
            architecture=architecture,
            preprocessing=preprocessing,
            value_head=head)
    raise NotImplementedError

def main():
    args = create_argparser().parse_args()
    comm = get_comm()

    env = create_env(
        num_states=args.num_states,
        num_actions=args.num_actions,
        max_episode_len=args.max_episode_len)

    policy_net = create_net(
        net_type='policy',
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_features=args.num_features,
        context_size=args.meta_episode_len)

    value_net = create_net(
        net_type='value',
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_features=args.num_features,
        context_size=args.meta_episode_len)

    policy_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(policy_net, args.adam_wd),
        lr=args.adam_lr,
        eps=args.adam_eps)
    value_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(value_net, args.adam_wd),
        lr=args.adam_lr,
        eps=args.adam_eps)

    policy_scheduler = None
    value_scheduler = None

    pol_iters_so_far = 0
    if comm.Get_rank() == ROOT_RANK:
        a = maybe_load_checkpoint(
            checkpoint_dir='./policy_checkpoints',
            model_name=f"rl2/policy_net",
            model=policy_net,
            optimizer=policy_optimizer,
            scheduler=policy_scheduler,
            steps=None)

        b = maybe_load_checkpoint(
            checkpoint_dir='./value_checkpoints',
            model_name=f"rl2/value_net",
            model=value_net,
            optimizer=value_optimizer,
            scheduler=value_scheduler,
            steps=None)

        if a != b:
            raise RuntimeError(
                "Policy and value iterates not aligned in latest checkpoint!")
        pol_iters_so_far = a

    pol_iters_so_far = comm.bcast(pol_iters_so_far, root=ROOT_RANK)
    sync_state(
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler,
        comm=comm,
        root=ROOT_RANK)
    sync_state(
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler,
        comm=comm,
        root=ROOT_RANK)

    policy_checkpoint_fn = partial(
        save_checkpoint,
        checkpoint_dir='./policy_checkpoints',
        model_name=f"rl2/policy_net",
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler)

    value_checkpoint_fn = partial(
        save_checkpoint,
        checkpoint_dir='./value_checkpoints',
        model_name=f"rl2/value_net",
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler)

    if args.meta_episodes_per_policy_update == -1:
        numer = 240000
        denom = comm.Get_size() * args.meta_episode_len
        meta_episodes_per_policy_update = numer // denom
    else:
        meta_episodes_per_policy_update = args.meta_episodes_per_policy_update

    training_loop(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_scheduler=policy_scheduler,
        value_scheduler=value_scheduler,
        meta_episodes_per_policy_update=meta_episodes_per_policy_update,
        meta_episodes_per_learner_batch=args.meta_episodes_per_learner_batch,
        meta_episode_len=args.meta_episode_len,
        ppo_opt_epochs=args.ppo_opt_epochs,
        ppo_clip_param=args.ppo_clip_param,
        ppo_ent_coef=args.ppo_ent_coef,
        discount_gamma=args.discount_gamma,
        gae_lambda=args.gae_lambda,
        standardize_advs=bool(args.standardize_advs),
        max_pol_iters=args.max_pol_iters,
        pol_iters_so_far=pol_iters_so_far,
        policy_checkpoint_fn=policy_checkpoint_fn,
        value_checkpoint_fn=value_checkpoint_fn,
        comm=comm)

if __name__ == '__main__':
    main()
