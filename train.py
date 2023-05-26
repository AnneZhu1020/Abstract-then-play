import subprocess
import time
import os

import numpy as np
import torch
import logger
import argparse
import jericho
import logging
import json
from os.path import basename, dirname
from drrn import *
from model import RMS
from env import JerichoEnv
# from env2 import JerichoEnv2
from jericho.util import clean
from copy import deepcopy
from vec_env import VecEnv
from torch import from_numpy
from torch.nn.functional import log_softmax

import os
os.environ["WANDB_MODE"] = "offline"

# from lm import *

logging.getLogger().setLevel(logging.CRITICAL)


def configure_logger(log_dir, wandb):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    type_strs = ['json', 'stdout', 'csv']
    if wandb and log_dir != 'logs': type_strs += ['wandb']
    tb = logger.Logger(log_dir, [logger.make_output_format(type_str, log_dir) for type_str in type_strs])
    global log
    log = logger.log


def evaluate(agent, env, nb_episodes=1, lm=False, lm_model=None, n_skills=None, continuous_skill=False):
    with torch.no_grad():
        total_score = 0
        for ep in range(nb_episodes):
            log("Starting evaluation episode {}".format(ep))
            score = evaluate_episode(agent, env, lm, lm_model, n_skills, continuous_skill)
            log("Evaluation episode {} ended with score {}\n\n".format(ep, score))
            total_score += score
        avg_score = total_score / nb_episodes
        return avg_score


def evaluate_episode(agent, env, lm=None, lm_model=None, n_skills=None, continuous_skill=None):
    step = 0
    done = False
    ob, info = env.reset()
    if lm:
        state = agent.build_lm_state(lm_model, [ob], [info])
    else:
        state = agent.build_state(ob, info)
    if n_skills:
        if continuous_skill:
            scale_factor = 1

            # just random generate
            context = np.random.random(n_skills)
            # sample normal distribution
            # context = np.random.normal(0, 1, size=n_skills)

            p_z = context * 2 * scale_factor - scale_factor
        else:
            p_z = [np.full(n_skills, 1 / n_skills)]

    log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
    step = 0
    while not done:
        step += 1
        if n_skills:
            if continuous_skill:
                zs = torch.FloatTensor([p_z]).to(device)
            else:
                zs = [np.random.choice(n_skills, p=p) for p in p_z]
                zs = torch.tensor(zs)
        else:
            zs = None

        valid_acts = info['valid']
        if lm:
            valid_ids = [lm_model.act2ids(a) for a in info['valid']]
        else:
            valid_ids = agent.encode(valid_acts)
        _, action_idx, action_values = agent.act([state], [valid_ids], sample=False, z=zs,
                                                 continuous_skill=continuous_skill)
        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        log('Action{}: {}, Q-Value {:.2f}'.format(step, action_str, action_values[action_idx].item()))
        s = ''
        for idx, (act, val) in enumerate(sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True), 1):
            s += "{}){:.2f} {} ".format(idx, val.item(), act)
        log('Q-Values: {}'.format(s))
        ob, rew, done, info = env.step(action_str)
        log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))
        step += 1
        log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
        if lm:
            state = agent.build_lm_state(lm_model, [ob], [info])
        else:
            state = agent.build_state(ob, info)
    return info['score']


def pairwise_dist(source, target):
    dists = torch.sum(source ** 2, dim=1, keepdim=True) + torch.sum(target ** 2, dim=1) - 2 * source @ target.T
    return torch.sqrt(dists)


def train(agent, eval_env, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, r_for, lm=False,
          lm_model=None, args=None):
    start, max_score, max_reward = time.time(), 0, 0
    obs, infos = envs.reset()

    if lm:
        states = agent.build_lm_state(lm_model, obs, infos)
        valid_ids = [[lm_model.act2ids(a) for a in info['valid']] for info in infos]
    else:
        if args.multiprocess:
            states = agent.build_states(obs, infos)
            valid_ids = [agent.encode(info['valid']) for info in infos]
        else:
            states = [agent.build_state(obs, infos)]
            valid_ids = [agent.encode(infos['valid'])]
            infos = [infos]

    if args.n_skills:
        if args.continuous_skill:
            scale_factor = 1
            # just random generate
            context = np.random.random(args.n_skills)
            # sample normal distribution
            # context = np.random.normal(0, 1, size=args.n_skills)
            context = np.tile(context, args.num_envs).reshape(args.num_envs,
                                                              args.n_skills)  # size([batch_size, n_skills])
            p_z = context * 2 * scale_factor - scale_factor  # scale to [-a, a)

            rms = RMS()
        else:
            p_z = np.full(args.n_skills, 1 / args.n_skills)
            p_z = np.tile(p_z, args.num_envs).reshape(args.num_envs, args.n_skills)  # size([batch_size, n_skills])
        p_z = torch.FloatTensor(p_z).to(device)

    transitions = [[] for info in infos]

    for step in range(1, max_steps + 1):
        if args.n_skills:
            if args.continuous_skill:
                zs = p_z
            else:
                zs = [torch.multinomial(input=p, num_samples=1, replacement=True) for p in p_z]
                zs = torch.stack(zs)
        else:
            zs = None
        action_ids, action_idxs, action_values = agent.act(states, valid_ids, sample=True,
                                                           eps=0.05 ** (step / max_steps), lm=lm,
                                                           lm_model=lm_model, z=zs,
                                                           continuous_skill=args.continuous_skill, step=step)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

        # log envs[0] 
        examples = [(action, value) for action, value in zip(infos[0]['valid'], action_values[0].tolist())]
        examples = sorted(examples, key=lambda x: -x[1])
        log('State  {}: {}'.format(step, clean(obs[0] + infos[0]['inv'] + infos[0]['look'])))
        log('Actions{}: {}'.format(step, [action for action, _ in examples]))
        log('Qvalues{}: {}'.format(step, [round(value, 2) for _, value in examples]))
        log('>> Action{}: {}'.format(step, action_strs[0]))

        # step
        if not args.multiprocess:
            action_strs = action_strs[0]

        obs, rewards, dones, infos = envs.step(action_strs)

        if lm:
            next_states = agent.build_lm_state(lm_model, obs, infos, prev_obs=obs, prev_acts=action_strs)
            next_valids = [[lm_model.act2ids(a) for a in info['valid']] for info in infos]
        else:
            if args.multiprocess:
                next_states = agent.build_states(obs, infos)
                next_valids = [agent.encode(info['valid']) for info in infos]
            else:
                next_states = [agent.build_state(obs, infos)]
                next_valids = [agent.encode(infos['valid'])]
                obs = [obs]
                rewards = [rewards]
                dones = [dones]
                infos = [infos]

        if args.r_for:
            reward_curiosity, _ = agent.network.inv_loss_decode(states, next_states, [[a] for a in action_ids],
                                                                hat=True, reduction='none')
            rewards = rewards + reward_curiosity.detach().numpy() * -1.0
            tb.logkv_mean('Curiosity', reward_curiosity.mean().item())

        if args.w_skills > 0:
            state_out = agent.network.state_rep(states)
            logits = agent.network.discriminator(state_out)  # size(8, 10)
            if args.continuous_skill:
                r_intrinsic = - F.mse_loss(logits, p_z, reduction='sum')
            else:
                logq_z_ns = log_softmax(logits, dim=-1)  # size(8, 10)
                p_z = p_z.gather(-1, zs)  # size([8, 1])
                r_intrinsic = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)  # size(8)

            rewards = rewards + r_intrinsic.squeeze(-1).cpu().detach().numpy() * args.w_skills
            tb.logkv_mean('Intrinsic', r_intrinsic.mean().item())

        if args.r_cic:
            # compute intrinsic reward
            state_out = agent.network.state_rep(states)  # size([8, 128])
            next_state_out = agent.network.state_rep(next_states)   # size([8, 128])
            _, act_out = agent.network.act_rep([[a] for a in action_ids])   # size([8, 128])
            cic_reward = agent.cic_loss(state_out, next_state_out, zs, act_out, reduction=False)

            # sim between state and next_state
            # sim_matrix = pairwise_dist(source=state_out, target=next_state_out)  # size([8, 8])
            # reward, _ = sim_matrix.topk(1, dim=1, largest=False, sorted=True)  # size([bs, k])
            # reward, _ = sim_matrix.topk(args.topk_knn, dim=1, largest=False, sorted=True)  # size([bs, k])

            knn_avg = False
            use_rms = True
            knn_clip = 0.0005

            # if not knn_avg:
                # only keep k-th nearest neighbor
                # reward = reward[:, -1]
                # reward = reward.reshape(-1, 1)  # size([bs, 1])
                # if use_rms:
                #     moving_mean, moving_std = rms(reward)
                #     reward = reward / moving_std
                # knn_clip: 0.0005
                # reward = torch.max(reward - knn_clip, torch.zeros_like(reward).to(device))  # size([bs, ])
            # else:
            #     reward = reward.reshape(-1, 1)  # size([bs * k, 1])
            #     if use_rms:
            #         moving_mean, moving_std = rms(reward)
            #         reward = reward / moving_std
            #     reward = torch.max(reward - knn_clip, torch.zeros_like(reward).to(device))
            #     reward = reward.reshape(state_out.size(0), args.topk_knn)
            #     reward = reward.mean(dim=1)  # size([bs, ])
            # cic_reward = torch.log(reward + 1.0)
            #
            rewards = rewards + cic_reward.squeeze(-1).cpu().detach().numpy() * -1.0
            tb.logkv_mean('CIC reward', cic_reward.mean().item())

        if args.r_linearizer > 0:
            state_out = agent.network.state_rep(states)  # size([8, 128])
            next_state_out = agent.network.state_rep(next_states)   # size([8, 128])
            # linearizer_reward = agent.linearizer_reward(state=state_out, skill=p_z)  # => bigger
            linearizer_reward = agent.linearizer_reward(state=next_state_out - state_out, skill=p_z)  # => bigger
            rewards = rewards + linearizer_reward.squeeze(-1).cpu().detach().numpy() * args.r_linearizer
            tb.logkv_mean('Linearizer reward', linearizer_reward.mean().item())

        if args.r_rnd > 0:
            next_state_out = agent.network.state_rep(next_states)   # size([8, 128])
            rnd_reward = agent.compute_rnd_reward(next_state_out)
            rewards = rewards + rnd_reward.squeeze(-1).cpu().detach().numpy() * -1.0
            tb.logkv_mean('RND reward', rnd_reward.mean().item())

        # for skill-based:
        for i, (ob, reward, done, info, state, next_state, z) in enumerate(
                zip(obs, rewards, dones, infos, states, next_states, zs)):
            transition = Transition(state, action_ids[i], reward, next_state, next_valids[i], done, valid_ids[i], z)

            # for i, (ob, reward, done, info, state, next_state) in enumerate(
            #         zip(obs, rewards, dones, infos, states, next_states)):
            #     transition = Transition(state, action_ids[i], reward, next_state, next_valids[i], done, valid_ids[i])

            transitions[i].append(transition)
            agent.observe(transition)

            if i == 0:
                log("Reward{}: {}, Score {}, Done {}\n".format(step, reward, info['score'], done))
            if done:
                tb.logkv_mean('EpisodeScore', info['score'])

                # obs[i], infos[i] = env.reset()
                # next_states[i] = agent.build_state(obs[i], infos[i])
                # next_valids[i] = agent.encode(infos[i]['valid'])
                if info['score'] >= max_score:  # put in alpha queue
                    if info['score'] > max_score:
                        agent.memory.clear_alpha()
                        max_score = info['score']
                    for transition in transitions[i]:
                        agent.observe(transition, is_prior=True)
                transitions[i] = []

                # if args.n_skills and args.continuous_skill:
                #     context = np.random.random(args.n_skills)
                #     context = np.tile(context, args.num_envs).reshape(args.num_envs,
                #                                                       args.n_skills)  # size([batch_size, n_skills])
                #     p_z = context * 2 * scale_factor - scale_factor  # scale to [-a, a)
                #     p_z = torch.FloatTensor(p_z).to(device)

        states, valid_ids = next_states, next_valids

        if args.n_skills and args.continuous_skill and step % args.update_skill_every == 0:
            if args.continuous_skill:
                # option1: just random generate
                context = np.random.random(args.n_skills)
                # option2: sample normal distribution
                # context = np.random.normal(0, 1, size=args.n_skills)
            else:
                context = np.full(args.n_skills, 1 / args.n_skills)

            context = np.tile(context, args.num_envs).reshape(args.num_envs,
                                                              args.n_skills)  # size([batch_size, n_skills])
            p_z = context * 2 * scale_factor - scale_factor  # scale to [-a, a)
            p_z = torch.FloatTensor(p_z).to(device)

        if step % log_freq == 0:
            tb.logkv('Step', step)
            tb.logkv("FPS", int((step * args.num_envs) / (time.time() - start)))
            tb.logkv("EpisodeScores100", envs.get_end_scores().mean())
            tb.logkv('MaxScore', max_score)
            tb.logkv('Step', step)
            # if envs[0].cache is not None:
            #     tb.logkv('#dict', len(envs[0].cache)) 
            #     tb.logkv('#locs', len(envs[0].cache['loc'])) 
            tb.dumpkvs()

        if step % update_freq == 0:
            res = agent.update()
            if res is not None:
                for k, v in res.items():
                    tb.logkv_mean(k, v)

        if (args.w_rnd or args.r_rnd) and step % args.target_update_freq == 0:
            agent.network.rnd_target.load_state_dict(agent.network.rnd_predictor.state_dict())

        if step % checkpoint_freq == 0:
            agent.save(str(step))
            # json_path = envs[0].rom_path.replace('.z5', '.json')
            # if os.path.exists(json_path):
            #     envs[0].cache.update(json.load(open(json_path)))
            # json.dump(envs[0].cache, open(json_path, 'w'))
            # agent.save_emb(str(step))

        if step % eval_freq == 0:
            eval_score = evaluate(agent, eval_env, lm=lm, lm_model=lm_model, n_skills=args.n_skills,
                                  continuous_skill=args.continuous_skill)
            tb.logkv('EvalScore', eval_score)
            tb.dumpkvs()


def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--load', default=None)  
    parser.add_argument('--spm_path', default='unigram_8k.model')
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=1000, type=int)
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--memory_alpha', default=.4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float, choices=[5, 40])
    # parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    parser.add_argument('--wandb', default=1, type=int)

    parser.add_argument('--type_inv', default='decode')
    parser.add_argument('--type_for', default='decode')
    parser.add_argument('--w_inv', default=1, type=float)
    parser.add_argument('--w_for', default=0, type=float)
    parser.add_argument('--w_act', default=0, type=float)
    parser.add_argument('--r_for', default=0, type=float)

    parser.add_argument('--nor', default=0, type=int, help='no game reward')
    parser.add_argument('--randr', default=10.0, type=float,
                        help='random game reward by objects and locations within episode, default 10')
    parser.add_argument('--perturb', default=0, type=int, help='perturb state and action')

    parser.add_argument('--hash_rep', default=0, type=int, help='hash for representation')
    parser.add_argument('--act_obs', default=0, type=int, help='action set as state representation')
    parser.add_argument('--fix_rep', default=0, type=int, help='fix representation')

    # language model
    parser.add_argument('--lm', default=0, type=float, help='whether to use lm-pretrained to tokenize and embedding')
    parser.add_argument('--lm_path', default='lm/checkpoint-131580')

    parser.add_argument('--multiprocess', default=True, help='whether to multiprocess')
    parser.add_argument('--n_skills', default=8, type=int,
                        help='if skill is discrete: number of skills; if continuous:'
                             'the dimension of continuous vector')
    # skill-based: DIAYN
    parser.add_argument('--w_skills', default=0, type=float, help='DIAYN intrinsic reward')
    # skill-based: CIC
    parser.add_argument('--continuous_skill', default=1, help='skill: continuous 1|| discrete 0')
    parser.add_argument('--w_cic', default=0, type=float, help='cic loss')
    parser.add_argument('--r_cic', default=0, type=float, help='cic intrinsic reward')
    parser.add_argument('--update_skill_every', default=5, type=int, help='update skill frequency, 5',
                        choices=[5, 10, 20, 16, 32, 3])
    parser.add_argument('--discriminator_feature', default='diff_ss', type=str, choices=['ss', 'sas', 'diff_ss',
                                                                                         'diff_ss_a'])
    parser.add_argument('--r_linearizer', default=0, type=float, help='linearizer intrinsic reward')
    # random network distillation -> intrinsic reward
    parser.add_argument('--w_rnd', default=0, type=float, help='rnd loss')
    parser.add_argument('--r_rnd', default=0, type=float, help='intrinsic reward for random network distillation')
    parser.add_argument('--target_update_freq', default=500, type=int, help='rnd target network update frequency')
    # parser.add_argument('--topk_knn', default=4, type=int)
    parser.add_argument('--w_cross_matrix', default=1, type=float, help='cross correlation matrix for loss')
    parser.add_argument('--r_cross_matrix', default=0, type=float, help='cross correlation matrix for reward')

    parser.add_argument('--w_simsiam', default=0, type=float, help='simsam loss')

    parser.add_argument('--w_uniform', default=1, type=float, help='uniform loss')
    parser.add_argument('--w_alignment', default=0, type=float, help='alignment loss')
    return parser.parse_known_args()[0]


def start_redis():
    print('Starting Redis')
    subprocess.Popen(['redis-server', '--save', '\"\"', '--appendonly', 'no'])
    time.sleep(1)


def main():
    args = parse_args()
    print(args)
    print(args.output_dir)
    configure_logger(args.output_dir, args.wandb)
    if args.lm:
        args.embedding_dim = 768
        lm_model = DistilBERTLM(args.lm_path)
        args.vocab_size = len(lm_model.tokenizer)
    else:
        args.embedding_dim = 128
    agent = DRRN_Agent(args)
    if args.load is not None:
        print("Load from model: ", args.load)
    agent.load(args.load)


    # cache = {'loc': set()}
    cache = None
    if args.perturb:

        args.perturb_dict = {}
        from transformers import FSMTForConditionalGeneration, FSMTTokenizer

        mname1 = "facebook/wmt19-en-de"
        args.tokenizer1 = FSMTTokenizer.from_pretrained(mname1)
        args.model1 = FSMTForConditionalGeneration.from_pretrained(mname1)

        mname2 = "facebook/wmt19-de-en"
        args.tokenizer2 = FSMTTokenizer.from_pretrained(mname2)
        args.model2 = FSMTForConditionalGeneration.from_pretrained(mname2)

    # for Jericho version 2.X
    start_redis()
    # env = JerichoEnv2(args.rom_path, args.seed, args.env_step_limit, args)
    # for Jericho version 3.X
    env = JerichoEnv(args.rom_path, args.seed, args.env_step_limit, get_valid=True, cache=cache, args=args)
    # envs = [JerichoEnv(args.rom_path, args.seed, args.env_step_limit, get_valid=True, cache=cache, args=args) for _ in range(args.num_envs)]
    if args.multiprocess:
        envs = VecEnv(args.num_envs, env)
    else:
        envs = env

    # for jericho version 2
    # env.create()
    if args.lm:
        train(agent, env, envs, args.max_steps, args.update_freq, args.eval_freq, args.checkpoint_freq, args.log_freq,
              args.r_for, args.lm, lm_model)
    else:
        train(agent, env, envs, args.max_steps, args.update_freq, args.eval_freq, args.checkpoint_freq, args.log_freq,
              args.r_for, args=args)


if __name__ == "__main__":
    main()
