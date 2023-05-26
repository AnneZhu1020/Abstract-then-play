import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import os
from os.path import join as pjoin
from memory import *
from model import DRRN, CIC
from util import *

import logger
from transformers import BertTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRRN_Agent:
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        if args.lm:
            self.vocab_size = args.vocab_size
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vocab_size = len(self.tokenizer)
        # self.network = DRRN(args, self.vocab_size, args.embedding_dim, args.hidden_dim, args.fix_rep, args.hash_rep,
        #                     args.act_obs, args.lm, args.lm_path, args.multiprocess, args.n_skills).to(device)
        self.network = DRRN(args, self.vocab_size).to(device)
        if not args.lm:
            self.network.tokenizer = self.tokenizer
        self.memory = ABReplayMemory(args.memory_size, args.memory_alpha)

        self.save_path = args.output_dir
        self.clip = args.clip

        self.type_inv = args.type_inv
        self.type_for = args.type_for
        self.w_inv = args.w_inv
        self.w_for = args.w_for
        self.w_act = args.w_act
        self.perturb = args.perturb

        self.act_obs = args.act_obs

        self.n_skills = args.n_skills
        self.w_skills = args.w_skills
        self.w_cic = args.w_cic
        self.w_rnd = args.w_rnd
        self.w_simsiam = args.w_simsiam
        self.w_cross_matrix = args.w_cross_matrix
        self.r_cross_matrix = args.r_cross_matrix
        self.last_cross_matrix = 0
        self.continuous_skill = 0

        self.w_uniform = args.w_uniform
        self.w_alignment = args.w_alignment

        if self.n_skills:
            if args.continuous_skill:
                self.discriminator_loss = torch.nn.MSELoss()
                self.continuous_skill = 1
            else:
                self.discriminator_loss = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.learning_rate)

        self.discriminator_feature = args.discriminator_feature
        # for rnd network
        self.forward_mse = nn.MSELoss(reduction='none')
        self.update_proportion = 0.25

    def observe(self, transition, is_prior=False):
        self.memory.push(transition, is_prior)

    def build_state(self, ob, info):
        """ Returns a state representation built from various info sources. """
        if self.act_obs:
            acts = self.encode(info['valid'])
            obs_ids, look_ids, inv_ids = [], [], []
            for act in acts: obs_ids += act
            return State(obs_ids, look_ids, inv_ids)
        obs_ids = self.tokenizer.encode(ob)
        look_ids = self.tokenizer.encode(info['look'])
        inv_ids = self.tokenizer.encode(info['inv'])
        return State(obs_ids, look_ids, inv_ids)

    def build_states(self, obs, infos):
        return [self.build_state(ob, info) for ob, info in zip(obs, infos)]

    def build_lm_state(self, lm, obs, infos, prev_obs=None, prev_acts=None):
        """
        Return a state representation built from various info sources.
        """
        if prev_obs is None:
            return [State(lm.sent2ids(ob), lm.sent2ids(info['look']), lm.sent2ids(info['inv']), ob)
                    for ob, info in zip(obs, infos)]
        else:
            states = []
            for prev_ob, ob, info, act in zip(prev_obs, obs, infos, prev_acts):
                #             sent = "[CLS] %s [SEP] %s [SEP] %s [SEP]" % (prev_ob, act, ob + info['inv'] + info['look'])
                sent = " %s " % (ob + info['inv'] + info['look'])
                states.append(State(lm.sent2ids(ob), lm.act2ids(info['look']), lm.act2ids(info['inv']), sent))
            return states

    def encode(self, obs_list):
        """ Encode a list of observations """
        return [self.tokenizer.encode(o) for o in obs_list]

    def act(self, states, poss_acts, sample=True, eps=0.1, lm=None, lm_model=None, z=None, continuous_skill=None, step=None):
        """ Returns a string action from poss_acts. """
        idxs, values = self.network.act(states, poss_acts, sample, eps=eps, lm=lm, lm_model=lm_model, z=z,
                                        continuous_skill=continuous_skill, step=step)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, values

    def q_loss(self, transitions, need_qvals=False, need_cross_matrix=False, res_cross_matrix=None):
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        if self.n_skills:
            skills = torch.stack(batch.skill)
            next_qvals = self.network(batch.next_state, batch.next_acts, skills, self.continuous_skill)
        else:
            next_qvals = self.network(batch.next_state, batch.next_acts)

        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1 - torch.tensor(batch.done, dtype=torch.float, device=device))
        reward = torch.tensor(batch.reward, dtype=torch.float, device=device)

        # get intrinsic reward from cross correlation matrix-based loss
        if self.r_cross_matrix > 0 and need_cross_matrix:
            print("res_cross_matrix.size(): ", res_cross_matrix.size())
            reward += self.r_cross_matrix * res_cross_matrix
        # if self.n_skills:
        #     reward += self.w_skills * r_intrinsic

        targets = reward + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        if self.n_skills:
            qvals = self.network(batch.state, nested_acts, skills, self.continuous_skill)
        else:
            qvals = self.network(batch.state, nested_acts)

        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)
        loss = F.smooth_l1_loss(qvals, targets.detach())

        return (loss, qvals) if need_qvals else loss

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        nested_acts = tuple([[a] for a in batch.act])
        terms, loss = {}, 0

        # Compute Inverse dynamics loss
        if self.w_inv > 0:
            if self.type_inv == 'decode':
                terms['Loss_id'], terms['Acc_id'] = self.network.inv_loss_decode(batch.state, batch.next_state,
                                                                                 nested_acts, hat=True)
            elif self.type_inv == 'ce':
                terms['Loss_id'], terms['Acc_id'] = self.network.inv_loss_ce(batch.state, batch.next_state, nested_acts,
                                                                             batch.acts)
            else:
                raise NotImplementedError
            loss += self.w_inv * terms['Loss_id']

        # Compute Act reconstruction loss
        if self.w_act > 0:
            terms['Loss_act'], terms['Acc_act'] = self.network.inv_loss_decode(batch.state, batch.next_state,
                                                                               nested_acts, hat=False)
            loss += self.w_act * terms['Loss_act']

        # Compute Forward dynamics loss
        if self.w_for > 0:
            if self.type_for == 'l2':
                terms['Loss_fd'] = self.network.for_loss_l2(batch.state, batch.next_state, nested_acts)
            elif self.type_for == 'ce':
                terms['Loss_fd'], terms['Acc_fd'] = self.network.for_loss_ce(batch.state, batch.next_state, nested_acts,
                                                                             batch.acts)
            elif self.type_for == 'decode':
                terms['Loss_fd'], terms['Acc_fd'] = self.network.for_loss_decode(batch.state, batch.next_state,
                                                                                 nested_acts, hat=True)
            elif self.type_for == 'decode_obs':
                terms['Loss_fd'], terms['Acc_fd'] = self.network.for_loss_decode(batch.state, batch.next_state,
                                                                                 nested_acts, hat=False)

            loss += self.w_for * terms['Loss_fd']



        # Compute skill discriminator loss
        if self.w_skills > 0:
            state_out = self.network.state_rep(batch.state)
            logits = self.network.discriminator(state_out)  # size(8, 10)
            if self.continuous_skill:
                skills = torch.stack(batch.skill)
            else:
                skills = torch.tensor(batch.skill).to(device)
            terms['Loss_discriminator'] = self.discriminator_loss(logits, skills)
            loss += self.w_skills * terms['Loss_discriminator']

        if self.w_cic or self.w_cross_matrix or self.w_simsiam or self.w_alignment or self.w_uniform:
            state_out = self.network.state_rep(batch.state)
            next_state_out = self.network.state_rep(batch.next_state)
            _, action_out = self.network.act_rep(nested_acts)
            if self.continuous_skill:
                skills = torch.stack(batch.skill)
            else:
                skill_list = []
                for z in batch.skill:
                    z_one_hot = torch.zeros(self.n_skills)
                    z_one_hot[z] = 1
                    skill_list.append(z_one_hot.to(device))
                skills = torch.stack(skill_list)

        if self.w_cic > 0:
            terms['cic_loss'] = self.cic_loss(state_out, next_state_out, skills,
                                              action_out)  # state: size([39, 128]) skill: size([39, 10])
            loss += self.w_cic * terms['cic_loss']

        # cross correlation matrix
        if self.w_cross_matrix > 0:
            terms['cross_matrix'] = self.compute_barlowTwins_loss(state_out, next_state_out, skills,
                                                                  action_out)  # state: size([39, 128]) skill: size([39, 10])
            loss += self.w_cross_matrix * terms['cross_matrix']

        if self.w_rnd > 0:
            next_state_out = self.network.state_rep(batch.next_state)
            terms['rnd_loss'] = self.compute_rnd_loss(next_state_out)
            loss += self.w_rnd * terms['rnd_loss']

        if self.w_simsiam > 0:
            terms['simsam'] = self.compute_simsiam_loss(state_out, next_state_out, skills,
                                                        action_out)  # state: size([39, 128]) skill: size([39, 10])
            loss += self.w_simsiam * terms['simsam']

        if self.w_uniform > 0:
            query, key = self.network.cic_forward(state_out, next_state_out, skills, action_out)
            query = F.normalize(query, dim=1)
            key = F.normalize(key, dim=1)
            terms['uniform'] = (self.uniform_loss(query) + self.uniform_loss(key)) / 2
            loss += self.w_uniform * terms['uniform']

        if self.w_alignment > 0:
            query, key = self.network.cic_forward(state_out, next_state_out, skills, action_out)
            query = F.normalize(query, dim=1)
            key = F.normalize(key, dim=1)
            terms['alignment'] = self.align_loss(x=query, y=key)
            loss += self.w_alignment * terms['alignment']

        # Compute Q learning Huber loss
        if self.r_cross_matrix:
            terms['Loss_q'], qvals = self.q_loss(transitions, need_qvals=True, need_cross_matrix=True,
                                                 res_cross_matrix=terms['cross_matrix'])
        else:
            terms['Loss_q'], qvals = self.q_loss(transitions, need_qvals=True)
        loss += terms['Loss_q']

        # Backward
        terms.update({'Loss': loss, 'Q': qvals.mean()})
        self.optimizer.zero_grad()
        loss.backward()

        # for name, param in self.network.pred_net.named_parameters():
        #     print(str(name), ":", param.grad)
        # for name, param in self.network.skill_net.named_parameters():
        #     print(str(name), ":", param.grad)

        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return {k: float(v) for k, v in terms.items()}

    def cic_loss(self, state, next_state, skill, actions, temperature=1.0, reduction=True):
        query, key = self.network.cic_forward(state, next_state, skill, actions)

        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        cov = torch.mm(query, key.T)
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        row_sub = torch.zeros_like(neg).fill_(math.e ** (1 / temperature))
        neg = torch.clamp(neg - row_sub, min=1e-6)
        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature)
        # loss = - torch.log(pos / (neg + 1e-6)).mean()
        if reduction:
            loss = - torch.log(pos / (neg + 1e-6)).mean()
        else:
            loss = - torch.log(pos / (neg + 1e-6))
        return loss

    def linearizer_reward(self, state, skill):
        return self.network.cic_sim_state_skill(state, skill)

    def compute_rnd_reward(self, next_state):
        target_next_feature = self.network.compute_target_rnd(next_state)
        predict_next_feature = self.network.compute_predictor_rnd(next_state)

        intrinsic_rnd_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
        return intrinsic_rnd_reward

    def compute_rnd_loss(self, next_state):
        target_next_state_feature = self.network.compute_target_rnd(next_state)
        predict_next_state_feature = self.network.compute_predictor_rnd(next_state)
        rnd_loss = self.forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
        # Proportion of exp used for predictor update
        mask = torch.rand(len(rnd_loss)).to(device)
        mask = (mask < self.update_proportion).type(torch.FloatTensor).to(device)
        rnd_loss = (rnd_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))
        return rnd_loss[0]

    def compute_barlowTwins_loss(self, state, next_state, skill, actions):
        lambda_param = 5e-3  # default: 5e-3
        query, key = self.network.cic_forward(state, next_state, skill, actions)  # query <- skill

        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        bs = query.size(0)
        dim = query.size(1)
        # compute empirical cross-correlation matrix
        c = torch.mm(query.T, key) / bs
        # loss
        c_diff = (c - torch.eye(dim, device=device)).pow(2)
        c_diff[~torch.eye(dim, dtype=bool)] *= lambda_param  # size([num_skill, num_skill])
        loss = c_diff.sum()
        return loss

    def compute_simsiam_loss(self, state, next_state, skill, actions):
        # query, key = self.network.cic_forward(state, next_state, skill, actions)
        # query = F.normalize(query, dim=1)
        # key = F.normalize(key, dim=1)
        # loss = -F.cosine_similarity(query, key, dim=-1).mean()  # minimize loss -> make closer
        # return loss
        z1, z2, p1, p2 = self.network.simsiam_foward(state, next_state, skill, actions)
        loss1 = self.compute_neg_cosine_similarity(p1, z2.detach()).mean()
        loss2 = self.compute_neg_cosine_similarity(p2, z1.detach()).mean()
        avg_loss = 0.5 * loss1 + 0.5 * loss2
        return avg_loss

    def compute_neg_cosine_similarity(self, y1, y2):
        loss = -cosine_similarity(y1, y2, dim=1)
        return loss

    def align_loss(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def load(self, path=None):
        if path is None:
            return
        try:
            import os
            dir_path = os.path.dirname(os.path.realpath(__file__))
            print("dir_path:", dir_path)
            # self.memory = pickle.load(open(pjoin(path, 'memory.pkl'), 'rb'))
            network = torch.load(pjoin(path, 'model.pt'))
            # parts = ['embedding', 'encoder']  # , 'hidden', 'act_scorer']
            state_dict = network.state_dict()
            # state_dict = {k: v for k, v in state_dict.items() if any(part in k for part in parts)}
            print(state_dict.keys())
            self.network.load_state_dict(state_dict, strict=False)

        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())

    def save(self, step=''):
        try:
            os.makedirs(pjoin(self.save_path, step), exist_ok=True)
            pickle.dump(self.memory, open(pjoin(self.save_path, step, 'memory.pkl'), 'wb'))
            # pickle.dump(self.network.memory_emb, open(pjoin(self.save_path, step, 'memory_emb.pkl'), 'wb'))
            torch.save(self.network, pjoin(self.save_path, step, 'model.pt'))
        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())

    def save_emb(self, step=''):
        self.network.save_emb(self.save_path, str(step))
