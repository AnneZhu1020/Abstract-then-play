import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from transformers import DistilBertModel, DistilBertConfig
from torch.nn.functional import log_softmax
from torch import from_numpy
import os
import pickle
from os.path import join as pjoin


from util import pad_sequences
from memory import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRRN(torch.nn.Module):
    """
        Deep Reinforcement Relevance Network - He et al. '16

    """

    # def __init__(self, args, vocab_size, embedding_dim, hidden_dim, fix_rep=0, hash_rep=0, act_obs=0, lm=None, lm_path=None,
    #              multiprocess=None, n_skills=None):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.embedding = nn.Embedding(vocab_size, args.embedding_dim)
        self.obs_encoder = nn.GRU(args.embedding_dim, args.hidden_dim)
        self.look_encoder = nn.GRU(args.embedding_dim, args.hidden_dim)
        self.inv_encoder = nn.GRU(args.embedding_dim, args.hidden_dim)
        self.act_encoder = nn.GRU(args.embedding_dim, args.hidden_dim)
        if args.n_skills:
            self.hidden = nn.Linear(2 * args.hidden_dim + args.n_skills, args.hidden_dim)
        else:
            self.hidden = nn.Linear(2 * args.hidden_dim, args.hidden_dim)

        # self.hidden       = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.act_scorer = nn.Linear(args.hidden_dim, 1)

        self.state_encoder = nn.Linear(3 * args.hidden_dim, args.hidden_dim)
        self.inverse_dynamics = nn.Sequential(nn.Linear(2 * args.hidden_dim, 2 * args.hidden_dim), nn.ReLU(),
                                              nn.Linear(2 * args.hidden_dim, args.hidden_dim))
        self.forward_dynamics = nn.Sequential(nn.Linear(2 * args.hidden_dim, 2 * args.hidden_dim), nn.ReLU(),
                                              nn.Linear(2 * args.hidden_dim, args.hidden_dim))
        self.save_e = False

        if args.load is not None:
            self.save_e = True
            self.memory_emb = ReplayMemory(args.memory_size)

        # self.act_decoder = nn.GRU(hidden_dim, embedding_dim)
        # self.act_fc = nn.Linear(embedding_dim, vocab_size)
        #
        # self.obs_decoder = nn.GRU(hidden_dim, embedding_dim)
        # self.obs_fc = nn.Linear(embedding_dim, vocab_size)

        self.act_decoder = nn.GRU(args.embedding_dim, args.hidden_dim)
        self.act_fc = nn.Linear(args.hidden_dim, vocab_size)

        self.obs_decoder = nn.GRU(args.embedding_dim, args.hidden_dim)
        self.obs_fc = nn.Linear(args.hidden_dim, vocab_size)

        self.fix_rep = args.fix_rep
        self.hash_rep = args.hash_rep
        self.act_obs = args.act_obs
        self.hash_cache = {}

        self.lm = args.lm
        if self.lm:
            self.bert = DistilBertModel.from_pretrained(args.lm_path)
            for param in self.bert.parameters():
                param.requires_grad = False
            self.vocab_size = vocab_size

        self.multiprocess = args.multiprocess
        self.n_skills = args.n_skills
        if self.n_skills:
            self.hidden1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
            self.hidden2 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim // 2)
            self.q = nn.Linear(in_features=self.hidden_dim // 2, out_features=self.n_skills)

            if args.load is None:
                print("Init weight")
                self.init_weight(self.hidden1)
                self.hidden1.bias.data.zero_()
                self.init_weight(self.hidden2)
                self.hidden2.bias.data.zero_()
                self.q.bias.data.zero_()

        # if args.w_cic or args.w_cross_matrix or args.r_linearizer:
        #     self.obs_dim = args.hidden_dim
        #     self.skill_dim = args.n_skills
        #
        #     self.discriminator_feature = args.discriminator_feature
        #     if self.discriminator_feature == 'sas':
        #         input_dim = 3 * self.hidden_dim
        #     elif self.discriminator_feature == 'ss' or self.discriminator_feature == 'diff_ss_a':
        #         input_dim = 2 * self.hidden_dim
        #     elif self.discriminator_feature == 'diff_ss':
        #         input_dim = self.hidden_dim
        #     self.pred_net = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
        #                                   nn.ReLU(),
        #                                   nn.Linear(self.hidden_dim, self.skill_dim))
        #     project_skill = True
        #     if project_skill:
        #         self.skill_net = nn.Sequential(nn.Linear(self.skill_dim, self.hidden_dim),
        #                                        nn.ReLU(),
        #                                        nn.Linear(self.hidden_dim, self.skill_dim))
        #
        #     # self.pred_net = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
        #     #                               nn.Linear(self.hidden_dim, self.skill_dim))
        #     # project_skill = True
        #     # if project_skill:
        #     #     self.skill_net = nn.Sequential(nn.Linear(self.skill_dim, self.hidden_dim),
        #     #                                    nn.Linear(self.hidden_dim, self.skill_dim))
        #     else:
        #         self.skill_net = nn.Identity()

        if args.w_simsiam or args.r_linearizer or args.w_cic or args.w_cross_matrix or args.w_uniform or args.w_alignment:
            self.obs_dim = args.hidden_dim
            self.skill_dim = args.n_skills

            self.discriminator_feature = args.discriminator_feature
            if self.discriminator_feature == 'sas':
                input_dim = 3 * self.hidden_dim
            elif self.discriminator_feature == 'ss' or self.discriminator_feature == 'diff_ss_a':
                input_dim = 2 * self.hidden_dim
            elif self.discriminator_feature == 'diff_ss':
                input_dim = self.hidden_dim
            # self.pred_net = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
            #                               nn.ReLU(),
            #                               nn.Linear(self.hidden_dim, self.hidden_dim))
            # self.skill_net = nn.Sequential(nn.Linear(self.skill_dim, self.hidden_dim),
            #                                nn.ReLU(),
            #                                nn.Linear(self.hidden_dim, self.hidden_dim))
            # with projection and predictor network
            self.pred_net = nn.Sequential(nn.Linear(input_dim, self.hidden_dim))
            self.skill_net = nn.Sequential(nn.Linear(self.skill_dim, self.hidden_dim))

            # projection
            self.projection_net = nn.Sequential(nn.Linear(self.hidden_dim, args.hidden_dim // 2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim // 2, self.skill_dim))
            # predictor
            self.predictor_net = nn.Sequential(nn.Linear(self.skill_dim, args.hidden_dim // 2),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_dim // 2, self.skill_dim))



        if args.w_rnd or args.r_rnd:
            input_dim = self.hidden_dim
            output_dim = 64
            self.rnd_predictor = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_dim, output_dim))
            self.rnd_target = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                               nn.ReLU(),
                                               nn.Linear(self.hidden_dim, output_dim))
            for param in self.rnd_target.parameters():
                param.requires_grad = False

    def init_weight(self, layer, initializer='he normal'):
        if initializer == "xavier uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif initializer == 'he normal':
            nn.init.kaiming_normal_(layer.weight)

    def discriminator(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits

    def packed_hash(self, x):
        y = []
        for data in x:
            data = hash(tuple(data))
            if data in self.hash_cache:
                y.append(self.hash_cache[data])
            else:
                a = torch.zeros(self.hidden_dim).normal_(generator=torch.random.manual_seed(data))
                # torch.random.seed()
                y.append(a)
                self.hash_cache[data] = a
        y = torch.stack(y, dim=0).to(device)
        return y

    def packed_rnn(self, x, rnn):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.

            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """
        if self.hash_rep: return self.packed_hash(x)
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        if self.lm:
            embedding_output = self.bert(x_tt)[0]
            embed = embedding_output.permute(1, 0, 2)  # Time x Batch x EncDim
        else:
            embed = self.embedding(x_tt).permute(1, 0, 2)  # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out

    def state_rep(self, state_batch):
        # Zip the state_batch into an easy access format
        # if self.multiprocess:
        state = State(*zip(*state_batch))
        # else:
        #     print(state_batch)
        #     state = State(*state_batch)

        # Encode the various aspects of the state
        with torch.set_grad_enabled(not self.fix_rep):
            obs_out = self.packed_rnn(state.obs, self.obs_encoder)
            if self.act_obs: return obs_out
            look_out = self.packed_rnn(state.description, self.look_encoder)
            inv_out = self.packed_rnn(state.inventory, self.inv_encoder)
            state_out = self.state_encoder(torch.cat((obs_out, look_out, inv_out), dim=1))
        return state_out

    def act_rep(self, act_batch):
        # This is number of admissible commands in each element of the batch
        act_sizes = [len(a) for a in act_batch]
        # Combine next actions into one long list
        act_batch = list(itertools.chain.from_iterable(act_batch))
        with torch.set_grad_enabled(not self.fix_rep):
            act_out = self.packed_rnn(act_batch, self.act_encoder)
        return act_sizes, act_out

    def for_predict(self, state_batch, acts):
        _, act_out = self.act_rep(acts)
        state_out = self.state_rep(state_batch)
        next_state_out = state_out + self.forward_dynamics(torch.cat((state_out, act_out), dim=1))
        return next_state_out

    def inv_predict(self, state_batch, next_state_batch):
        state_out = self.state_rep(state_batch)
        next_state_out = self.state_rep(next_state_batch)
        # act_out = self.inverse_dynamics(torch.cat((state_out, next_state_out), dim=1))
        act_out = self.inverse_dynamics(torch.cat((state_out, next_state_out - state_out), dim=1))
        return act_out

    def inv_loss_l1(self, state_batch, next_state_batch, acts):
        _, act_out = self.act_rep(acts)
        act_out_hat = self.inv_predict(state_batch, next_state_batch)
        return F.l1_loss(act_out, act_out_hat)

    def inv_loss_l2(self, state_batch, next_state_batch, acts):
        _, act_out = self.act_rep(acts)
        act_out_hat = self.inv_predict(state_batch, next_state_batch)
        return F.mse_loss(act_out, act_out_hat)

    def inv_loss_ce(self, state_batch, next_state_batch, acts, valids, get_predict=False):
        act_sizes, valids_out = self.act_rep(valids)
        _, act_out = self.act_rep(acts)
        act_out_hat = self.inv_predict(state_batch, next_state_batch)
        now, loss, acc = 0, 0, 0
        if get_predict: predicts = []
        for i, j in enumerate(act_sizes):
            valid_out = valids_out[now: now + j]
            now += j
            values = valid_out.matmul(act_out_hat[i])
            label = valids[i].index(acts[i][0])
            loss += F.cross_entropy(values.unsqueeze(0), torch.LongTensor([label]).to(device))
            predict = values.argmax().item()
            acc += predict == label
            if get_predict: predicts.append(predict)
        return (loss / len(act_sizes), acc / len(act_sizes), predicts) if get_predict else (
            loss / len(act_sizes), acc / len(act_sizes))

    def inv_loss_decode(self, state_batch, next_state_batch, acts, hat=True, reduction='mean'):
        # hat: use rep(o), rep(o'); not hat: use rep(a)
        _, act_out = self.act_rep(acts)
        act_out_hat = self.inv_predict(state_batch, next_state_batch)

        acts_pad = pad_sequences([act[0] for act in acts])
        acts_tensor = torch.from_numpy(acts_pad).type(torch.long).to(device).transpose(0, 1)
        l, bs = acts_tensor.size()

        if self.lm:
            vocab = self.vocab_size
        else:
            vocab = self.embedding.num_embeddings
        outputs = torch.zeros(l, bs, vocab).to(device)
        input, z = acts_tensor[0].unsqueeze(0), (act_out_hat if hat else act_out).unsqueeze(0)
        for t in range(1, l):
            if self.lm:
                input = self.bert(input)[0]
            else:
                input = self.embedding(input)
            # print("input.size(): ", input.size())  # size([1, 8, 768])
            # print("z.size(): ", z.size())  # size([1, 8, 128])
            output, z = self.act_decoder(input, z)
            output = self.act_fc(output)
            outputs[t] = output
            top = output.argmax(2)
            input = top
        outputs, acts_tensor = outputs[1:], acts_tensor[1:]
        loss = F.cross_entropy(outputs.reshape(-1, vocab), acts_tensor.reshape(-1), ignore_index=0, reduction=reduction)
        if reduction == 'none':  # loss for each term in batch
            lens = [len(act[0]) - 1 for act in acts]
            loss = loss.reshape(-1, bs).sum(0).cpu() / torch.tensor(lens)
        nonzero = (acts_tensor > 0)
        same = (outputs.argmax(-1) == acts_tensor)
        acc_token = (same & nonzero).float().sum() / (nonzero).float().sum()  # token accuracy
        acc_action = (same.int().sum(0) == nonzero.int().sum(0)).float().sum() / same.size(1)  # action accuracy
        return loss, acc_action

    def for_loss_l2(self, state_batch, next_state_batch, acts):
        next_state_out = self.state_rep(next_state_batch)
        next_state_out_hat = self.for_predict(state_batch, acts)
        return F.mse_loss(next_state_out, next_state_out_hat)  # , reduction='sum')

    def for_loss_ce_batch(self, state_batch, next_state_batch, acts):
        # consider duplicates in next_state_batch
        next_states, labels = [], []
        for next_state in next_state_batch:
            if next_state not in next_states:
                labels.append(len(next_states))
                next_states.append(next_state)
            else:
                labels.append(next_states.index(next_state))
        labels = torch.LongTensor(labels).to(device)
        next_state_out = self.state_rep(next_states)
        next_state_out_hat = self.for_predict(state_batch, acts)
        logits = next_state_out_hat.matmul(next_state_out.transpose(0, 1))
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(1) == labels).float().sum() / len(labels)
        return loss, acc

    def for_loss_ce(self, state_batch, next_state_batch, acts, valids):
        # classify rep(o') from predict(o, a1), predict(o, a2), ...
        act_sizes, valids_out = self.act_rep(valids)
        _, act_out = self.act_rep(acts)
        next_state_out = self.state_rep(next_state_batch)
        now, loss, acc = 0, 0, 0
        for i, j in enumerate(act_sizes):
            valid_out = valids_out[now: now + j]
            now += j
            next_states_out_hat = self.for_predict([state_batch[i]] * j, [[_] for _ in valids[i]])
            values = next_states_out_hat.matmul(next_state_out[i])
            label = valids[i].index(acts[i][0])
            loss += F.cross_entropy(values.unsqueeze(0), torch.LongTensor([label]).to(device))
            predict = values.argmax().item()
            acc += predict == label
        return (loss / len(act_sizes), acc / len(act_sizes))

    def for_loss_decode(self, state_batch, next_state_batch, acts, hat=True):
        # hat: use rep(o), rep(a); not hat: use rep(o')
        next_state_out = self.state_rep(next_state_batch)  # size([39, 128])
        next_state_out_hat = self.for_predict(state_batch, acts)  # siz

        next_state = State(*zip(*next_state_batch))
        next_state_obs = torch.from_numpy(pad_sequences(next_state.obs)).type(torch.long).to(device)  # size([39, 102])
        next_state_inv = torch.from_numpy(pad_sequences(next_state.inventory)).type(torch.long).to(
            device)  # size([39, 125])
        next_state_look = torch.from_numpy(pad_sequences(next_state.description)).type(torch.long).to(
            device)  # size([39, 31])
        next_state_tensor = torch.cat((next_state_obs, next_state_inv, next_state_look), dim=1).transpose(0,
                                                                                                          1)  # size([258, 39])
        # import pdb; pdb.set_trace()
        # next_state_pad = pad_sequences(next_state_batch)
        # next_state_tensor = torch.from_numpy(next_state_pad).type(torch.long).to(device).transpose(0, 1)
        l, bs = next_state_tensor.size()
        if self.lm:
            vocab = self.vocab_size
        else:
            vocab = self.embedding.num_embeddings
        outputs = torch.zeros(l, bs, vocab).to(device)
        input, z = next_state_tensor[0].unsqueeze(0), (next_state_out_hat if hat else next_state_out).unsqueeze(0)
        for t in range(1, l):
            if self.lm:
                input = self.bert(input)[0]
            else:
                input = self.embedding(input)
            output, z = self.obs_decoder(input, z)
            output = self.obs_fc(output)
            outputs[t] = output
            top = output.argmax(2)
            input = top
        outputs, next_state_tensor = outputs[1:].reshape(-1, vocab), next_state_tensor[1:].reshape(-1)
        loss = F.cross_entropy(outputs, next_state_tensor, ignore_index=0)
        nonzero = (next_state_tensor > 0)
        same = (outputs.argmax(1) == next_state_tensor)
        acc = (same & nonzero).float().sum() / (nonzero).float().sum()  # token accuracy
        return loss, acc

    def save_emb(self, save_path, step=''):
        os.makedirs(pjoin(save_path, step), exist_ok=True)
        pickle.dump(self.memory_emb, open(pjoin(save_path, step, 'memory_emb.pkl'), 'wb'))

    def forward(self, state_batch, act_batch, skill_z=None, continuous_skill=None, step=None):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids

            Returns a tuple of tensors containing q-values for each item in the batch
        """
        state_out = self.state_rep(state_batch)  # size([64, 128])
        act_sizes, act_out = self.act_rep(act_batch)  # size([326, 128])

        # save state_out: size([64, 128])
        # save skill_z: size([64, 8])
        if self.save_e and step is not None:
            for (state, skill) in zip(state_out, skill_z):
                self.memory_emb.push(state, skill, step)

        # Expand the state to match the batches of actions
        state_out = torch.cat([state_out[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
        # if skill_z is not None:
        #     p_z = from_numpy(self.p_z).to(device)
        #     zs = skill_z.to(device)
        #     logits = self.discriminator(state_out)[0]  # size(10, )
        #     logq_z_ns = log_softmax(logits, dim=-1)  # size(10, )
        #     p_z = p_z.gather(-1, zs)
        #     r_intrinsic = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)  # size(8)
        #
        if torch.isnan(state_out).sum().item():
            print("state_out: ", state_out)
            print("skill_z: ", skill_z)
            print("act_out: ", act_out)
            print("state_batch: ", state_batch)
            print("act_batch: ", act_batch)

        if skill_z is not None:
            skill_z = torch.cat([skill_z[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
            if continuous_skill:
                state_out = torch.cat((state_out, skill_z), dim=1)
            else:  # one-hot concat
                state_out = self.concat_state_latent(state_out, skill_z, self.n_skills)
        z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        # Split up the q-values by batch

        return act_values.split(act_sizes)

    def concat_state_latent(self, state_out, skill_z, n):
        state_res = []
        for state, z in zip(state_out, skill_z):
            z_one_hot = torch.zeros(n)
            z_one_hot[z] = 1
            state_res.append(torch.cat([state, z_one_hot.to(device)]))
        return torch.stack(state_res)

    def act(self, states, act_ids, sample=True, eps=0.1, lm=None, lm_model=None, z=None, continuous_skill=None, step=None):
        """ Returns an action-string, optionally sampling from the distribution
            of Q-Values.
        """
        if lm:
            alpha = 0
            k = -1
            q_values = self.forward(states, act_ids)
            if alpha > 0 or (eps is not None and k != -1):  # need to use lm_values
                lm_values = [torch.tensor(lm_model.score(state.obs, act_ids), device=device) for state, act_ids in
                             zip(states, act_ids)]
                act_values = [q_value * (1 - alpha) + bert_value * alpha
                              for q_value, bert_value in zip(q_values, lm_values)]
            else:
                act_values = q_values

            if eps is None:  # sample ~ softmax(act_values)
                act_idxs = [torch.multinomial(F.softmax(vals, dim=0), num_samples=1).item() for vals in act_values]
            else:  # w.p. eps, ~ softmax(act_values) | uniform(top_k(act_values)), w.p. (1-eps) arg max q_values
                if k == 0:  # soft sampling
                    act_idxs = [torch.multinomial(F.softmax(vals, dim=0), num_samples=1).item() for vals in lm_values]
                elif k == -1:
                    act_idxs = [np.random.choice(range(len(vals))) for vals in q_values]
                else:  # hard (uniform) sampling
                    act_idxs = [np.random.choice(vals.topk(k=min(k, len(vals)), dim=0).indices.tolist()) for vals in
                                lm_values]
                act_idxs = [vals.argmax(dim=0).item() if np.random.rand() > eps else idx for idx, vals in
                            zip(act_idxs, q_values)]
            return act_idxs, act_values

        else:
            if self.n_skills:
                act_values = self.forward(states, act_ids, z, continuous_skill, step)
            else:
                act_values = self.forward(states, act_ids, step)

            if sample:
                act_probs = [F.softmax(vals, dim=0) for vals in act_values]
                act_idxs = [torch.multinomial(probs, num_samples=1).item() for probs in act_probs]
            else:
                act_idxs = [vals.argmax(dim=0).item() if np.random.rand() > eps else np.random.randint(len(vals)) for
                            vals
                            in act_values]
            return act_idxs, act_values

    # Contrastive intrinsic part

    def cic_forward(self, state, next_state, skill, actions):
        assert len(state.size()) == len(next_state.size())
        query = self.skill_net(skill)  # size([39, num_skill])
        if self.discriminator_feature == 'sas':
            key = self.pred_net(torch.cat([state, next_state, actions], dim=1))
        elif self.discriminator_feature == 'ss':
            key = self.pred_net(torch.cat([state, next_state], dim=1))
        elif self.discriminator_feature == 'diff_ss':
            key = self.pred_net(next_state-state)
        elif self.discriminator_feature == 'diff_ss_a':
            key = self.pred_net(torch.cat([next_state - state, actions], dim=1))
        return query, key

    def cic_sim_state_skill(self, state, skill):
        skill_out = self.skill_net(skill)  # size([39, num_skill])
        state_out = self.pred_net(state)
        res = torch.sum(skill_out * state_out, dim=-1)  # size(bs)
        return res

    def simsiam_foward(self, state, next_state, skill, actions):
        assert len(state.size()) == len(next_state.size())
        query = self.skill_net(skill)  # size([39, num_skill])
        if self.discriminator_feature == 'sas':
            key = self.pred_net(torch.cat([state, next_state, actions], dim=1))
        elif self.discriminator_feature == 'ss':
            key = self.pred_net(torch.cat([state, next_state], dim=1))
        elif self.discriminator_feature == 'diff_ss':
            key = self.pred_net(next_state-state)
        elif self.discriminator_feature == 'diff_ss_a':
            key = self.pred_net(torch.cat([next_state - state, actions], dim=1))
        z1, z2 = self.projection_net(query), self.projection_net(key)
        p1, p2 = self.predictor_net(z1), self.predictor_net(z2)
        return z1, z2, p1, p2


    # Random Network Distillation
    def compute_target_rnd(self, next_state):
        return self.rnd_target(next_state)

    def compute_predictor_rnd(self, next_state):
        return self.rnd_predictor(next_state)


class CIC(torch.nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, discriminator_feature, project_skill=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        # self.pred_net = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
        #                               nn.ReLU(),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.ReLU(),
        #                               nn.Linear(hidden_dim, skill_dim))
        # if project_skill:
        #     self.skill_net = nn.Sequential(nn.Linear(skill_dim, hidden_dim),
        #                                    nn.ReLU(),
        #                                    nn.Linear(hidden_dim, hidden_dim),
        #                                    nn.ReLU(),
        #                                    nn.Linear(hidden_dim, skill_dim))
        self.discriminator_feature = discriminator_feature
        if self.discriminator_feature == 'sas':
            input_dim = 3*hidden_dim
        elif self.discriminator_feature == 'ss' or self.discriminator_feature == 'diff_ss_a':
            input_dim = 2*hidden_dim
        elif self.discriminator_feature == 'diff_ss':
            input_dim = hidden_dim
        self.pred_net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, skill_dim))

        if project_skill:
            self.skill_net = nn.Sequential(nn.Linear(skill_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, skill_dim))
        else:
            self.skill_net = nn.Identity()

    def forward(self, state, next_state, skill, actions):
        assert len(state.size()) == len(next_state.size())
        query = self.skill_net(skill)  # size([39, num_skill])
        if self.discriminator_feature == 'sas':
            key = self.pred_net(torch.cat([state, next_state, actions], dim=1))
        elif self.discriminator_feature == 'ss':
            key = self.pred_net(torch.cat([state, next_state], dim=1))
        elif self.discriminator_feature == 'diff_ss':
            key = self.pred_net(next_state-state)
        elif self.discriminator_feature == 'diff_ss_a':
            key = self.pred_net(torch.cat([next_state - state, actions], dim=1))
        return query, key

    def sim_state_skill(self, state, skill):
        skill_out = self.skill_net(skill)  # size([39, num_skill])
        state_out = self.pred_net(state)
        res = torch.sum(skill_out * state_out, dim=-1)  # size(bs)
        return res


class RMS:
    def __init__(self, epsilon=1e-4, shape=(1, )):
        self.M = torch.zeros(shape, requires_grad=False).to(device)
        self.S = torch.ones(shape, requires_grad=False).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S
