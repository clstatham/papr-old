import random
import functools
import re
from datetime import datetime
import numpy as np
from collections import namedtuple
from torch import nn
from torch.nn import functional as F
import torch
from tqdm import tqdm
import logging
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from parsec import *

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
outdir = f"training/{timestamp}"
os.makedirs(outdir, exist_ok=True)

MAX_OBS_LEN = 100
# BATCH_SIZE = 16

successful_renders = 0

ALL_TOKENS = []


builtin_tokens = {
    # "Sin",
    # "Cos",
    # "Exp",
    "Tanh": 1,
    "Abs": 1,
    "SineFm": 4,
    "SineOsc": 2,
    "SawOsc": 2,
    "SquareOsc": 2,
    # "Clock",
    # "Delay",
    # "Redge",
    # "Fedge",
    # "Var",
    "Max": 2,
    "Min": 2,
    # "Clip",
    # "If",
    # "Not",
}
audio_tokens = {"@" + tok: val for tok, val in builtin_tokens.items()}
# control_tokens = {"#" + tok: val for tok, val in builtin_tokens.items()}
builtin_tokens = {**audio_tokens}  # , **control_tokens
ALL_TOKENS += list(builtin_tokens.keys())
# SOT_TOKEN = "<|START|>"
EOT_TOKEN = "<|END|>"
operator_tokens = [
    # "<", ">",
    # "+", "-", "*", "/",
    # "&", "|", "^",
    # "!=",
]
paren_tokens = ["(", ")"]
keyword_tokens = []

misc_tokens = [";", "="]

ALL_TOKENS += misc_tokens + keyword_tokens + operator_tokens + paren_tokens

MAX_VARS = 5
# MAX_GRAPHS = 1

var_tokens = ["@var" + str(i) for i in range(MAX_VARS)]
# var_tokens += ["#var" + str(i) for i in range(MAX_VARS, MAX_VARS * 2)]
# graph_tokens = ["#Graph" + str(i) for i in range(MAX_GRAPHS)]
# graph_tokens += ["@Graph" + str(i) for i in range(MAX_GRAPHS)]
var_tokens += ["@dac0"]
ALL_TOKENS += var_tokens

# ALL_TOKENS += [chr(x) for x in range(32, 127)]
ALL_TOKENS += [' ']
ALL_TOKENS += ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

ALL_TOKENS = ALL_TOKENS + [EOT_TOKEN]

whitespace0 = regex(r'\s*', re.MULTILINE)
whitespace1 = regex(r'\s+', re.MULTILINE)
def lexeme(p): return p << whitespace0


def to_parser(tokens): return functools.reduce(
    lambda x, y: x ^ y, [lexeme(string(x)) for x in tokens])


eot = lexeme(string(EOT_TOKEN))

equals = lexeme(string('='))
opparen = lexeme(string('('))
clparen = lexeme(string(')'))
semi = lexeme(string(';'))
capital = one_of(''.join([chr(x)
                          for x in range(ord('A'), ord('Z') + 1)]))
lowercase = one_of(''.join([chr(x)
                            for x in range(ord('a'), ord('z') + 1)]))
digit = one_of('1234567890')
underscore = string('_')
signalrate = one_of("@#")
var = to_parser(var_tokens)
# op = to_parser(operator_tokens)
builtin = to_parser(builtin_tokens.keys())

TOKEN_CATEGORIES = {
    # "special": [EOT_TOKEN],
    "whitespace": [" "],
    "equals": ["="],
    "opparen": ["("],
    "clparen": [")"],
    "semi": [";"],
    "underscore": ["_"],
    "var": var_tokens,
    "builtin": list(builtin_tokens.keys()),
}


def parser(script):
    def inc_if_ok(inp, x, p: Parser):
        try:
            _, inp2 = p.parse_partial(inp)
            return True, inp2, x + 1
        except ParseError:
            return False, inp, x

    def number(inp, reward):
        go, inp, reward = inc_if_ok(inp, reward, many1(
            digit).parsecmap(lambda l: ''.join(l)))
        if not go:
            return False, inp, reward
        _, inp, reward = inc_if_ok(inp, reward, string('.'))
        _, inp, reward = inc_if_ok(inp, reward, many1(
            digit).parsecmap(lambda l: ''.join(l)))
        return True, inp, reward

    # def infix_op(inp, reward):
    #     # go, inp, reward = inc_if_ok(inp, reward, opparen)
    #     # if not go:
    #     #     return False, inp, reward
    #     go, inp, reward = expr(inp, reward)
    #     if not go:
    #         return False, inp, reward
    #     go, inp, reward = inc_if_ok(inp, reward, op)
    #     if not go:
    #         return False, inp, reward
    #     go, inp, reward = expr(inp, reward)
    #     if not go:
    #         return False, inp, reward
    #     # go, inp, reward = inc_if_ok(inp, reward, clparen)
    #     # if not go:
    #     #     return False, inp, reward
    #     return True, inp.strip(), reward + 3

    def graphcall(inp, reward):
        try:
            graphname, inp = builtin.parse_partial(inp)
            reward += 1
        except ParseError:
            return False, inp, reward
        go, inp, reward = inc_if_ok(inp, reward, opparen)
        if not go:
            return False, inp, reward
        for _ in range(builtin_tokens[graphname]):
            go, inp, reward = expr(inp, reward)
            if not go:
                return False, inp, reward
            go, inp, reward = inc_if_ok(inp, reward, string(' '))
            if not go and builtin_tokens[graphname] > 1:
                return False, inp, reward
        go, inp, reward = inc_if_ok(inp, reward, clparen)
        if not go:
            return False, inp, reward
        return True, inp, reward + 3

    def expr(inp, reward):
        go, inp2, reward = inc_if_ok(inp, reward, var)
        if go:
            return True, inp2, reward
        go, inp2, reward = graphcall(inp, reward)
        if go:
            return True, inp2, reward
        # go, inp, reward = infix_op(inp, reward)
        # if go:
        #     return True, inp, reward
        go, inp, reward = number(inp, reward)
        if not go:
            return False, inp, reward

        return True, inp, reward + 3

    def statement(inp, reward):
        go, inp, reward = inc_if_ok(inp, reward, var)
        if not go:
            return False, inp, reward
        go, inp, reward = inc_if_ok(inp, reward, equals)
        if not go:
            return False, inp, reward
        go, inp, reward = expr(inp, reward)
        if not go:
            return False, inp, reward
        go, inp, reward = inc_if_ok(inp, reward, semi)
        if not go:
            return False, inp, reward
        return True, inp, reward + 5

    def statements(inp, reward):
        go, inp, reward = statement(inp, reward)
        if not go:
            return False, inp, reward
        while True:
            go, inp, reward = statement(inp, reward)
            if not go:
                break
        go, inp, reward = inc_if_ok(inp, reward, eot)
        if not go:
            return False, inp, reward
        return True, inp, reward + 10

    reward = 0
    _, script, reward = statements(script, reward)
    return reward


def random_valid_tokens(n):
    # kinda monte-carlo
    def next_token(total):
        last_reward = parser(''.join(total))
        for _ in range(100):
            cat = random.choice(list(TOKEN_CATEGORIES.values()))
            nx = random.choice(cat)
            tmp_total = total + [nx]
            tmp_reward = parser(''.join(tmp_total))
            if tmp_reward > last_reward:
                return nx
        return None
    total = []
    for _ in range(n):
        nx = next_token(total)
        if nx is None:
            return total
        else:
            total = total + [nx]
    return total


def encode(tokens):
    toks = [torch.tensor(ALL_TOKENS.index(
        token), dtype=torch.int64) for token in tokens]
    return torch.stack(toks).to(torch.int64)


class PaaiprEnv(object):
    def reset(self):
        self.script = [random.choice(var_tokens)]
        self.last_parse_score = parser(''.join(self.script))
        self.initial_parse_score = self.last_parse_score
        return encode(self.script)

    def step(self, action):
        reward = 0.0
        token = ALL_TOKENS[action.squeeze(-1).item()]
        terminated = token == EOT_TOKEN
        # if token == self.script[-1]:
        #     # punish repeating the same token over and over (particularly open paren)
        #     reward -= 2.0
        self.script += [token]

        global successful_renders

        if len(self.script) > MAX_OBS_LEN:
            terminated = True

        parser_reward = parser(''.join(self.script))
        if parser_reward > self.last_parse_score:
            reward += 1.0
            # reward += 1.0
        elif len(self.script) > 1:
            reward += parser_reward - self.initial_parse_score
            terminated = True
            # punish unmatched parentheses
            paren_count = 0
            for t in self.script:
                if t == "(":
                    paren_count += 1
                elif t == ")":
                    paren_count -= 1
            reward -= abs(paren_count)
        self.last_parse_score = parser_reward

        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        return encode(self.script), reward, terminated

    def __init__(self, device="cpu"):
        # self.script = random_valid_tokens(50)
        self.device = device
        # self.last_parse_score = parser(''.join(self.script))
        # self.initial_parse_score = self.last_parse_score


class PositionalEncoding(nn.Module):
    def __init__(self, width, device):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(p=0.1)
        position = torch.arange(MAX_OBS_LEN).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2) *
                             (-np.log(10000.0) / width))
        pe = torch.zeros(MAX_OBS_LEN, 1, width).to(device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), 0]
        return self.dropout(x)


class Policy(nn.Module):
    def __init__(self, width, device, nhead, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.width = width
        self.pe = PositionalEncoding(width, device=device)
        encoder_layers = nn.TransformerEncoderLayer(
            width, nhead, device=device)
        self.encoder = nn.TransformerEncoder(encoder_layers, 3).to(device)
        self.embed = nn.Embedding(len(ALL_TOKENS), width, device=device)
        self.action_head = nn.Linear(width, len(ALL_TOKENS), device=device)
        # self.value_head = nn.Linear(num_cells, 1, device=device)

        self.saved_actions = []
        self.rewards = []

        self.init_weights()

    def forward(self, x):
        x = self.embed(x) * np.sqrt(self.width)
        x = self.pe(x)
        x = self.encoder(x)
        action_prob = self.action_head(x)
        action_prob_softmax = F.softmax(action_prob, dim=-1)
        # state_values = self.value_head(x)
        return action_prob_softmax[..., -1, :]

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.action_head.weight.data.uniform_(-initrange, initrange)
        self.action_head.bias.data.zero_()


SavedAction = namedtuple('SavedAction', ['log_prob'])
if __name__ == "__main__":
    print("All Tokens:")
    print(ALL_TOKENS)

    device = "cuda" if torch.has_cuda else "cpu"
    width = 1024
    lr = 1e-5

    env = PaaiprEnv(device=device)

    model = Policy(width, nhead=2, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(outdir)
    running_reward = 0
    pbar = tqdm()
    i_episode = 0
    # epsilon = 0.2
    # decay = 0.99999
    best = 0.0
    try:
        while True:
            state = env.reset().to(device)
            ep_rewards = []
            while True:
                probs = model(state)
                m = Categorical(probs)
                # if np.random.rand() < epsilon:
                #     action = torch.tensor(
                #         np.random.choice(range(len(ALL_TOKENS)))).to(device)
                # else:
                action = m.sample()

                # probs = torch.argsort(probs, -1, descending=True)
                # action = probs[..., 0]
                model.saved_actions.append(
                    SavedAction(m.log_prob(action)))
                state, reward, done = env.step(action=action)
                state = state.to(device)
                model.rewards.append(reward)
                ep_rewards.append(reward.cpu().numpy())
                # epsilon *= decay
                if done:
                    break
            ep_reward = np.sum(ep_rewards)
            running_reward = 0.4 * ep_reward + 0.6 * running_reward
            R = 0
            returns = []
            for r in model.rewards[::-1]:
                R = r + 0.99 * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / \
                (returns.std(unbiased=False) + float(np.finfo(np.float32).eps))
            policy_losses = []
            # value_losses = []
            for (log_prob,), R in zip(model.saved_actions, returns):
                # advantage = R - value.item()
                policy_losses.append(-log_prob * R)
                # value_losses.append(F.smooth_l1_loss(
                #     value, torch.tensor([R], device=device)))
            optim.zero_grad()
            loss = torch.stack(policy_losses).sum()
            # torch.stack(value_losses).sum()
            loss.to(device).backward()
            optim.step()

            # mean_reward = torch.stack(model.rewards).mean().item()
            # max_reward = torch.stack(model.rewards).max().item()

            del model.rewards[:]
            del model.saved_actions[:]

            pbar.set_description(
                f"ep_reward: {ep_reward:4.4f}, avg_reward: {running_reward:4.4f}, loss: {loss.item():4.4f}")
            pbar.update()

            writer.add_scalar("Reward/Episode", ep_reward, i_episode)
            writer.add_scalar("Reward/RunningAvg", running_reward, i_episode)
            writer.add_scalar("Loss", loss.item(), i_episode)
            # writer.add_scalar("Epsilon", epsilon, i_episode)

            with open(f"{outdir}/out.papr", "w", encoding="utf-8") as f:
                f.write(''.join(env.script[:-1]))
            if running_reward > best:
                best = running_reward
                with open(f"{outdir}/out_best.papr", "w", encoding="utf-8") as f:
                    f.write(''.join(env.script[:-1]))
                torch.save(
                    model.cpu(), f"{outdir}/model_best.pt")
                torch.save(
                    optim, f"{outdir}/optim_best.pt")
                model.to(device)
            if running_reward >= MAX_OBS_LEN:
                print("Running reward reached threshold, stopping training")
                break
            i_episode += 1
    except KeyboardInterrupt:
        pass
    pbar.close()
    torch.save(model.cpu(), f"{outdir}/model.pt")
    torch.save(optim, f"{outdir}/optim.pt")
