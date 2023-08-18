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
import subprocess
import os
import shutil
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
control_tokens = {"#" + tok: val for tok, val in builtin_tokens.items()}
builtin_tokens = {**audio_tokens, **control_tokens}
ALL_TOKENS += list(builtin_tokens.keys())
SOT_TOKEN = "<|START|>"
EOT_TOKEN = "<|END|>"
operator_tokens = [
    "<", ">",
    "+", "-", "*", "/",
    "&", "|", "^",
    "!=",
]
paren_tokens = ["(", ")"]
keyword_tokens = []

misc_tokens = [";", "="]

ALL_TOKENS += misc_tokens + keyword_tokens + operator_tokens + paren_tokens

MAX_VARS = 3
# MAX_GRAPHS = 1

var_tokens = ["@var" + str(i) for i in range(MAX_VARS)]
var_tokens += ["#var" + str(i) for i in range(MAX_VARS, MAX_VARS * 2)]
# graph_tokens = ["#Graph" + str(i) for i in range(MAX_GRAPHS)]
# graph_tokens += ["@Graph" + str(i) for i in range(MAX_GRAPHS)]
var_tokens += ["@dac0"]
ALL_TOKENS += var_tokens

ALL_TOKENS = [SOT_TOKEN] + ALL_TOKENS + [EOT_TOKEN]
assert ALL_TOKENS[0] == SOT_TOKEN

whitespace = regex(r'\s*', re.MULTILINE)
def lexeme(p): return p << whitespace


def to_parser(tokens): return functools.reduce(
    lambda x, y: x ^ y, [lexeme(string(x)) for x in tokens])


sot = lexeme(string(SOT_TOKEN))
eot = lexeme(string(EOT_TOKEN))

equals = lexeme(string('='))
opparen = lexeme(string('('))
clparen = lexeme(string(')'))
semi = lexeme(string(';'))
var = to_parser(var_tokens)
op = to_parser(operator_tokens)
builtin = to_parser(builtin_tokens.keys())


def parser(script):
    def inc_if_ok(inp, x, p: Parser):
        try:
            _, inp = p.parse_partial(inp)
            return True, inp, x + 1
        except ParseError:
            return False, inp, x

    def infix_op(inp, depth):
        go, inp, depth = inc_if_ok(inp, depth, opparen)
        if not go:
            return False, inp, depth
        go, inp, depth = expr(inp, depth)
        if not go:
            return False, inp, depth
        go, inp, depth = inc_if_ok(inp, depth, op)
        if not go:
            return False, inp, depth
        go, inp, depth = expr(inp, depth)
        if not go:
            return False, inp, depth
        return inc_if_ok(inp, depth, clparen)

    def graphcall(inp, depth):
        try:
            graphname, inp = builtin.parse_partial(inp)
            depth += 1
        except ParseError:
            return False, inp, depth
        go, inp, depth = inc_if_ok(inp, depth, opparen)
        if not go:
            return False, inp, depth
        for _ in range(builtin_tokens[graphname]):
            go, inp, depth = expr(inp, depth)
            if not go:
                return False, inp, depth
        return inc_if_ok(inp, depth, clparen)

    def expr(inp, depth):
        go, inp, depth = inc_if_ok(inp, depth, var)
        if go:
            return True, inp, depth
        go, inp, depth = infix_op(inp, depth)
        if go:
            return True, inp, depth
        return graphcall(inp, depth)

    def statement(inp, depth):
        go, inp, depth = inc_if_ok(inp, depth, var)
        if not go:
            return False, inp, depth
        go, inp, depth = inc_if_ok(inp, depth, equals)
        if not go:
            return False, inp, depth
        go, inp, depth = expr(inp, depth)
        if not go:
            return False, inp, depth
        return inc_if_ok(inp, depth, semi)

    def statements(inp, depth):
        go, inp, depth = statement(inp, depth)
        if not go:
            return False, inp, depth
        while True:
            go, inp, depth = statement(inp, depth)
            if not go:
                break
        return inc_if_ok(inp, depth, eot)

    depth = 0
    go, script, depth = statements(script, depth)
    return depth


def encode(tokens):
    toks = [torch.tensor(ALL_TOKENS.index(
        token), dtype=torch.int64) for token in tokens]
    # toks = [F.one_hot(token, num_classes=len(ALL_TOKENS)) for token in toks]
    return torch.stack(toks).to(torch.int64)


class PaaiprEnv(object):
    def reset(self):
        self.script = [SOT_TOKEN]
        self.last_parse_score = 0
        return encode(self.script)

    def step(self, action):
        reward = 0.0
        token = ALL_TOKENS[action.squeeze(-1).item()]
        terminated = token == EOT_TOKEN
        self.script += [token]

        global successful_renders

        if len(self.script) > MAX_OBS_LEN:
            terminated = True

        parser_reward = parser(' '.join(self.script[1:]))
        if parser_reward > self.last_parse_score:
            reward = 1.0
            # reward += 1.0
        elif len(self.script) > 1:
            reward = parser_reward
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

        # momentum for long chains of correct tokens
        # reward += 1.01 ** parser_reward - 1.0

        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        return encode(self.script), reward, terminated

    def __init__(self, device="cpu"):
        self.script = list()
        self.device = device
        self.last_parse_score = 0


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
if __name__ == "__main__":

    print("All Tokens:")
    print(ALL_TOKENS)

    device = "cuda" if torch.has_cuda else "cpu"
    num_cells = 1024
    lr = 1e-4

    env = PaaiprEnv(device=device)

    class Policy(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.embed = nn.Embedding(
                len(ALL_TOKENS), num_cells, device=device)
            self.lstm1 = nn.LSTM(num_cells, num_cells,
                                 batch_first=True, num_layers=1, device=device)
            self.atn2 = nn.MultiheadAttention(
                num_cells, 2, batch_first=True, device=device)
            self.action_head = nn.Sequential(
                nn.Linear(num_cells, num_cells, device=device),
                nn.Tanh(),
                nn.Linear(num_cells, len(ALL_TOKENS), device=device))
            self.value_head = nn.Linear(num_cells, 1, device=device)

            self.saved_actions = []
            self.rewards = []

        def forward(self, x):
            x = self.embed(x)
            x = x.unsqueeze(0).to(device)
            x, _ = self.lstm1(x)
            x, _ = self.atn2(x, key=x, value=x)
            x = x[..., -1, :]
            action_prob = self.action_head(x)
            action_prob_softmax = F.softmax(action_prob, dim=-1)
            state_values = self.value_head(x)
            return action_prob_softmax.squeeze(0), state_values.squeeze(0)

    model = Policy()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(outdir)
    running_reward = 0
    pbar = tqdm()
    i_episode = 0
    epsilon = 0.2
    decay = 0.99999
    best = 0.0
    try:
        while True:
            state = env.reset().to(device)
            ep_rewards = []
            while True:
                probs, state_value = model(state)
                m = Categorical(probs)
                if np.random.rand() < epsilon:
                    action = torch.tensor(
                        np.random.choice(range(len(ALL_TOKENS)))).to(device)
                else:
                    action = m.sample()

                # probs = torch.argsort(probs, -1, descending=True)
                # action = probs[..., 0]
                model.saved_actions.append(
                    SavedAction(m.log_prob(action), state_value))
                state, reward, done = env.step(action=action)
                state = state.to(device)
                model.rewards.append(reward)
                ep_rewards.append(reward.cpu().numpy())
                epsilon *= decay
                if done:
                    break
            ep_reward = np.sum(ep_rewards)
            running_reward = 0.1 * ep_reward + 0.9 * running_reward
            R = 0
            returns = []
            for r in model.rewards[::-1]:
                R = r + 0.99 * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            old_returns = returns.clone()
            returns = (returns - returns.mean()) / \
                (returns.std(unbiased=False) + float(np.finfo(np.float32).eps))
            # print(old_returns.mean(), returns.mean())
            policy_losses = []
            value_losses = []
            for (log_prob, value), R in zip(model.saved_actions, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(F.smooth_l1_loss(
                    value, torch.tensor([R], device=device)))
            optim.zero_grad()
            loss = torch.stack(policy_losses).sum() + \
                torch.stack(value_losses).sum()
            if torch.any(torch.isnan(loss)):
                print(loss)
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
            writer.add_scalar("Epsilon", epsilon, i_episode)

            if ep_reward > best:
                best = ep_reward
                with open(f"{outdir}/out_{best:.4f}_{i_episode}.papr", "w", encoding="utf-8") as f:
                    f.write(' '.join(env.script[1:-1]))
                torch.save(
                    model.cpu(), f"{outdir}/model_best.pt")
                torch.save(
                    optim, f"{outdir}/optim_best.pt")
                model.to(device)
            if running_reward > MAX_OBS_LEN * 0.9:
                print("Running reward reached threshold, stopping training")
                break
            i_episode += 1
    except KeyboardInterrupt:
        pass
    pbar.close()
    torch.save(model.cpu(), f"{outdir}/model.pt")
    torch.save(optim, f"{outdir}/optim.pt")
