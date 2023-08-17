import numpy as np
from collections import defaultdict, namedtuple
from tensordict import TensorDict
from tensordict.nn import *
from tensordict.nn.distributions import *
from torchrl.collectors import *
from torchrl.objectives.value import *
from torchrl.objectives import *
from torchrl.modules import *
from torchrl.envs.utils import *
from torchrl.envs import *
from torchrl.data import *
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

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

MAX_OBS_LEN = 200
BATCH_SIZE = 16

minibatch_size = 8
successful_renders = 0

checkpoint = "Salesforce/codegen2-1B"

os.makedirs("training", exist_ok=True)
os.makedirs("training_tokenizer", exist_ok=True)

all_tokens = []

builtin_tokens = [
    "Sin",
    "Cos",
    "Exp",
    "Tanh",
    "Abs",
    "SineFm",
    "SineOsc",
    "SawOsc",
    "SquareOsc",
    "Clock",
    "Delay",
    "Redge",
    "Fedge",
    "Var",
    "Max",
    "Min",
    "Clip",
    "If",
    "Not",
]
audio_tokens = ["@" + tok for tok in builtin_tokens]
control_tokens = ["#" + tok for tok in builtin_tokens]
builtin_tokens += audio_tokens + control_tokens
all_tokens += builtin_tokens
SOT_TOKEN = "<|START|>"
EOT_TOKEN = "<|END|>"
operator_tokens = [  # "=", ";",
    #    "{", "}", "[", "]", "<", ">", "(", ")",
    "+", "-", "*", "/",
                   "&", "|", "^",
                   "==", "!="]
keyword_tokens = []

all_tokens += keyword_tokens + operator_tokens

MAX_VARS = 3
MAX_GRAPHS = 1

var_tokens = ["@var" + str(i) for i in range(MAX_VARS)]
var_tokens += ["#var" + str(i) for i in range(MAX_VARS, MAX_VARS * 2)]
# graph_tokens = ["#Graph" + str(i) for i in range(MAX_GRAPHS)]
# graph_tokens += ["@Graph" + str(i) for i in range(MAX_GRAPHS)]
var_tokens += ["@dac0"]
all_tokens += var_tokens

# with open("builtin_tokens.txt", "w") as f:
#     f.write(' '.join(builtin_tokens))
# with open("operator_tokens.txt", "w") as f:
#     f.write(' '.join(operator_tokens))
# with open("keyword_tokens.txt", "w") as f:
#     f.write(' '.join(keyword_tokens))
# with open("var_tokens.txt", "w") as f:
#     f.write(' '.join(var_tokens))
valid_after_assign_tokens = var_tokens + builtin_tokens
valid_after_var_tokens = ["="]
valid_after_graph_tokens = ["("]

valid_token_storage = {
    "valid_after_var_tokens": valid_after_var_tokens,
    "valid_after_assign_tokens": valid_after_assign_tokens,
}

valid_next_tokens = {
    # valid after SOT token
    SOT_TOKEN: var_tokens,
    # valid after a var token
    **{v: ["="] + var_tokens + operator_tokens for v in var_tokens},
    # valid after a = token
    "=": var_tokens + builtin_tokens + ["("],
    # valid after an operator token
    **{o: var_tokens for o in operator_tokens},
    # valid after a builtin token
    **{b: ["("] for b in builtin_tokens},
    # valid after a ( token
    "(": [")"] + var_tokens,
    # valid after a ")" token
    ")": [";"] + var_tokens,
    # valid after a ";" token
    ";": [EOT_TOKEN] + var_tokens,
    EOT_TOKEN: None,
}
print(valid_next_tokens)

all_tokens = [SOT_TOKEN] + all_tokens + [EOT_TOKEN]
assert all_tokens[0] == SOT_TOKEN


def encode(tokens):
    toks = [torch.tensor(all_tokens.index(
        token), dtype=torch.int64) for token in tokens]
    toks = [F.one_hot(token, num_classes=len(all_tokens)) for token in toks]
    return torch.stack(toks).to(torch.float32)


def decode(last_token: str, logits: torch.Tensor):
    logits = logits.softmax(-1)
    valid_next = valid_next_tokens[last_token]
    valid_next_indices = [all_tokens.index(v) for v in valid_next]
    am = logits.argmax().item()
    i = 1
    while am not in valid_next_indices:
        i += 1
        am = torch.topk(logits, i).indices[-1].item()
    token = all_tokens[am]
    return token, i


class PaaiprEnv(object):
    def reset(self):
        self.script = [SOT_TOKEN]
        return encode(self.script)

    def step(self, action, penalty):
        token = all_tokens[action.squeeze(-1).item()]
        terminated = token == EOT_TOKEN
        self.script += [token]
        # if len(self.script) > MAX_OBS_LEN:
        #     terminated = True
        reward = self.get_reward(token, terminated)
        reward = torch.tensor(reward).to(self.device)
        reward -= penalty / len(all_tokens)
        # terminated = torch.full_like(
        #     reward, terminated, dtype=torch.bool).to(self.device)

        return encode(self.script), reward, terminated

    def __init__(self, device="cpu"):
        self.script = list()
        self.device = device

    def get_reward(self, new_token, finish):
        global successful_renders
        reward = 0.0

        # if new_token == EOT_TOKEN:
        # with open(f"training_tokenizer/token.papr", "w", encoding="utf-8") as f:
        #     f.write(new_token)
        # result_tokens = subprocess.run(
        #     ["../target/release/papr-tokenizer", f"training_tokenizer/token.papr"], capture_output=True)
        # reward += int(
        #     str(result_tokens.stdout.strip(), encoding="utf-8"))
        # if finish:
        script = list(filter(lambda x: x not in [
                      SOT_TOKEN, EOT_TOKEN], self.script))
        reward -= len(self.script) - len(script)
        if len(script) > 0:
            full_program = ' '.join(script)
            full_program = f"""
            graph Main {{
                |{' '.join(var_tokens)}| -> |@dac0|
                ~ {{
                    {full_program}
                }}
            }}
            """
            # print("Predicted program:")
            # print(full_program)
            with open(f"training/full_program.papr", "w", encoding="utf-8") as f:
                f.write(full_program)

            result_tokens = subprocess.run(
                ["../target/release/papr-tokenizer", f"training/full_program.papr"], capture_output=True)
            reward += int(
                str(result_tokens.stdout.strip(), encoding="utf-8"))
            reward -= len(var_tokens)

            if finish:
                result_full = subprocess.run([
                    "../target/release/papr",
                    f"training/full_program.papr",
                    "--headless",
                    "--run-for",
                    "1000",
                    "--out-path",
                    f"training/full_program.wav",
                ], capture_output=True)

                if result_full.returncode == 0:
                    reward += 100
                    shutil.copyfile(
                        f"training/full_program.wav", f"training/success_{successful_renders}.wav")
                    shutil.copyfile(
                        f"training/full_program.papr", f"training/success_{successful_renders}.papr")
                    successful_renders += 1

            # print("Reward:", reward)
        reward /= MAX_OBS_LEN
        return reward


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
if __name__ == "__main__":
    print("All Tokens:")
    print(all_tokens)

    device = "cuda" if torch.has_cuda else "cpu"
    num_cells = 1024
    lr = 3e-4

    env = PaaiprEnv(device=device)

    class Policy(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.lstm1 = nn.LSTM(len(all_tokens), num_cells,
                                 batch_first=True, num_layers=2)
            self.action_head = nn.Linear(num_cells, len(all_tokens))
            self.value_head = nn.Linear(num_cells, 1)

            self.saved_actions = []
            self.rewards = []

        def forward(self, x):
            x = x.unsqueeze(0)
            x, _ = self.lstm1(x)
            x = x[..., -1, :]
            action_prob = self.action_head(x)
            action_prob_softmax = F.softmax(action_prob, dim=-1)
            state_values = self.value_head(x)
            return action_prob_softmax.squeeze(0), state_values.squeeze(0)

    model = Policy()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # collector = SyncDataCollector(
    #     env, policy, frames_per_batch=50, total_frames=1_000_000)
    # rb = TensorDictReplayBuffer(storage=ListStorage(
    #     10_000), batch_size=BATCH_SIZE, prefetch=10)

    writer = SummaryWriter("./training")
    nepoch = 100000000
    pbar = tqdm(total=nepoch)
    try:
        for i_episode in range(nepoch):
            state = env.reset()
            for t in range(MAX_OBS_LEN):
                probs, state_value = model(state)
                m = Categorical(probs)
                action = m.sample()
                # probs = torch.argsort(probs, -1, descending=True)
                # action = probs[..., 0]
                penalty = 0
                # while all_tokens[action.squeeze(-1).item()] not in valid_next_tokens[env.script[-1]]:
                #     penalty += 1
                #     action = m.sample()
                # action = probs[..., penalty]
                model.saved_actions.append(
                    SavedAction(m.log_prob(action), state_value))
                state, reward, done = env.step(action=action, penalty=penalty)
                model.rewards.append(reward)
                if done:
                    break

            R = 0
            returns = []
            for r in model.rewards[::-1]:
                R = r + 0.99 * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            policy_losses = []
            value_losses = []
            for (log_prob, value), R in zip(model.saved_actions, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            optim.zero_grad()
            loss = torch.stack(policy_losses).sum() + \
                torch.stack(value_losses).sum()
            loss.backward()
            optim.step()

            mean_reward = torch.stack(model.rewards).mean().item()
            max_reward = torch.stack(model.rewards).max().item()

            del model.rewards[:]
            del model.saved_actions[:]

            pbar.set_description(
                f"reward_mean: {mean_reward}, reward_max: {max_reward}, loss: {loss.item(): 4.4f}")
            pbar.update()

            writer.add_scalar("Reward/Mean", mean_reward, i_episode)
            writer.add_scalar("Reward/Max", max_reward, i_episode)
            writer.add_scalar("Loss", loss.item(), i_episode)

            # if i % 50 == 0:
            #     with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            #         rollout = env.rollout(1000, stoch_policy)
            #         print("\nNet reward:", rollout.get(
            #             ("next", "reward")).sum().item())
    except KeyboardInterrupt:
        pass
    torch.save(model.cpu(), "training/model.pt")
    torch.save(optim, "training/optim.pt")
