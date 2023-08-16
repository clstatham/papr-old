import numpy as np
from collections import defaultdict
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

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

# OBS_LEN = 1000
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
all_tokens += builtin_tokens
# audio_tokens = ["@" + tok for tok in builtin_tokens]
# control_tokens = ["#" + tok for tok in builtin_tokens]
# builtin_tokens = audio_tokens + control_tokens
SOT_TOKEN = "<|SOT|>"
EOT_TOKEN = "<|EOT|>"
NONE_TOKEN = "<|NONE|>"
extra_tokens = ["graph", "let",
                "=", ";",
                "{", "}", "[", "]", "<", ">",
                "+", "-", "*", "/",
                "&", "|", "^",
                "==", "!=",
                "@dac0", "#", "@", " ", EOT_TOKEN, SOT_TOKEN]

all_tokens += extra_tokens

MAX_VARS = 20
MAX_GRAPHS = 1

var_tokens = ["var" + str(i) for i in range(MAX_VARS)]
graph_tokens = ["Graph" + str(i) for i in range(MAX_GRAPHS)]

all_tokens += var_tokens + graph_tokens

all_tokens = [NONE_TOKEN] + all_tokens
assert all_tokens[0] == NONE_TOKEN


def encode(token):
    toks = torch.tensor(all_tokens.index(
        token), dtype=torch.int64)
    toks = F.one_hot(toks, num_classes=len(all_tokens))
    return toks.to(torch.float32)


class PaaiprEnv(EnvBase):
    metadata = {}
    batch_locked = False

    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _make_spec(self, td_params):
        self.observation_spec = CompositeSpec(
            state=OneHotDiscreteTensorSpec(
                len(all_tokens), device=self.device, dtype=torch.float32),
            params=make_composite_from_td(td_params["params"]),
            device=self.device,
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = OneHotDiscreteTensorSpec(
            len(all_tokens), device=self.device)
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(*td_params.shape, 1), device=self.device)
        self.done_spec = BinaryDiscreteTensorSpec(
            n=1, device=self.device, dtype=torch.bool)

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size=self.batch_size)

        self._script = [SOT_TOKEN]

        out = TensorDict(
            {
                "state": encode(self._script[-1]),
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out

    def _step(self, tensordict):
        state = tensordict["state"]
        action = torch.argmax(tensordict["action"].squeeze(-1))
        token = all_tokens[action]
        terminated = token == EOT_TOKEN
        reward = self.get_reward(token, terminated)
        reward = torch.full(tensordict.shape, reward).to(self.device)
        terminated = torch.full_like(
            reward, terminated, dtype=torch.bool).to(self.device)
        self._script += [token]
        out = TensorDict(
            {
                "next": {
                    "state": encode(self._script[-1]),
                    "params": tensordict["params"],
                    "reward": reward,
                    "done": terminated,
                }
            },
            batch_size=tensordict.shape,
        )
        return out

    @staticmethod
    def gen_params(batch_size=None):
        if batch_size is None:
            batch_size = []
        td = TensorDict({"params": TensorDict({}, [])}, [])
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    _reset = _reset
    _step = _step
    _set_seed = _set_seed

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()
        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._script = list()

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
        if finish:
            full_program = ''.join(self._script[1:])  # cut SOT token
            # full_program = f"""
            # graph Main {{
            #     || -> |@dac0|
            #     ~ {{
            #         {full_program}
            #     }}
            # }}
            # """
            print("Predicted program:")
            print(full_program)

            with open(f"training/full_program.papr", "w", encoding="utf-8") as f:
                f.write(full_program)
            result_full = subprocess.run([
                "../target/release/papr",
                f"training/full_program.papr",
                "--headless",
                "--run-for",
                "10000",
                "--out-path",
                f"training/full_program.wav",
            ], capture_output=True)

            if result_full.returncode == 0:
                reward += 1
                shutil.copyfile(
                    f"training/full_program.wav", f"training/success_{successful_renders}.wav")
                shutil.copyfile(
                    f"training/full_program.papr", f"training/success_{successful_renders}.papr")
                successful_renders += 1
            else:
                reward -= 1
            # print("Reward:", reward)
        return reward


if __name__ == "__main__":
    print("All Tokens:")
    print(all_tokens)

    device = "cuda" if torch.has_cuda else "cpu"
    num_cells = 256
    lr = 3e-4
    max_grad_norm = 1.0
    sub_batch_size = 64
    num_epochs = 10
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    env = PaaiprEnv(device=device)
    env = TransformedEnv(env, Compose(StepCounter(), InitTracker()))
    # env = TransformedEnv(env, Compose(
    #     # ObservationNorm(in_keys=["state"]),
    #     # DoubleToFloat(in_keys=["state"]),
    #     StepCounter()))
    # env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    # print("normalization constant shape:", env.transform[0].loc.shape)
    check_env_specs(env)
    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("done_spec:", env.done_spec)
    print("action_spec:", env.action_spec)
    print("state_spec:", env.state_spec)

    rollout = env.rollout(3)
    print("Rollout of 3 steps:", rollout)
    print("Shape of rollout tensordict:", rollout.batch_size)

    in_mlp = MLP(out_features=num_cells, num_cells=[num_cells])
    in_mlp = TensorDictModule(in_mlp, in_keys=["state"], out_keys=["embeds"])

    out_mlp = MLP(
        out_features=len(all_tokens), num_cells=[num_cells])
    out_mlp[-1].bias.data.fill_(0.0)
    out_mlp = TensorDictModule(
        out_mlp, in_keys=["embeds"], out_keys=["action_value"])
    lstm = LSTMModule(input_size=num_cells, hidden_size=num_cells,
                      in_key="embeds", out_key="embeds")
    env.append_transform(lstm.make_tensordict_primer())
    qval = QValueModule(action_space=env.action_spec)
    stoch_policy = TensorDictSequential(
        in_mlp,
        lstm,
        out_mlp,
        qval
    )
    stoch_policy = EGreedyWrapper(
        stoch_policy, annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2).to(device)
    policy = TensorDictSequential(
        in_mlp, lstm.set_recurrent_mode(True), out_mlp, qval).to(device)

    policy(env.reset())

    loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)
    updater = SoftUpdate(loss_fn, eps=0.95)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    collector = SyncDataCollector(
        env, stoch_policy, frames_per_batch=50, total_frames=1_000_000)
    rb = TensorDictReplayBuffer(storage=ListStorage(
        1_000), batch_size=BATCH_SIZE, prefetch=10)

    utd = 16
    pbar = tqdm(total=1_000_000)
    longest = 0
    try:
        for i, data in enumerate(collector):
            if i == 0:
                print(data)
            pbar.update(data.numel())
            rb.extend(data.clone().unsqueeze(0).cpu())
            for _ in tqdm(range(utd)):
                s = rb.sample().to(device)
                loss_vals = loss_fn(s)
                loss_vals["loss"].backward()
                optim.step()
                optim.zero_grad()
            longest = max(longest, data["step_count"].max().item())
            pbar.set_description(
                f"steps: {longest}, loss_val: {loss_vals['loss'].item(): 4.4f}")
            stoch_policy.step(data.numel())
            updater.step()

            if i % 50 == 0:
                with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                    rollout = env.rollout(10000, stoch_policy)
                    print("Max reward:", rollout.get(
                        ("next", "reward")).max().item())
    except KeyboardInterrupt:
        pass
    torch.save(stoch_policy.cpu(), "training/stoch_policy.pt")
    torch.save(policy.cpu(), "training/policy.pt")
    torch.save(optim, "training/optim.pt")
