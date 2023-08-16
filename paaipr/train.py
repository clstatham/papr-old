from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import sys
import subprocess
import re
import shutil

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

minibatch_size = 8
successful_renders = 0

checkpoint = "Salesforce/codegen2-1B"

os.makedirs("training", exist_ok=True)
os.makedirs("training_tokenizer", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# extra_tokens = [
#     "Sin",
#     "Cos",
#     "Exp",
#     "Tanh",
#     "Abs",
#     "SineFm",
#     "SineOsc",
#     "SawOsc",
#     "SquareOsc",
#     "Clock",
#     "Delay",
#     "Redge",
#     "Fedge",
#     "Var",
#     "Max",
#     "Min",
#     "Clip",
#     "If",
#     "Not",
# ]
# audio_tokens = ["@" + tok for tok in extra_tokens]
# control_tokens = ["#" + tok for tok in extra_tokens]
# extra_tokens = audio_tokens + control_tokens
# extra_tokens += ["let", "=", ";", "@dac0"]
# print(extra_tokens)
# tokenizer.add_tokens(extra_tokens)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype="auto").cuda()
# model.resize_token_embeddings(len(tokenizer))


class PaaiprEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):
        global successful_renders
        reward = [0] * len(predicted_list)
        # if finish:
        for i, pred in enumerate(predicted_list):
            flt = [p for p in pred if re.match(
                r"(<mask_\d+>|<sep>)", p) is None]
            if len(flt) > 0:
                with open(f"training_tokenizer/{i}.papr", "w", encoding="utf-8") as f:
                    f.write(flt[-1])
                result_tokens = subprocess.run(
                    ["../target/release/papr-tokenizer.exe", f"training_tokenizer/{i}.papr"], capture_output=True)
                reward[i] += int(
                    str(result_tokens.stdout.strip(), encoding="utf-8"))
                if finish:
                    full_program = ' '.join(
                        flt)
                    full_program = f"""
                    graph Main {{
                        || -> |@dac0|
                        ~ {{
                            {full_program}
                        }}
                    }}
                    """
                    print("Predicted program:")
                    print(full_program)

                    with open(f"training/{i}.papr", "w", encoding="utf-8") as f:
                        f.write(full_program)
                    result_full = subprocess.run([
                        "../target/release/papr.exe",
                        f"training/{i}.papr",
                        "--headless",
                        "--run-for",
                        "10000",
                        "--out-path",
                        f"training/{i}.wav",
                    ], capture_output=True)

                # print(result.stdout)
                # print(result.stderr)

                    if result_full.returncode == 0:
                        reward[i] += 1000
                        shutil.copyfile(
                            f"training/{i}.wav", f"training/success_{successful_renders}.wav")
                        shutil.copyfile(
                            f"training/{i}.papr", f"training/success_{successful_renders}.papr")

                        successful_renders += 1

                # reward[i] /= 100.0
            print(reward[i])
        return reward


observation_list = [
    {"input": """// Create a generative musical piece\n"""}
]
env = PaaiprEnv(model, tokenizer,
                observation_input=observation_list, compare_sample=minibatch_size, max_length=100)
actor = TextRLActor(env, model, tokenizer, act_deterministically=False,
                    temperature=1.0, top_k=0.0, top_p=1.0)
agent = actor.agent_ppo(
    update_interval=10, minibatch_size=minibatch_size, epochs=20)
# print(actor.predict(observation_list[0]))
# agent.load("training/best")
train_agent_with_evaluation(agent, env, steps=1000000, eval_n_steps=None,
                            eval_n_episodes=1, eval_interval=10, outdir='training', use_tensorboard=True)

print(actor.predict(observation_list[0]))
