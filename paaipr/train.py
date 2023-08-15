import torch
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

checkpoint = "mrm8488/llama-2-coder-7b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.add_tokens([
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
], special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

model = model.cuda()
model.resize_token_embeddings(len(tokenizer))


class PaaiprEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish):
        reward = [0]
        if finish:
            reward = [1]
            print(predicted_list)
        return reward


observation_list = [
    {"input": "You are a coding assistant that will help the user to resolve the following instruction:\n### Instruction: Create a generative musical piece.\n\n### Solution:\n"}]
env = TextRLEnv(model, tokenizer,
                observation_input=observation_list, compare_sample=2)
actor = TextRLActor(env, model, tokenizer, act_deterministically=False,
                    temperature=1.0, top_k=0.0, top_p=1.0)
agent = actor.agent_ppo(update_interval=2, minibatch_size=2, epochs=10)
print(actor.predict(observation_list[0]))

train_agent_with_evaluation(agent, env, steps=100, eval_n_steps=None,
                            eval_n_episodes=1, eval_interval=2, outdir='/home/cls/paaipr')

print(actor.predict(observation_list[0]))
