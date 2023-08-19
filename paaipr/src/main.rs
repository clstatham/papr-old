#![feature(generic_const_exprs)]

use chrono::prelude::*;
use dfdx::{optim::Adam, prelude::*};
use papr_lib::{dsp::*, graph::Connection, parser3::builtins::BuiltinNode};
use petgraph::{dot::Dot, prelude::DiGraph};
use rv::{
    prelude::Categorical,
    traits::{Entropy, Rv},
};
use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    ops::{Mul, Sub},
    time::SystemTime,
};
use tensorboard_rs::summary_writer::SummaryWriter;

pub type NodeIndex = usize;
// pub type FTensor<S> = Tensor<S, f32, Cpu>;

pub const MAX_NODES: usize = 10;
pub const MAX_NODE_INS: usize = 2;
pub const MAX_NODE_OUTS: usize = 2;

pub trait NumInputs {
    fn num_inputs(&self) -> usize;
}

pub trait NumOutputs {
    fn num_outputs(&self) -> usize;
}

impl NumInputs for BuiltinNode {
    fn num_inputs(&self) -> usize {
        match self {
            BuiltinNode::Abs => 1,
            BuiltinNode::Sine => 1,
            BuiltinNode::Cosine => 1,
            BuiltinNode::Tanh => 1,
            BuiltinNode::Exp => 1,
            BuiltinNode::FmSineOsc => generators::FmSineOsc::INPUTS.len() - 1,
            BuiltinNode::SineOsc => generators::SineOsc::INPUTS.len() - 1,
            BuiltinNode::BlSawOsc => generators::BlSawOsc::INPUTS.len() - 1,
            BuiltinNode::BlSquareOsc => generators::BlSquareOsc::INPUTS.len() - 1,
            BuiltinNode::MidiToFreq => midi::MidiToFreq::INPUTS.len() - 1,
            BuiltinNode::Clock => time::Clock::INPUTS.len() - 1,
            BuiltinNode::Delay => time::Delay::INPUTS.len() - 1,
            BuiltinNode::NoteIn => todo!(),
            BuiltinNode::RisingEdge => basic::RisingEdge::INPUTS.len() - 1,
            BuiltinNode::FallingEdge => basic::FallingEdge::INPUTS.len() - 1,
            BuiltinNode::Var => graph_util::Var::INPUTS.len() - 1,
            BuiltinNode::Max => basic::Max::INPUTS.len() - 1,
            BuiltinNode::Min => basic::Min::INPUTS.len() - 1,
            BuiltinNode::Clip => basic::Clip::INPUTS.len() - 1,
            BuiltinNode::Debug => todo!(),
            BuiltinNode::If => basic::If::INPUTS.len() - 1,
            BuiltinNode::Not => 1,
            BuiltinNode::Sample => todo!(),
            BuiltinNode::Slider => 0,
            BuiltinNode::Button => 0,
            BuiltinNode::Toggle => 0,
            BuiltinNode::Led => 1,
            BuiltinNode::Constant => 0,
            BuiltinNode::Dac0 => 1,
        }
    }
}

impl NumOutputs for BuiltinNode {
    fn num_outputs(&self) -> usize {
        match self {
            BuiltinNode::Abs => 1,
            BuiltinNode::Sine => 1,
            BuiltinNode::Cosine => 1,
            BuiltinNode::Tanh => 1,
            BuiltinNode::Exp => 1,
            BuiltinNode::FmSineOsc => generators::FmSineOsc::OUTPUTS.len() - 1,
            BuiltinNode::SineOsc => generators::SineOsc::OUTPUTS.len() - 1,
            BuiltinNode::BlSawOsc => generators::BlSawOsc::OUTPUTS.len() - 1,
            BuiltinNode::BlSquareOsc => generators::BlSquareOsc::OUTPUTS.len() - 1,
            BuiltinNode::MidiToFreq => midi::MidiToFreq::OUTPUTS.len() - 1,
            BuiltinNode::Clock => time::Clock::OUTPUTS.len() - 1,
            BuiltinNode::Delay => time::Delay::OUTPUTS.len() - 1,
            BuiltinNode::NoteIn => todo!(),
            BuiltinNode::RisingEdge => basic::RisingEdge::OUTPUTS.len() - 1,
            BuiltinNode::FallingEdge => basic::FallingEdge::OUTPUTS.len() - 1,
            BuiltinNode::Var => graph_util::Var::OUTPUTS.len() - 1,
            BuiltinNode::Max => basic::Max::OUTPUTS.len() - 1,
            BuiltinNode::Min => basic::Min::OUTPUTS.len() - 1,
            BuiltinNode::Clip => basic::Clip::OUTPUTS.len() - 1,
            BuiltinNode::Debug => todo!(),
            BuiltinNode::If => basic::If::OUTPUTS.len() - 1,
            BuiltinNode::Not => 1,
            BuiltinNode::Sample => todo!(),
            BuiltinNode::Slider => 1,
            BuiltinNode::Button => 1,
            BuiltinNode::Toggle => 1,
            BuiltinNode::Led => 0,
            BuiltinNode::Constant => 1,
            BuiltinNode::Dac0 => 0,
        }
    }
}

macro_rules! include_nodes {
    ($($n:ident) *) => {
        pub const NODE_TYPES: &[BuiltinNode] = &[$(BuiltinNode::$n),*];
        pub const NUM_NODE_TYPES: usize = NODE_TYPES.len();
    };
}

include_nodes!(
    Abs
    Sine
    Cosine
    Tanh
    Exp
    // FmSineOsc
    SineOsc
    BlSawOsc
    BlSquareOsc
    // Clock
    // Delay
    // RisingEdge
    // FallingEdge
    // Var
    Max
    Min
    // Clip
    // If
    // Not
    // Slider
    // Button
    // Toggle
    // Led
    // Constant
    // Dac0
);

lazy_static::lazy_static! {
    static ref NODE_INDICES: HashMap<BuiltinNode, usize> = NODE_TYPES.iter().enumerate().map(|(i, n)| (*n, i)).collect();
}

pub type AdjMatrix =
    Tensor<Rank5<MAX_NODE_OUTS, MAX_NODE_INS, MAX_NODES, MAX_NODES, NUM_NODE_TYPES>, f32, Cpu>;

#[derive(Clone)]
pub struct Graph {
    nodes: Vec<BuiltinNode>,
    pub adj_matrix: AdjMatrix,
}

impl Graph {
    pub fn new(dev: &Cpu) -> Self {
        Self {
            nodes: Vec::new(),
            adj_matrix: dev.zeros(),
        }
    }
    pub fn add_node(&mut self, node: BuiltinNode) -> NodeIndex {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    #[must_use = "Adding an edge can fail if the target node has no more inputs available"]
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, conn: Connection) -> Option<()> {
        if from == to {
            // acyclic graphs only for now
            return None;
        }
        if conn.source_output >= self.nodes[from].num_outputs() {
            return None;
        }
        if conn.sink_input >= self.nodes[to].num_inputs() {
            return None;
        }
        if self.node_inputs(to).count() >= self.nodes[to].num_inputs() {
            return None;
        }

        self.adj_matrix[[
            conn.source_output,
            conn.sink_input,
            from,
            to,
            NODE_INDICES[&self.nodes[from]],
            // NODE_INDICES[&self.nodes[to]],
        ]] = 1.0;

        Some(())
    }

    pub fn remove_edge(&mut self, from: NodeIndex, to: NodeIndex, conn: Connection) {
        self.adj_matrix[[
            conn.source_output,
            conn.sink_input,
            from,
            to,
            NODE_INDICES[&self.nodes[from]],
            // NODE_INDICES[&self.nodes[to]],
        ]] = 0.0;
    }

    pub fn node_inputs(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        (0..self.nodes.len()).flat_map(move |a| {
            (0..MAX_NODE_OUTS).flat_map(move |b| {
                (0..MAX_NODE_INS).flat_map(move |c| {
                    if self.adj_matrix[[
                        b,
                        c,
                        a,
                        node,
                        NODE_INDICES[&self.nodes[a]],
                        // NODE_INDICES[&self.nodes[node]],
                    ]] > 0.0
                    {
                        Some(a)
                    } else {
                        None
                    }
                })
            })
        })
    }
    pub fn node_outputs(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        (0..self.nodes.len()).flat_map(move |a| {
            (0..MAX_NODE_OUTS).flat_map(move |b| {
                (0..MAX_NODE_INS).flat_map(move |c| {
                    if self.adj_matrix[[
                        b,
                        c,
                        node,
                        a,
                        NODE_INDICES[&self.nodes[node]],
                        // NODE_INDICES[&self.nodes[a]],
                    ]] > 0.0
                    {
                        Some(a)
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn all_edges(&self) -> impl Iterator<Item = (NodeIndex, NodeIndex, Connection)> + '_ {
        (0..self.nodes.len()).flat_map(move |a| {
            (0..self.nodes.len()).flat_map(move |b| {
                (0..MAX_NODE_OUTS).flat_map(move |c| {
                    (0..MAX_NODE_INS).flat_map(move |d| {
                        (0..NUM_NODE_TYPES).flat_map(move |e| {
                            // (0..NUM_NODE_TYPES).flat_map(move |f| {
                            if self.adj_matrix[[c, d, a, b, e]] > 0.0 {
                                Some((
                                    a,
                                    b,
                                    Connection {
                                        source_output: c,
                                        sink_input: d,
                                    },
                                ))
                            } else {
                                None
                            }
                            // })
                        })
                    })
                })
            })
        })
    }
    pub fn is_valid(&self) -> bool {
        for (i, node) in self.nodes.iter().enumerate() {
            if self.node_inputs(i).count() > node.num_inputs() {
                return false;
            }
        }

        true
    }
    pub fn as_petgraph(&self) -> DiGraph<BuiltinNode, Connection, usize> {
        let mut out = DiGraph::default();
        for node in self.nodes.iter() {
            out.add_node(*node);
        }
        for (from, to, conn) in self.all_edges() {
            out.add_edge(from.into(), to.into(), conn);
        }
        out
    }
    pub fn to_embedding(
        &self,
        dev: &Cuda,
    ) -> Tensor<Rank3<FLAT_IO, MAX_NODES, MAX_NODES>, f32, Cuda> {
        self.adj_matrix.to_device(dev).reshape()
    }
}

const FLAT_IO: usize = MAX_NODE_INS * MAX_NODE_OUTS * NUM_NODE_TYPES;

#[derive(Default)]
pub struct RolloutBuffer {
    pub actions: Vec<Tensor<(), usize, AutoDevice>>,
    pub states: Vec<Tensor<Rank3<FLAT_IO, MAX_NODES, MAX_NODES>, f32, AutoDevice>>,
    pub logprobs: Vec<Tensor<(), f32, AutoDevice>>,
    pub rewards: Vec<f32>,
    pub state_values: Vec<f32>,
    pub terminals: Vec<bool>,
}

impl RolloutBuffer {
    pub fn clear(&mut self) {
        self.actions.clear();
        self.states.clear();
        self.logprobs.clear();
        self.rewards.clear();
        self.state_values.clear();
        self.terminals.clear();
    }
}

fn main() {
    dfdx::flush_denormals_to_zero();

    let cpu = Cpu::seed_from_u64(rand::random());
    let dev = AutoDevice::seed_from_u64(rand::random());

    let stamp = Local::now().format("%Y%m%d%H%M%S");
    let outdir = format!("training/{stamp}");
    std::fs::create_dir_all(&outdir).unwrap();

    let mut writer = SummaryWriter::new(&outdir);

    fn new_graph(dev: &Cpu) -> Graph {
        let mut g = Graph::new(dev);
        for _ in 0..MAX_NODES {
            let a = dev.random_u64() as usize % NUM_NODE_TYPES;
            g.add_node(NODE_TYPES[a]);
        }
        g
    }

    #[rustfmt::skip]
    type Actor = (
        (Conv2D<FLAT_IO, 32, 3, 1, 1>, ReLU),
        (Conv2D<32, 64, 3, 1, 1>, ReLU),
        (Conv2D<64, 32, 3, 1, 1>, ReLU),
        (Conv2D<32, FLAT_IO, 3, 1, 1>,),
    );
    #[rustfmt::skip]
    type Critic = (
        (Conv2D<FLAT_IO, 32, 3, 1, 1>, ReLU),
        (Conv2D<32, 64, 3, 1, 1>, ReLU),
        (Conv2D<64, 32, 3, 1, 1>, ReLU),
        (Conv2D<32, 1, 3, 1, 1>, ReLU),
        Flatten2D,
        Linear<100, 1>,
    );

    let mut actor = dev.build_module::<Actor, f32>();
    let mut critic = dev.build_module::<Critic, f32>();
    let mut actor_old = actor.clone();
    let mut critic_old = critic.clone();

    let mut actor_grads = actor.alloc_grads();
    let mut critic_grads = critic.alloc_grads();
    let mut actor_opt = Adam::<_, f32, AutoDevice>::new(
        &actor,
        AdamConfig {
            lr: 1e-4,
            ..Default::default()
        },
    );
    let mut critic_opt = Adam::<_, f32, AutoDevice>::new(
        &critic,
        AdamConfig {
            lr: 1e-4,
            ..Default::default()
        },
    );

    const EPSILON: f32 = 0.1;

    let mut buffer = RolloutBuffer::default();

    let mut best_reward = 0.0;
    let mut i = 0;
    for episode in 0.. {
        let mut g = new_graph(&cpu);
        let mut total_reward = vec![];

        for _ in 0..1000 {
            let state = g.to_embedding(&dev);
            let action_probs = actor_old.forward(state.clone()); // .trace(actor_grads)
            let dist = Categorical::new(
                action_probs
                    .clone()
                    .softmax::<Axes3<0, 1, 2>>()
                    .as_vec()
                    .into_iter()
                    .map(|i| i as f64)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap();
            let action_idx: usize = dist.sample(1, &mut rand::thread_rng())[0];
            let action = dev.tensor(action_idx);
            let action_logprob = action_probs
                .clone()
                .log_softmax::<Axes3<0, 1, 2>>()
                .reshape_like(&(state.as_vec().len(),))
                .select(action.clone());
            let state_values = critic_old.forward(state.clone());

            let mut reward = 0.0;
            let mut terminal = i == 99;

            let g_old = g.adj_matrix.clone();
            let mut g_new = g.clone();
            let [_sa, sb, sc, sd, se] = g.adj_matrix.shape().concrete();
            let a = action_idx / (sb * sc * sd * se);
            let b = (action_idx / (sc * sd * se)) % sb;
            let c = (action_idx / (sd * se)) % sc;
            let d = (action_idx / (se)) % sd;
            // let e = action_idx % se;
            if slice(
                g_new.adj_matrix.clone(),
                (a..=a, b..=b, c..=c, d..=d, 0..se),
            )
            .as_vec()
            .iter()
            .any(|f| *f == 1.0)
            {
                reward -= 1.0;
                terminal = true;
            } else if g_new
                .add_edge(
                    c,
                    d,
                    Connection {
                        source_output: a,
                        sink_input: b,
                    },
                )
                .is_some()
            {
                reward += 1.0;
                if g_new.node_inputs(d).count() == g_new.nodes[d].num_inputs() {
                    reward += 1.0;
                }
                if g_new.node_outputs(c).count() >= g_new.nodes[c].num_outputs() {
                    reward += 1.0;
                }
            } else {
                g_new.adj_matrix = g_old;
                reward -= 1.0;
                terminal = true;
            }
            g = g_new;

            total_reward.push(reward);

            buffer.states.push(state);
            buffer.actions.push(action);
            buffer.logprobs.push(action_logprob);
            buffer.rewards.push(reward);
            buffer.state_values.push(state_values.array()[0]);
            buffer.terminals.push(terminal);

            if i % 100 == 0 {
                let mut rewards = VecDeque::new();
                let mut discounted_reward = 0.0;
                for (reward, terminal) in buffer
                    .rewards
                    .iter()
                    .rev()
                    .zip(buffer.terminals.iter().rev())
                {
                    if *terminal {
                        discounted_reward = 0.0;
                    }
                    discounted_reward = reward + 0.99 * discounted_reward;
                    rewards.push_front(discounted_reward);
                }

                let rewards = dev.tensor_from_vec(rewards.clone().into(), (rewards.len(),));
                let rewards = rewards.normalize(1e-7);

                let old_states = buffer.states.clone().stack();
                let old_actions = buffer.actions.clone().stack();
                let old_logprobs = buffer.logprobs.clone().stack();
                let old_state_values =
                    dev.tensor_from_vec(buffer.state_values.clone(), (buffer.state_values.len(),));

                let advantages = rewards.clone().sub(old_state_values.clone());
                let mut total_loss = vec![];
                for _ in 0..100 {
                    let action_probs = actor.forward(old_states.trace(actor_grads)); // .trace(actor_grads)
                    let dist = Categorical::new(
                        action_probs
                            .with_empty_tape()
                            .softmax::<Axes3<0, 1, 2>>()
                            .as_vec()
                            .into_iter()
                            .map(|i| i as f64)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                    .unwrap();
                    let action_logprob = action_probs
                        .log_softmax::<Axes3<1, 2, 3>>()
                        .reshape_like(&(
                            buffer.states.len(),
                            old_states.as_vec().len() / buffer.states.len(),
                        ))
                        .select(old_actions.clone());

                    let dist_entropy = dist.entropy() as f32;
                    let state_values = critic.forward(old_states.clone().trace(critic_grads));

                    let ratios = (action_logprob - old_logprobs.clone()).exp();

                    let surr1 = ratios.with_empty_tape().mul(advantages.clone());
                    let surr2 = ratios
                        .clamp(1.0 - EPSILON, 1.0 + EPSILON)
                        .mul(advantages.clone());
                    let actor_loss = -surr2.minimum(surr1).mean();
                    let s = *state_values.shape();
                    let critic_loss = mse_loss(state_values, rewards.clone().broadcast_like(&s))
                        .mul(0.5)
                        .sub(dist_entropy * 0.01);

                    let loss = actor_loss.to_device(&cpu)[[]] + critic_loss.to_device(&cpu)[[]];
                    total_loss.push(loss);
                    actor_grads = actor_loss.backward();
                    critic_grads = critic_loss.backward();
                    actor_opt.update(&mut actor, &actor_grads).unwrap();
                    critic_opt.update(&mut critic, &critic_grads).unwrap();
                    actor.zero_grads(&mut actor_grads);
                    critic.zero_grads(&mut critic_grads);
                }
                let loss = total_loss.iter().sum::<f32>() / total_loss.len() as f32;
                writer.add_scalar("Loss", loss, episode);
                println!("Loss: {loss:#.4}");

                buffer.clear();

                actor_old.clone_from(&actor);
                critic_old.clone_from(&critic);
            }

            i += 1;

            if terminal {
                break;
            }
        }

        let reward = total_reward.iter().sum::<f32>();
        println!("Episode {} reward = {:#4.4}", episode, reward);

        writer.add_scalar("Reward", reward, episode);
        if reward > best_reward {
            let mut f = File::create(format!("{outdir}/best_{reward:#4.4}_{episode}.dot")).unwrap();
            use std::io::Write;
            write!(f, "{:?}", Dot::new(&g.as_petgraph())).unwrap();
            best_reward = reward;

            actor
                .save_safetensors(format!("{outdir}/actor.safetensors"))
                .unwrap();
            critic
                .save_safetensors(format!("{outdir}/critic.safetensors"))
                .unwrap();
        }
        // if episode % 1000 == 0 {
        //     let mut f = File::create(format!("{outdir}/out_{total_reward}_{episode}.dot")).unwrap();
        //     use std::io::Write;
        //     write!(f, "{:?}", Dot::new(&g.as_petgraph())).unwrap();
        // }
    }
    writer.flush();
}
