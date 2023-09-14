#![feature(generic_const_exprs)]

use chrono::prelude::*;
use dfdx::{
    optim::{Adam, Sgd},
    prelude::*,
};
use papr_lib::{
    dsp::*,
    graph::{Connection, Input, Output},
    parser3::builtins::BuiltinNode,
};
use petgraph::{dot::Dot, prelude::DiGraph};
use rv::misc::ln_pflip;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs::File,
    ops::{Mul, Sub},
};
use tensorboard_rs::summary_writer::SummaryWriter;

pub type NodeIndex = usize;
// pub type FTensor<S> = Tensor<S, f32, Cpu>;

pub const MIN_VAR_INS: usize = 2;
pub const MAX_NODES: usize = 16;
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
            BuiltinNode::Add => 2,
            BuiltinNode::Sub => 2,
            BuiltinNode::Mul => 2,
            BuiltinNode::Div => 2,
            BuiltinNode::VariableInput => 0,
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
            BuiltinNode::Add => 1,
            BuiltinNode::Sub => 1,
            BuiltinNode::Mul => 1,
            BuiltinNode::Div => 1,
            BuiltinNode::VariableInput => 1,
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
    Add
    Sub
    Mul
    Div
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
    VariableInput
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
    Tensor<Rank5<NUM_NODE_TYPES, MAX_NODE_OUTS, MAX_NODE_INS, MAX_NODES, MAX_NODES>, f32, Cpu>;

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
            NODE_INDICES[&self.nodes[from]],
            conn.source_output,
            conn.sink_input,
            from,
            to,
            // NODE_INDICES[&self.nodes[to]],
        ]] = 1.0;

        Some(())
    }

    pub fn remove_edge(&mut self, from: NodeIndex, to: NodeIndex, conn: Connection) {
        self.adj_matrix[[
            NODE_INDICES[&self.nodes[from]],
            conn.source_output,
            conn.sink_input,
            from,
            to,
            // NODE_INDICES[&self.nodes[to]],
        ]] = 0.0;
    }

    pub fn node_inputs(&self, node: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        (0..self.nodes.len()).flat_map(move |a| {
            (0..MAX_NODE_OUTS).flat_map(move |b| {
                (0..MAX_NODE_INS).flat_map(move |c| {
                    if self.adj_matrix[[
                        NODE_INDICES[&self.nodes[a]],
                        b,
                        c,
                        a,
                        node,
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
                        NODE_INDICES[&self.nodes[node]],
                        b,
                        c,
                        node,
                        a,
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
                            if self.adj_matrix[[e, c, d, a, b]] > 0.0 {
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
    pub fn as_petgraph(&self) -> DiGraph<BuiltinNode, Connection> {
        let mut out = DiGraph::default();
        let mut mine_to_theirs = BTreeMap::new();
        for (i, node) in self.nodes.iter().enumerate().filter_map(|(i, node)| {
            if self.node_inputs(i).count() > 0 || self.node_outputs(i).count() > 0 {
                Some((i, node))
            } else {
                None
            }
        }) {
            mine_to_theirs.insert(i, out.add_node(*node));
        }
        for (from, to, conn) in self.all_edges() {
            if let (Some(from), Some(to)) = (mine_to_theirs.get(&from), mine_to_theirs.get(&to)) {
                out.add_edge(*from, *to, conn);
            }
        }
        out
    }
    pub fn to_embedding(
        &self,
        dev: &Cuda,
    ) -> Tensor<Rank3<FLAT_IO, MAX_NODES, MAX_NODES>, f32, Cuda> {
        self.adj_matrix.to_device(dev).reshape()
    }
    pub fn build(&self, name: Option<&str>) -> papr_lib::graph::Graph {
        let pg = self.as_petgraph();
        let mut n_ins = 0;
        let mut my_ins = vec![];
        let inputs = pg
            .node_indices()
            .filter(|node| matches!(pg[*node], BuiltinNode::VariableInput))
            .map(|node| {
                let inp = Input::new(&format!("in{n_ins}"), Some(Signal::new(0.0)));
                my_ins.push(node);
                n_ins += 1;
                inp
            })
            .collect();
        let mut n_outs = 0;
        let mut my_outs = vec![];
        let outputs = pg
            .externals(petgraph::Direction::Outgoing)
            .map(|node| {
                let out = Output {
                    name: format!("{:?}{n_outs}", pg[node]),
                };
                my_outs.push(node);
                n_outs += 1;
                out
            })
            .collect();
        let mut out = papr_lib::graph::Graph::new(name.map(Into::into), inputs, outputs);
        let mut my_node_to_out_node = BTreeMap::default();
        for (their_inp, my_inp) in out.input_node_indices.iter().zip(my_ins.iter()) {
            my_node_to_out_node.insert(*my_inp, *their_inp);
        }
        for (their_out, my_out) in out.output_node_indices.iter().zip(my_outs.iter()) {
            my_node_to_out_node.insert(*my_out, *their_out);
        }
        for my_i in pg.node_indices() {
            #[allow(clippy::map_entry)]
            if !my_node_to_out_node.contains_key(&my_i) {
                my_node_to_out_node.insert(
                    my_i,
                    out.add_node(pg[my_i].create_node(
                        &format!("{:?}{}", pg[my_i], my_node_to_out_node.len()),
                        pg[my_i].default_creation_args(),
                    )),
                );
            }
        }
        for edge in pg.raw_edges() {
            out.add_edge(
                my_node_to_out_node[&edge.source()],
                my_node_to_out_node[&edge.target()],
                edge.weight.clone(),
            );
        }

        out
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
        for _ in 0..MIN_VAR_INS {
            g.add_node(BuiltinNode::VariableInput);
        }
        for _ in 0..MAX_NODES - MIN_VAR_INS {
            let a = dev.random_u64() as usize % NUM_NODE_TYPES;
            g.add_node(NODE_TYPES[a]);
        }
        g
    }

    #[rustfmt::skip]
    type Actor = (
        Flatten2D,
        (Linear<{FLAT_IO * MAX_NODES * MAX_NODES}, 512>, ReLU),
        (Linear<512, 512>, ReLU),
        (Linear<512, {MAX_NODE_INS * MAX_NODE_OUTS * MAX_NODES * MAX_NODES}>,),
    );
    #[rustfmt::skip]
    type Critic = (
        Flatten2D,
        (Linear<{FLAT_IO * MAX_NODES * MAX_NODES}, 512>, ReLU),
        (Linear<512, 512>, ReLU),
        (Linear<512, 1>,),
    );

    type ActorCritic = SplitInto<(Actor, Critic)>;

    let mut model = dev.build_module::<ActorCritic, f32>();
    dbg!(model.num_trainable_params());
    let mut model_old = model.clone();

    let mut grads = model.alloc_grads();
    let mut opt = Adam::<_, f32, AutoDevice>::new(
        &model,
        AdamConfig {
            lr: 1e-3,
            ..Default::default()
        },
    );

    const EPSILON: f32 = 0.2;

    let mut buffer = RolloutBuffer::default();

    fn ln_softmax<S: Shape, Ax: Axes, T: Shape, Tp: Tape<f32, AutoDevice>>(
        t: Tensor<S, f32, AutoDevice, Tp>,
    ) -> Tensor<S, f32, AutoDevice, Tp>
    where
        S: ReduceShapeTo<T, Ax>,
    {
        let s = *t.shape();
        t.retaped::<Tp>() - t.exp().sum::<T, Ax>().ln().broadcast_like(&s)
    }

    let mut best_reward = 0.0;
    let mut i = 0;
    for episode in 0.. {
        let mut g = new_graph(&cpu);
        let mut total_reward = vec![];

        for j in 0..1000 {
            let state = g.to_embedding(&dev);
            let (action_logits, state_values) = model_old.forward(state.clone());
            let action_logits =
                action_logits
                    .reshape::<Rank3<{ MAX_NODE_INS * MAX_NODE_OUTS }, MAX_NODES, MAX_NODES>>();
            let action_probs = ln_softmax::<_, Axes3<0, 1, 2>, _, _>(action_logits.clone());
            let action_idx = ln_pflip(
                &action_probs
                    .as_vec()
                    .into_iter()
                    .map(|i| i as f64)
                    .collect::<Vec<_>>(),
                1,
                true,
                &mut rand::thread_rng(),
            )[0];
            let action = dev.tensor(action_idx);
            let action_logprob = action_probs
                .clone()
                .reshape_like(&(state.as_vec().len() / NUM_NODE_TYPES,))
                .select(action.clone());

            let mut reward = 0.0;
            let mut terminal = j == 999;

            let g_old = g.adj_matrix.clone();
            let mut g_new = g.clone();
            let [_sa, sb, sc, sd, se] = g.adj_matrix.shape().concrete();
            // let a = action_idx / (sb * sc * sd * se);
            let b = (action_idx / (sc * sd * se)) % sb;
            let c = (action_idx / (sd * se)) % sc;
            let d = (action_idx / (se)) % sd;
            let e = action_idx % se;
            if slice(g_new.adj_matrix.clone(), (0.., b..=b, c..=c, d..=d, e..=e))
                .as_vec()
                .iter()
                .any(|f| *f == 1.0)
            {
                reward -= 1.0;
                terminal = true;
            } else if g_new
                .add_edge(
                    d,
                    e,
                    Connection {
                        source_output: b,
                        sink_input: c,
                    },
                )
                .is_some()
            {
                reward += 1.0;
                if g_new.node_inputs(e).count() == g_new.nodes[e].num_inputs() {
                    reward += 1.0;
                }
                if g_new.node_outputs(d).count() >= g_new.nodes[d].num_outputs() {
                    reward += 1.0;
                }
                let pg = g_new.as_petgraph();
                reward -= pg
                    .externals(petgraph::Direction::Incoming)
                    .filter(|x| !matches!(pg[*x], BuiltinNode::VariableInput))
                    .count() as f32;
                let gg = g_new.build(Some("g_new"));
                reward *= gg.partitions.len() as f32;
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

            if i % 3000 == 0 {
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
                for _ in 0..80 {
                    let (action_logits, state_values) = model.forward(old_states.trace(grads));
                    let action_logits = action_logits.reshape_like(&(
                        buffer.states.len(),
                        MAX_NODE_INS * MAX_NODE_OUTS,
                        MAX_NODES,
                        MAX_NODES,
                    ));
                    let action_probs = ln_softmax::<_, Axes3<0, 1, 2>, _, _>(action_logits);
                    let action_logprob = action_probs
                        .reshape_like(&(
                            buffer.states.len(),
                            old_states.as_vec().len() / buffer.states.len() / NUM_NODE_TYPES,
                        ))
                        .select(old_actions.clone());

                    let ratios = (action_logprob - old_logprobs.clone()).exp();

                    let surr1 = ratios.with_empty_tape().mul(advantages.clone());
                    let surr2 = ratios
                        .clamp(1.0 - EPSILON, 1.0 + EPSILON)
                        .mul(advantages.clone());
                    let actor_loss = -surr2.minimum(surr1).mean();
                    let s = *state_values.shape();
                    let critic_loss =
                        mse_loss(state_values, rewards.clone().broadcast_like(&s)).mul(0.5);
                    // .sub(dist_entropy * 0.01);

                    let loss = actor_loss + critic_loss;
                    total_loss.push(loss.to_device(&cpu)[[]]);

                    grads = loss.backward();
                    opt.update(&mut model, &grads).unwrap();
                    model.zero_grads(&mut grads);
                }
                let loss = total_loss.iter().sum::<f32>() / total_loss.len() as f32;
                writer.add_scalar("Loss", loss, episode);
                println!("Ep {episode} Loss: {loss:#.4}");

                buffer.clear();

                model_old.clone_from(&model);
            }

            i += 1;

            if terminal {
                break;
            }
        }

        let reward = total_reward.iter().sum::<f32>();
        // println!("Episode {} reward = {:#4.4}", episode, reward);

        writer.add_scalar("Reward", reward, episode);
        if reward > best_reward {
            best_reward = reward;

            let mut f = File::create(format!("{outdir}/best_{reward:#4.4}_{episode}.dot")).unwrap();
            use std::io::Write;
            write!(f, "{:?}", Dot::new(&g.as_petgraph())).unwrap();

            let gg = g.build(Some("best"));
            gg.write_dot(&format!("{outdir}/best_built_{reward:#4.4}_{episode}.dot"));

            model
                .save_safetensors(format!("{outdir}/model.safetensors"))
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
