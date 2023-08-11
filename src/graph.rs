use std::{
    collections::VecDeque,
    sync::{Arc, RwLock},
};

use petgraph::{
    dot::{Config, Dot},
    prelude::*,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    dsp::{
        graph_util::{GraphInput, GraphOutput},
        AudioRate, ControlRate, Processor, Signal, SignalRate,
    },
    Scalar,
};

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, derive_more::Display, derive_more::Into, PartialOrd, Ord,
)]
pub struct NodeName(String);

impl NodeName {
    pub fn new(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl Default for NodeName {
    fn default() -> Self {
        Self("node".to_owned())
    }
}

impl From<String> for NodeName {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for NodeName {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl AsRef<str> for NodeName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Connection {
    pub source_output: usize,
    pub sink_input: usize,
}

#[derive(Clone)]
#[non_exhaustive]
pub struct Input<T: SignalRate> {
    pub name: String,
    pub minimum: Option<Signal<T>>,
    pub maximum: Option<Signal<T>>,
    pub default: Option<Signal<T>>,
    pub implicit: bool,
    pub is_ui: bool,
}

impl<T: SignalRate> Input<T> {
    pub fn new(name: &str, default: Option<Signal<T>>) -> Self {
        Self {
            name: name.to_owned(),
            minimum: None,
            maximum: None,
            default,
            implicit: false,
            is_ui: false,
        }
    }

    pub fn new_ui(name: &str, minimum: Signal<T>, maximum: Signal<T>, default: Signal<T>) -> Self {
        Self {
            name: name.to_owned(),
            minimum: Some(minimum),
            maximum: Some(maximum),
            default: Some(default),
            implicit: false,
            is_ui: true,
        }
    }
}

#[derive(Clone)]
pub struct Output {
    pub name: String,
}

pub enum ProcessorType<T: SignalRate + 'static>
where
    Graph<T>: Processor<T>,
{
    Boxed(Box<dyn Processor<T>>),
    Subgraph(Graph<T>),
}

impl<T: SignalRate + 'static> ProcessorType<T>
where
    Graph<T>: Processor<T>,
{
    pub fn as_graph(&self) -> Option<&Graph<T>> {
        match self {
            Self::Boxed(_) => None,
            Self::Subgraph(g) => Some(g),
        }
    }

    pub fn as_graph_mut(&mut self) -> Option<&mut Graph<T>> {
        match self {
            Self::Boxed(_) => None,
            Self::Subgraph(g) => Some(g),
        }
    }
}

impl<T: SignalRate + 'static> Processor<T> for ProcessorType<T>
where
    Graph<T>: Processor<T>,
{
    fn process_sample(
        &self,
        buffer_idx: usize,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<T as SignalRate>::SiblingNode>>,
        inputs: &[Signal<T>],
        outputs: &mut [Signal<T>],
    ) {
        match self {
            Self::Boxed(p) => {
                p.process_sample(buffer_idx, sample_rate, sibling_node, inputs, outputs)
            }
            Self::Subgraph(p) => {
                p.process_sample(buffer_idx, sample_rate, sibling_node, inputs, outputs)
            }
        }
    }

    fn process_buffer(
        &self,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<T as SignalRate>::SiblingNode>>,
        inputs: &[Vec<Signal<T>>],
        outputs: &mut [Vec<Signal<T>>],
    ) {
        match self {
            Self::Boxed(p) => p.process_buffer(sample_rate, sibling_node, inputs, outputs),
            Self::Subgraph(p) => p.process_buffer(sample_rate, sibling_node, inputs, outputs),
        }
    }

    fn ui_update(&self, ui: &mut eframe::egui::Ui) {
        match self {
            Self::Boxed(p) => p.ui_update(ui),
            Self::Subgraph(p) => p.ui_update(ui),
        }
    }
}

impl<T: SignalRate + 'static, P: Processor<T> + 'static> From<Box<P>> for ProcessorType<T>
where
    Graph<T>: Processor<T>,
{
    fn from(value: Box<P>) -> Self {
        Self::Boxed(value)
    }
}

#[non_exhaustive]
pub struct Node<T: SignalRate + 'static>
where
    Graph<T>: Processor<T>,
{
    pub name: NodeName,
    pub sibling_node: Option<Arc<T::SiblingNode>>,
    pub inputs: Vec<Input<T>>,
    pub outputs: Vec<Output>,
    pub processor: ProcessorType<T>,
    inputs_cache: RwLock<Vec<Vec<Signal<T>>>>,
    outputs_cache: RwLock<Vec<Vec<Signal<T>>>>,
}

impl<T: SignalRate> std::fmt::Debug for Node<T>
where
    Graph<T>: Processor<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name.0)
    }
}

impl<T: SignalRate + 'static> Node<T>
where
    Graph<T>: Processor<T>,
{
    pub fn new(
        name: NodeName,
        signaltype_buffer_len: usize,
        mut inputs: Vec<Input<T>>,
        outputs: Vec<Output>,
        processor: ProcessorType<T>,
        sibling_node: Option<Arc<T::SiblingNode>>,
    ) -> Self {
        if !inputs.iter().any(|i| i.name == "t") {
            let mut t = Input::new("t", None);
            t.implicit = true;
            inputs.push(t);
        }
        Self {
            name,
            inputs_cache: RwLock::new(
                inputs
                    .iter()
                    .map(|v| vec![v.default.unwrap_or(Signal::new(0.0)); signaltype_buffer_len])
                    .collect(),
            ),
            outputs_cache: RwLock::new(
                outputs
                    .iter()
                    .map(|_| vec![Signal::new(0.0); signaltype_buffer_len])
                    .collect(),
            ),

            inputs,
            outputs,
            processor,
            sibling_node,
        }
    }

    pub fn from_graph(graph: Graph<T>) -> Self {
        Self::new(
            graph.name.clone(),
            graph.signaltype_buffer_len,
            graph
                .input_node_indices
                .iter()
                .map(|idx| {
                    let node = &graph.digraph[*idx];
                    Input::new(&node.name.0, node.inputs[0].default)
                })
                .collect(),
            graph
                .output_node_indices
                .iter()
                .map(|idx| {
                    let node = &graph.digraph[*idx];
                    Output {
                        name: node.name.clone().into(),
                    }
                })
                .collect(),
            ProcessorType::Subgraph(graph),
            None,
        )
    }

    pub fn output_named(&self, name: &str) -> Option<usize> {
        self.outputs
            .iter()
            .enumerate()
            .find_map(|(i, p)| if p.name == name { Some(i) } else { None })
    }

    pub fn input_named(&self, name: &str) -> Option<usize> {
        self.inputs
            .iter()
            .enumerate()
            .find_map(|(i, p)| if p.name == name { Some(i) } else { None })
    }
}

impl Node<ControlRate> {
    pub fn cached_input(&self, idx: usize) -> Option<Signal<ControlRate>> {
        self.inputs_cache.read().unwrap().get(idx).map(|inp| inp[0])
    }
}

pub struct Graph<T: SignalRate + 'static>
where
    Self: Processor<T>,
{
    pub name: NodeName,
    signaltype_buffer_len: usize,
    pub digraph: DiGraph<Arc<Node<T>>, Connection>,
    node_indices_by_name: BTreeMap<String, NodeIndex>,
    pub input_node_indices: Vec<NodeIndex>,
    pub output_node_indices: Vec<NodeIndex>,
    partitions: Vec<Vec<NodeIndex>>,
}

impl Graph<AudioRate> {
    pub fn new(
        name: Option<NodeName>,
        audio_buffer_len: usize,
        mut inputs: Vec<Input<AudioRate>>,
        outputs: Vec<Output>,
    ) -> Graph<AudioRate> {
        let t = Input {
            name: "t".to_owned(),
            minimum: None,
            maximum: None,
            default: None,
            implicit: true,
            is_ui: false,
        };
        inputs.push(t);
        let mut this = Self {
            name: name.unwrap_or(NodeName::new("graph")),
            signaltype_buffer_len: audio_buffer_len,
            digraph: DiGraph::new(),
            node_indices_by_name: BTreeMap::default(),
            input_node_indices: Vec::default(),
            output_node_indices: Vec::default(),
            partitions: Vec::default(),
        };

        for inp in inputs {
            let an = GraphInput::create_audio_node(&inp.name, audio_buffer_len, inp.clone());
            let idx = this.add_node(an);
            this.node_indices_by_name.insert(inp.name.to_owned(), idx);
            this.input_node_indices.push(idx);
        }
        for out in outputs {
            let (an, _cn) = GraphOutput::create_nodes(&out.name, audio_buffer_len, 0.0);
            let idx = this.add_node(an);
            this.node_indices_by_name.insert(out.name.to_owned(), idx);
            this.output_node_indices.push(idx);
        }

        this
    }
}

impl Graph<ControlRate> {
    pub fn new(
        name: Option<NodeName>,
        mut inputs: Vec<Input<ControlRate>>,
        outputs: Vec<Output>,
    ) -> Graph<ControlRate> {
        let t = Input {
            name: "t".to_owned(),
            minimum: None,
            maximum: None,
            default: None,
            implicit: true,
            is_ui: false,
        };
        inputs.push(t);
        let mut this = Self {
            name: name.unwrap_or(NodeName::new("graph")),
            signaltype_buffer_len: 1, // control rate graph doesn't use buffers
            digraph: DiGraph::new(),
            node_indices_by_name: BTreeMap::default(),
            input_node_indices: Vec::default(),
            output_node_indices: Vec::default(),
            partitions: Vec::default(),
        };

        for inp in inputs {
            let cn = GraphInput::create_control_node(&inp.name, inp.clone());
            let idx = this.add_node(cn);
            this.node_indices_by_name.insert(inp.name.to_owned(), idx);
            this.input_node_indices.push(idx);
        }
        for out in outputs {
            let (_an, cn) = GraphOutput::create_nodes(&out.name, 1, 0.0);
            let idx = this.add_node(cn);
            this.node_indices_by_name.insert(out.name.to_owned(), idx);
            this.output_node_indices.push(idx);
        }

        this
    }
}

impl<T: SignalRate + 'static + Send + Sync> Graph<T>
where
    Self: Processor<T>,
    Signal<T>: std::fmt::Debug,
{
    pub fn write_dot(&self, name: &str) {
        use std::io::Write;
        let mut f = std::fs::File::create(name).unwrap();
        write!(
            f,
            "{:?}",
            Dot::with_config(&self.digraph, &[Config::EdgeNoLabel])
        )
        .unwrap();
    }

    pub fn add_node(&mut self, node: Arc<Node<T>>) -> NodeIndex {
        let name = node.name.to_owned();
        let idx = self.digraph.add_node(node);
        self.node_indices_by_name.insert(name.to_string(), idx);
        self.repartition();
        idx
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: Connection,
    ) -> EdgeIndex {
        let idx = self.digraph.add_edge(source, sink, connection);
        self.repartition();
        idx
    }

    fn repartition(&mut self) {
        self.partitions.clear();

        let starts = self.digraph.externals(Direction::Incoming);
        let mut bfs_stack: VecDeque<NodeIndex> = VecDeque::new();
        let mut bfs_visited = BTreeSet::default();
        if let Some(id) = self.node_id_by_name("t") {
            bfs_stack.push_back(id);
        }
        for node in starts {
            if !bfs_stack.contains(&node) {
                bfs_stack.push_back(node);
            }
        }
        for node in self.input_node_indices.iter() {
            if !bfs_stack.contains(node) {
                bfs_stack.push_back(*node);
            }
        }

        self.partitions.push(bfs_stack.clone().into());

        loop {
            let mut next_layer = Vec::new();
            while let Some(node) = bfs_stack.pop_front() {
                if !bfs_visited.contains(&node) {
                    bfs_visited.insert(node);
                    for edge in self.digraph.edges_directed(node, Direction::Outgoing) {
                        if !next_layer.contains(&edge.target())
                            && !bfs_visited.contains(&edge.target())
                            && self
                                .digraph
                                .edges_directed(node, Direction::Incoming)
                                .all(|edge| bfs_visited.contains(&edge.source()))
                        {
                            next_layer.push(edge.target());
                        }
                    }
                }
            }
            if next_layer.is_empty() {
                break;
            }
            self.partitions.push(next_layer.clone());
            bfs_stack = next_layer.into();
        }
    }

    pub fn node_id_by_name(&self, name: &str) -> Option<NodeIndex> {
        self.node_indices_by_name.get(name).copied()
    }

    pub fn process_graph(
        &self,
        sample_rate: Scalar,
        inputs: &BTreeMap<NodeIndex, &Vec<Signal<T>>>,
        outputs: &mut BTreeMap<NodeIndex, &mut Vec<Signal<T>>>,
    ) {
        // early check for empty graph (nothing to do)
        if self.digraph.node_count() == 0 {
            return;
        }

        // copy the provided input values into each input node's input cache
        for (inp_idx, value) in inputs.iter() {
            self.digraph[*inp_idx]
                .inputs_cache
                .write()
                .unwrap()
                .get_mut(0)
                .unwrap()
                .copy_from_slice(value);
        }

        // walk the BFS...
        for layer in self.partitions.iter() {
            layer.iter().copied().for_each(|node_id| {
                // for each incoming connection into the visited node:
                // - grab the cached outputs from earlier in the graph
                // - copy them to the input cache of the currently visited node
                for edge in self.digraph.edges_directed(node_id, Direction::Incoming) {
                    let out = {
                        &self.digraph[edge.source()].outputs_cache.read().unwrap()
                            [edge.weight().source_output]
                    };
                    self.digraph[node_id]
                        .inputs_cache
                        .write()
                        .unwrap()
                        .get_mut(edge.weight().sink_input)
                        .unwrap()
                        .copy_from_slice(out);
                }

                // create a copy of the inputs from the cache (necessary because we mutably borrow `self` in the next step)
                let in_cache = self.digraph[node_id].inputs_cache.read().unwrap();
                let mut inps = in_cache.iter().cloned().collect::<Vec<_>>();

                // manually copy over `t` since it's implicit
                inps[self.digraph[node_id].input_named("t").unwrap()]
                    .copy_from_slice(inputs[&self.node_id_by_name("t").unwrap()]);

                let out_cache = { self.digraph[node_id].outputs_cache.read().unwrap().clone() };
                let mut outs = out_cache
                    .iter()
                    .map(|v| vec![Signal::new(0.0); v.len()])
                    .collect::<Vec<_>>();

                // run the processing logic for this node, which will store its results directly in our output cache
                self.digraph[node_id].processor.process_buffer(
                    sample_rate,
                    self.digraph[node_id].sibling_node.as_ref(),
                    &inps,
                    &mut outs,
                );
                let mut out_cache = self.digraph[node_id].outputs_cache.write().unwrap();
                for (i, out) in outs.into_iter().enumerate() {
                    out_cache[i].copy_from_slice(&out);
                }
            });
        }

        // copy the cached (and now updated) output values into the mutable passed outputs
        for (out_name, out) in outputs.iter_mut() {
            out.copy_from_slice(&self.digraph[*out_name].outputs_cache.read().unwrap()[0]);
        }
    }

    pub fn into_node(self) -> Node<T> {
        Node::from_graph(self)
    }
}

impl<T: SignalRate + Send + Sync> Processor<T> for Graph<T>
where
    Self: Send + Sync,
    Signal<T>: std::fmt::Debug,
{
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<T as SignalRate>::SiblingNode>>,
        _inputs: &[Signal<T>],
        _outputs: &mut [Signal<T>],
    ) {
        unimplemented!()
    }
    fn process_buffer(
        &self,
        sample_rate: Scalar,
        _sibling_node: Option<&Arc<<T as SignalRate>::SiblingNode>>,
        inputs: &[Vec<Signal<T>>],
        outputs: &mut [Vec<Signal<T>>],
    ) {
        let inputs = BTreeMap::from_iter(
            inputs
                .iter()
                .enumerate()
                .map(|(i, inp)| (self.input_node_indices[i], inp)),
        );
        let mut outputs = BTreeMap::from_iter(
            outputs
                .iter_mut()
                .enumerate()
                .map(|(i, out)| (self.output_node_indices[i], out)),
        );
        self.process_graph(sample_rate, &inputs, &mut outputs);
    }
}

#[macro_export]
macro_rules! dual_graphs {
    {
        $name:expr;
        $audio_buffer_len:expr;
        @in {$($audio_inputs:literal = $ai_default_values:expr)*}
        @out {$($audio_outputs:literal)*}
        #in {$($control_inputs:literal = $ci_default_values:expr)*}
        #out {$($control_outputs:literal)*}
    } => {
        {
            let a_outs = vec![$(($crate::graph::Output { name: $audio_outputs.to_owned() })),*];
            let c_outs = vec![$(($crate::graph::Output { name: $control_outputs.to_owned() })),*];
            let a_ins = vec![$(($crate::graph::Input::new($audio_inputs, Some($crate::dsp::Signal::new_audio($ai_default_values))))),*];
            let c_ins = vec![$(($crate::graph::Input::new($control_inputs, Some($crate::dsp::Signal::new_control($ci_default_values))))),*];
            let ag = $crate::graph::Graph::<AudioRate>::new(Some($crate::graph::NodeName::new($name)), $audio_buffer_len, a_ins, a_outs);
            let cg = $crate::graph::Graph::<ControlRate>::new(Some($crate::graph::NodeName::new($name)), c_ins, c_outs);
            (ag, cg)
        }
    };
}

pub trait CreateNodes {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>);
}
