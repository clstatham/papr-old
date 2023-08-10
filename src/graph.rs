use std::{
    collections::VecDeque,
    sync::{Arc, RwLock},
};

use petgraph::prelude::*;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    dsp::{
        graph_util::{GraphInput, GraphOutput},
        AudioRate, ControlRate, Processor, Signal, SignalRate,
    },
    Scalar,
};

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, derive_more::Display, derive_more::Into, derive_more::From,
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

impl From<InputName> for NodeName {
    fn from(value: InputName) -> Self {
        Self(value.0)
    }
}

impl From<OutputName> for NodeName {
    fn from(value: OutputName) -> Self {
        Self(value.0)
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, derive_more::Display, derive_more::Into, derive_more::From,
)]
pub struct InputName(String);

impl InputName {
    pub fn new(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl Default for InputName {
    fn default() -> Self {
        Self("input".to_owned())
    }
}

impl From<NodeName> for InputName {
    fn from(value: NodeName) -> Self {
        Self(value.0)
    }
}

impl From<OutputName> for InputName {
    fn from(value: OutputName) -> Self {
        Self(value.0)
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, Hash, derive_more::Display, derive_more::Into, derive_more::From,
)]
pub struct OutputName(String);

impl Default for OutputName {
    fn default() -> Self {
        Self("out".to_owned())
    }
}

impl OutputName {
    pub fn new(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl From<InputName> for OutputName {
    fn from(value: InputName) -> Self {
        Self(value.0)
    }
}

impl From<NodeName> for OutputName {
    fn from(value: NodeName) -> Self {
        Self(value.0)
    }
}

#[derive(Clone, Debug)]
pub struct Connection {
    pub source_output: OutputName,
    pub sink_input: InputName,
}

#[derive(Clone)]
#[non_exhaustive]
pub struct Input<T: SignalRate> {
    pub name: InputName,
    pub minimum: Option<Signal<T>>,
    pub maximum: Option<Signal<T>>,
    pub default: Option<Signal<T>>,
    pub implicit: bool,
}

impl<T: SignalRate> Input<T> {
    pub fn new(name: &str, default: Option<Signal<T>>) -> Self {
        Self {
            name: InputName::new(name),
            minimum: None,
            maximum: None,
            default,
            implicit: false,
        }
    }

    pub fn new_bounded(
        name: &str,
        minimum: Signal<T>,
        maximum: Signal<T>,
        default: Signal<T>,
    ) -> Self {
        Self {
            name: InputName::new(name),
            minimum: Some(minimum),
            maximum: Some(maximum),
            default: Some(default),
            implicit: false,
        }
    }
}

#[derive(Clone)]
pub struct Output {
    pub name: OutputName,
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
        inputs: &FxHashMap<InputName, Signal<T>>,
        outputs: &mut FxHashMap<OutputName, Signal<T>>,
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
        inputs: &FxHashMap<InputName, Vec<Signal<T>>>,
        outputs: &mut FxHashMap<OutputName, Vec<Signal<T>>>,
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
    pub inputs: FxHashMap<InputName, Input<T>>,
    pub outputs: FxHashMap<OutputName, Output>,
    pub processor: ProcessorType<T>,
    inputs_cache: RwLock<FxHashMap<InputName, Vec<Signal<T>>>>,
    outputs_cache: RwLock<FxHashMap<OutputName, Vec<Signal<T>>>>,
}

impl<T: SignalRate + 'static> Node<T>
where
    Graph<T>: Processor<T>,
{
    pub fn new(
        name: NodeName,
        signaltype_buffer_len: usize,
        mut inputs: FxHashMap<InputName, Input<T>>,
        outputs: FxHashMap<OutputName, Output>,
        processor: ProcessorType<T>,
        sibling_node: Option<Arc<T::SiblingNode>>,
    ) -> Self {
        inputs.insert(
            InputName::new("t"),
            Input {
                name: InputName::new("t"),
                minimum: None,
                maximum: None,
                default: None,
                implicit: true,
            },
        );
        Self {
            name,
            inputs_cache: RwLock::new(
                inputs
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.to_owned(),
                            vec![v.default.unwrap_or(Signal::new(0.0)); signaltype_buffer_len],
                        )
                    })
                    .collect(),
            ),
            outputs_cache: RwLock::new(
                outputs
                    .keys()
                    .map(|k| (k.to_owned(), vec![Signal::new(0.0); signaltype_buffer_len]))
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
                .graph_inputs
                .iter()
                .map(|idx| {
                    let node = &graph.digraph[*idx];
                    (
                        InputName::new(&node.name.0.to_owned()),
                        Input::new(&node.name.0, node.inputs[&InputName::default()].default),
                    )
                })
                .collect(),
            graph
                .graph_outputs
                .iter()
                .map(|idx| {
                    let node = &graph.digraph[*idx];
                    (
                        node.name.clone().into(),
                        Output {
                            name: node.name.clone().into(),
                        },
                    )
                })
                .collect(),
            ProcessorType::Subgraph(graph),
            None,
        )
    }
}

impl Node<ControlRate> {
    pub fn cached_input(&self, inp: &InputName) -> Option<Signal<ControlRate>> {
        self.inputs_cache.read().unwrap().get(inp).map(|inp| inp[0])
    }
}

pub struct Graph<T: SignalRate + 'static>
where
    Self: Processor<T>,
{
    pub name: NodeName,
    signaltype_buffer_len: usize,
    pub digraph: DiGraph<Arc<Node<T>>, Connection>,
    node_indices_by_name: FxHashMap<NodeName, NodeIndex>,
    pub graph_inputs: Vec<NodeIndex>,
    pub graph_outputs: Vec<NodeIndex>,
    partitions: Vec<Vec<NodeIndex>>,
}

impl Graph<AudioRate> {
    pub fn new(
        name: Option<NodeName>,
        audio_buffer_len: usize,
        mut inputs: Vec<Input<AudioRate>>,
        outputs: Vec<Output>,
    ) -> Graph<AudioRate> {
        let mut this = Self {
            name: name.unwrap_or(NodeName::new("graph")),
            signaltype_buffer_len: audio_buffer_len,
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            graph_inputs: Vec::default(),
            graph_outputs: Vec::default(),
            partitions: Vec::default(),
        };
        let t = Input {
            name: InputName::new("t"),
            minimum: None,
            maximum: None,
            default: None,
            implicit: true,
        };
        inputs.push(t);
        for inp in inputs {
            let an = GraphInput::create_audio_node(&inp.name.0, audio_buffer_len, inp.clone());
            let idx = this.add_node(an);
            this.node_indices_by_name
                .insert(NodeName::new(&inp.name.0.to_owned()), idx);
            if !inp.implicit {
                this.graph_inputs.push(idx);
            }
        }
        for out in outputs {
            let (an, _cn) = GraphOutput::create_nodes(&out.name.0, audio_buffer_len, 0.0);
            let idx = this.add_node(an);
            this.node_indices_by_name
                .insert(NodeName::new(&out.name.0.to_owned()), idx);
            this.graph_outputs.push(idx);
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
        let mut this = Self {
            name: name.unwrap_or(NodeName::new("graph")),
            signaltype_buffer_len: 1, // control rate graph doesn't use buffers
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            graph_inputs: Vec::default(),
            graph_outputs: Vec::default(),
            partitions: Vec::default(),
        };
        let t = Input {
            name: InputName::new("t"),
            minimum: None,
            maximum: None,
            default: None,
            implicit: true,
        };
        inputs.push(t);
        for inp in inputs {
            let cn = GraphInput::create_control_node(&inp.name.0, inp.clone());
            let idx = this.add_node(cn);
            this.node_indices_by_name
                .insert(NodeName::new(&inp.name.0.to_owned()), idx);
            if !inp.implicit {
                this.graph_inputs.push(idx);
            }
        }
        for out in outputs {
            let (_an, cn) = GraphOutput::create_nodes(&out.name.0, 1, 0.0);
            let idx = this.add_node(cn);
            this.node_indices_by_name
                .insert(NodeName::new(&out.name.0.to_owned()), idx);
            this.graph_outputs.push(idx);
        }

        this
    }
}

impl<T: SignalRate + 'static + Send + Sync> Graph<T>
where
    Self: Processor<T>,
{
    pub fn add_node(&mut self, node: Arc<Node<T>>) -> NodeIndex {
        let name = node.name.to_owned();
        let idx = self.digraph.add_node(node);
        self.node_indices_by_name.insert(name, idx);
        self.repartition();
        idx
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: Connection,
    ) -> EdgeIndex {
        assert!(
            self.digraph[source]
                .outputs
                .contains_key(&connection.source_output),
            "Graph::add_edge(): No output named `{}` on node",
            &connection.source_output.0
        );
        assert!(
            self.digraph[sink]
                .inputs
                .contains_key(&connection.sink_input),
            "Graph::add_edge(): No input named `{}` on node",
            &connection.source_output.0
        );
        self.digraph.add_edge(source, sink, connection)
    }

    fn repartition(&mut self) {
        self.partitions.clear();

        let starts = self.digraph.externals(Direction::Incoming);
        let mut bfs_stack = VecDeque::new();
        let mut bfs_visited = FxHashSet::default();
        if let Some(id) = self.node_id_by_name(&NodeName::new("t")) {
            bfs_stack.push_back(id);
        }
        for node in starts {
            if !bfs_stack.contains(&node) {
                bfs_stack.push_back(node);
            }
        }
        for node in self.graph_inputs.iter() {
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

        // dbg!(&self.name, &self.partitions);
    }

    pub fn node_id_by_name(&self, name: &NodeName) -> Option<NodeIndex> {
        self.node_indices_by_name.get(name).copied()
    }

    pub fn process_graph(
        &self,
        sample_rate: Scalar,
        inputs: &FxHashMap<InputName, Vec<Signal<T>>>,
        outputs: &mut FxHashMap<OutputName, Vec<Signal<T>>>,
    ) {
        // early check for empty graph (nothing to do)
        if self.digraph.node_count() == 0 {
            return;
        }

        // copy the provided input values into each input node's input chache
        for (input_name, value) in inputs.iter() {
            let inp_idx = self.node_id_by_name(&input_name.clone().into()).unwrap();
            self.digraph[inp_idx]
                .inputs_cache
                .write()
                .unwrap()
                .get_mut(&InputName::default())
                .unwrap()
                .copy_from_slice(value);
        }

        // walk the BFS...
        for layer in self.partitions.iter() {
            layer.par_iter().for_each(|node_id| {
                let node_id = *node_id;
                // for each incoming connection into the visited node:
                // - grab the cached outputs from earlier in the graph
                // - copy them to the input cache of the currently visited node
                for edge in self.digraph.edges_directed(node_id, Direction::Incoming) {
                    let out = {
                        &self.digraph[edge.source()].outputs_cache.read().unwrap()
                            [&edge.weight().source_output]
                    };
                    self.digraph[node_id]
                        .inputs_cache
                        .write()
                        .unwrap()
                        .get_mut(&edge.weight().sink_input)
                        .unwrap()
                        .copy_from_slice(out);
                }

                // create a copy of the inputs from the cache (necessary because we mutably borrow `self` in the next step)
                let mut inps = self.digraph[node_id]
                    .inputs_cache
                    .read()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.to_owned(), v.clone()))
                    .collect::<FxHashMap<_, _>>();
                inps.insert(
                    InputName::new("t"),
                    inputs[&InputName::new("t")].clone(),
                );
                let mut outs = self.digraph[node_id]
                    .outputs_cache
                    .read()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.to_owned(), vec![Signal::new(0.0); v.len()]))
                    .collect::<FxHashMap<_, _>>();
                // run the processing logic for this node, which will store its results directly in our output cache

                self.digraph[node_id].processor.process_buffer(
                    sample_rate,
                    self.digraph[node_id].sibling_node.as_ref(),
                    &inps,
                    &mut outs,
                );
                for (name, out) in self.digraph[node_id]
                    .outputs_cache
                    .write()
                    .unwrap()
                    .iter_mut()
                {
                    out.copy_from_slice(&outs[name]);
                }
            });
        }

        // copy the cached (and now updated) output values into the mutable passed outputs
        for (out_name, out) in outputs.iter_mut() {
            let node_idx = self.node_id_by_name(&out_name.clone().into()).unwrap();
            out.copy_from_slice(
                &self.digraph[node_idx].outputs_cache.read().unwrap()[&OutputName::default()],
            );
        }
    }

    pub fn into_node(self) -> Node<T> {
        Node::from_graph(self)
    }
}

impl<T: SignalRate + Send + Sync> Processor<T> for Graph<T>
where
    Self: Send + Sync,
{
    fn process_sample(
        &self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _sibling_node: Option<&Arc<<T as SignalRate>::SiblingNode>>,
        _inputs: &FxHashMap<InputName, Signal<T>>,
        _outputs: &mut FxHashMap<OutputName, Signal<T>>,
    ) {
        unimplemented!()
    }
    fn process_buffer(
        &self,
        sample_rate: Scalar,
        _sibling_node: Option<&Arc<<T as SignalRate>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Vec<Signal<T>>>,
        outputs: &mut FxHashMap<OutputName, Vec<Signal<T>>>,
    ) {
        self.process_graph(sample_rate, inputs, outputs)
    }
}

#[macro_export]
macro_rules! dual_graphs {
    {
        $name:literal
        $audio_buffer_len:expr;
        @in {$($audio_inputs:literal = $ai_default_values:expr)*}
        @out {$($audio_outputs:literal)*}
        #in {$($control_inputs:literal = $ci_default_values:expr)*}
        #out {$($control_outputs:literal)*}
    } => {
        {
            let a_outs = vec![$(($crate::graph::Output { name: $crate::graph::OutputName::new($audio_outputs) })),*];
            let c_outs = vec![$(($crate::graph::Output { name: $crate::graph::OutputName::new($control_outputs) })),*];
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
