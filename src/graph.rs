use std::{
    collections::VecDeque,
    sync::{Arc, RwLock},
};

use petgraph::{
    dot::{Config, Dot},
    prelude::*,
};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
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

#[derive(Clone, Debug)]
pub struct Connection {
    pub source_output: String,
    pub sink_input: String,
}

#[derive(Clone)]
#[non_exhaustive]
pub struct Input<T: SignalRate> {
    pub name: String,
    pub minimum: Option<Signal<T>>,
    pub maximum: Option<Signal<T>>,
    pub default: Option<Signal<T>>,
    pub implicit: bool,
}

impl<T: SignalRate> Input<T> {
    pub fn new(name: &str, default: Option<Signal<T>>) -> Self {
        Self {
            name: name.to_owned(),
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
            name: name.to_owned(),
            minimum: Some(minimum),
            maximum: Some(maximum),
            default: Some(default),
            implicit: false,
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
        inputs: &BTreeMap<&str, Signal<T>>,
        outputs: &mut BTreeMap<&str, Signal<T>>,
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
        inputs: &BTreeMap<&str, Vec<Signal<T>>>,
        outputs: &mut BTreeMap<&str, Vec<Signal<T>>>,
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
    pub inputs: BTreeMap<String, Input<T>>,
    pub outputs: BTreeMap<String, Output>,
    pub processor: ProcessorType<T>,
    inputs_cache: RwLock<BTreeMap<String, Vec<Signal<T>>>>,
    outputs_cache: RwLock<BTreeMap<String, Vec<Signal<T>>>>,
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
        mut inputs: BTreeMap<String, Input<T>>,
        outputs: BTreeMap<String, Output>,
        processor: ProcessorType<T>,
        sibling_node: Option<Arc<T::SiblingNode>>,
    ) -> Self {
        inputs.insert(
            "t".to_owned(),
            Input {
                name: "t".to_owned(),
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
                            k.to_string(),
                            vec![v.default.unwrap_or(Signal::new(0.0)); signaltype_buffer_len],
                        )
                    })
                    .collect(),
            ),
            outputs_cache: RwLock::new(
                outputs
                    .keys()
                    .map(|k| (k.to_string(), vec![Signal::new(0.0); signaltype_buffer_len]))
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
                        node.name.to_string(),
                        Input::new(&node.name.0, node.inputs["input"].default),
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
    pub fn cached_input(&self, inp: &str) -> Option<Signal<ControlRate>> {
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
    node_indices_by_name: BTreeMap<String, NodeIndex>,
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
            node_indices_by_name: BTreeMap::default(),
            graph_inputs: Vec::default(),
            graph_outputs: Vec::default(),
            partitions: Vec::default(),
        };
        let t = Input {
            name: "t".to_owned(),
            minimum: None,
            maximum: None,
            default: None,
            implicit: true,
        };
        inputs.push(t);
        for inp in inputs {
            let an = GraphInput::create_audio_node(&inp.name, audio_buffer_len, inp.clone());
            let idx = this.add_node(an);
            this.node_indices_by_name.insert(inp.name.to_owned(), idx);
            if !inp.implicit {
                this.graph_inputs.push(idx);
            }
        }
        for out in outputs {
            let (an, _cn) = GraphOutput::create_nodes(&out.name, audio_buffer_len, 0.0);
            let idx = this.add_node(an);
            this.node_indices_by_name.insert(out.name.to_owned(), idx);
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
            node_indices_by_name: BTreeMap::default(),
            graph_inputs: Vec::default(),
            graph_outputs: Vec::default(),
            partitions: Vec::default(),
        };
        let t = Input {
            name: "t".to_owned(),
            minimum: None,
            maximum: None,
            default: None,
            implicit: true,
        };
        inputs.push(t);
        for inp in inputs {
            let cn = GraphInput::create_control_node(&inp.name, inp.clone());
            let idx = this.add_node(cn);
            this.node_indices_by_name.insert(inp.name.to_owned(), idx);
            if !inp.implicit {
                this.graph_inputs.push(idx);
            }
        }
        for out in outputs {
            let (_an, cn) = GraphOutput::create_nodes(&out.name, 1, 0.0);
            let idx = this.add_node(cn);
            this.node_indices_by_name.insert(out.name.to_owned(), idx);
            this.graph_outputs.push(idx);
        }

        this
    }
}

impl<T: SignalRate + 'static + Send + Sync> Graph<T>
where
    Self: Processor<T>,
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
        assert!(
            self.digraph[source]
                .outputs
                .contains_key(&connection.source_output),
            "Graph::add_edge(): No output named `{}` on node",
            &connection.source_output
        );
        assert!(
            self.digraph[sink]
                .inputs
                .contains_key(&connection.sink_input),
            "Graph::add_edge(): No input named `{}` on node",
            &connection.source_output
        );
        self.digraph.add_edge(source, sink, connection)
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

    pub fn node_id_by_name(&self, name: &str) -> Option<NodeIndex> {
        self.node_indices_by_name.get(name).copied()
    }

    pub fn process_graph(
        &self,
        sample_rate: Scalar,
        inputs: &BTreeMap<&str, Vec<Signal<T>>>,
        outputs: &mut BTreeMap<&str, Vec<Signal<T>>>,
    ) {
        // early check for empty graph (nothing to do)
        if self.digraph.node_count() == 0 {
            return;
        }

        // copy the provided input values into each input node's input chache
        for (input_name, value) in inputs.iter() {
            let inp_idx = self.node_id_by_name(input_name).unwrap();
            self.digraph[inp_idx]
                .inputs_cache
                .write()
                .unwrap()
                .get_mut("input")
                .unwrap()
                .copy_from_slice(value);
        }

        // walk the BFS...
        for layer in self.partitions.iter() {
            layer.iter().for_each(|node_id| {
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
                let in_cache = self.digraph[node_id].inputs_cache.read().unwrap();
                let mut inps = in_cache
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.clone()))
                    .collect::<BTreeMap<_, _>>();
                inps.insert("t", inputs["t"].clone());
                let out_cache = { self.digraph[node_id].outputs_cache.read().unwrap().clone() };
                let mut outs = out_cache
                    .iter()
                    .map(|(k, v)| (k.as_str(), vec![Signal::new(0.0); v.len()]))
                    .collect::<BTreeMap<_, _>>();
                // run the processing logic for this node, which will store its results directly in our output cache

                self.digraph[node_id].processor.process_buffer(
                    sample_rate,
                    self.digraph[node_id].sibling_node.as_ref(),
                    &inps,
                    &mut outs,
                );
                let mut out_cache = self.digraph[node_id].outputs_cache.write().unwrap();
                for (name, out) in out_cache.iter_mut() {
                    out.copy_from_slice(&outs[name.as_str()]);
                }
            });
        }

        // copy the cached (and now updated) output values into the mutable passed outputs
        for (out_name, out) in outputs.iter_mut() {
            let node_idx = self.node_id_by_name(out_name.as_ref()).unwrap();
            out.copy_from_slice(&self.digraph[node_idx].outputs_cache.read().unwrap()["out"]);
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
        _inputs: &BTreeMap<&str, Signal<T>>,
        _outputs: &mut BTreeMap<&str, Signal<T>>,
    ) {
        unimplemented!()
    }
    fn process_buffer(
        &self,
        sample_rate: Scalar,
        _sibling_node: Option<&Arc<<T as SignalRate>::SiblingNode>>,
        inputs: &BTreeMap<&str, Vec<Signal<T>>>,
        outputs: &mut BTreeMap<&str, Vec<Signal<T>>>,
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
