use std::sync::{Arc, RwLock};

use petgraph::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    dsp::{
        graph_util::{GraphInput, GraphOutput},
        AudioRate, ControlRate, Processor, Signal, SignalType,
    },
    Scalar,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display)]
pub struct NodeName(pub String);

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display)]
pub struct InputName(pub String);

impl Default for InputName {
    fn default() -> Self {
        Self("in".to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display)]
pub struct OutputName(pub String);

impl Default for OutputName {
    fn default() -> Self {
        Self("out".to_owned())
    }
}

#[derive(Clone, Debug)]
pub struct Connection {
    pub source_output: OutputName,
    pub sink_input: InputName,
}

#[derive(Clone)]
#[non_exhaustive]
pub struct Input<T: SignalType> {
    pub name: InputName,
    pub minimum: Option<Signal<T>>,
    pub maximum: Option<Signal<T>>,
    pub default: Signal<T>,
}

impl<T: SignalType> Input<T> {
    pub fn new(name: &str, default: Signal<T>) -> Self {
        Self {
            name: InputName(name.to_owned()),
            minimum: None,
            maximum: None,
            default,
        }
    }

    pub fn new_bounded(
        name: &str,
        minimum: Option<Signal<T>>,
        maximum: Option<Signal<T>>,
        default: Signal<T>,
    ) -> Self {
        Self {
            name: InputName(name.to_owned()),
            minimum,
            maximum,
            default,
        }
    }
}

#[derive(Clone)]
pub struct Output {
    pub name: OutputName,
}

pub enum ProcessorType<T: SignalType + 'static>
where
    Graph<T>: Processor<T>,
{
    Boxed(Box<dyn Processor<T>>),
    Subgraph(Graph<T>),
}

impl<T: SignalType + 'static> ProcessorType<T>
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

impl<T: SignalType + 'static> Processor<T> for ProcessorType<T>
where
    Graph<T>: Processor<T>,
{
    fn process(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        sibling_node: Option<&Arc<<T as SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<T>>,
        outputs: &mut FxHashMap<OutputName, Signal<T>>,
    ) {
        match self {
            Self::Boxed(p) => p.process(t, sample_rate, sibling_node, inputs, outputs),
            Self::Subgraph(p) => p.process(t, sample_rate, sibling_node, inputs, outputs),
        }
    }

    fn ui_update(&self, ui: &mut eframe::egui::Ui) {
        match self {
            Self::Boxed(p) => p.ui_update(ui),
            Self::Subgraph(p) => p.ui_update(ui),
        }
    }
}

impl<T: SignalType + 'static, P: Processor<T> + 'static> From<Box<P>> for ProcessorType<T>
where
    Graph<T>: Processor<T>,
{
    fn from(value: Box<P>) -> Self {
        Self::Boxed(value)
    }
}

#[non_exhaustive]
pub struct Node<T: SignalType + 'static>
where
    Graph<T>: Processor<T>,
{
    pub name: NodeName,
    pub sibling_node: Option<Arc<T::SiblingNode>>,
    pub inputs: FxHashMap<InputName, Input<T>>,
    pub outputs: FxHashMap<OutputName, Output>,
    pub processor: ProcessorType<T>,
    inputs_cache: RwLock<FxHashMap<InputName, Signal<T>>>,
    outputs_cache: RwLock<FxHashMap<OutputName, Signal<T>>>,
}

impl<T: SignalType + 'static> Node<T>
where
    Graph<T>: Processor<T>,
{
    pub fn new(
        name: NodeName,
        inputs: FxHashMap<InputName, Input<T>>,
        outputs: FxHashMap<OutputName, Output>,
        processor: ProcessorType<T>,
        sibling_node: Option<Arc<T::SiblingNode>>,
    ) -> Self {
        Self {
            name,
            inputs_cache: RwLock::new(
                inputs
                    .iter()
                    .map(|(k, v)| (k.to_owned(), v.default))
                    .collect(),
            ),
            outputs_cache: RwLock::new(
                outputs
                    .keys()
                    .map(|k| (k.to_owned(), Signal::new(0.0)))
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
            graph
                .graph_inputs
                .iter()
                .map(|idx| {
                    let node = &graph.digraph[*idx];
                    (
                        InputName(node.name.0.to_owned()),
                        Input::new(&node.name.0, Signal::new(0.0)),
                    )
                })
                .collect(),
            graph
                .graph_outputs
                .iter()
                .map(|idx| {
                    let node = &graph.digraph[*idx];
                    (
                        OutputName(node.name.0.to_owned()),
                        Output {
                            name: OutputName(node.name.0.to_owned()),
                        },
                    )
                })
                .collect(),
            ProcessorType::Subgraph(graph),
            None,
        )
    }

    pub fn cached_input(&self, inp_name: &InputName) -> Option<Signal<T>> {
        self.inputs_cache.read().unwrap().get(inp_name).copied()
    }
}

pub struct Graph<T: SignalType + 'static>
where
    Self: Processor<T>,
{
    pub name: NodeName,
    pub digraph: DiGraph<Arc<Node<T>>, Connection>,
    node_indices_by_name: FxHashMap<NodeName, NodeIndex>,
    pub graph_inputs: Vec<NodeIndex>,
    pub graph_outputs: Vec<NodeIndex>,
}

impl Graph<AudioRate> {
    pub fn new(
        name: Option<NodeName>,
        inputs: Vec<Input<AudioRate>>,
        outputs: Vec<Output>,
    ) -> Graph<AudioRate> {
        let mut this = Self {
            name: name.unwrap_or(NodeName("graph".to_owned())),
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            graph_inputs: Vec::default(),
            graph_outputs: Vec::default(),
        };
        for inp in inputs {
            let (an, _cn) = GraphInput::create_nodes(&inp.name.0);
            let idx = this.add_node(an);
            this.node_indices_by_name
                .insert(NodeName(inp.name.0.to_owned()), idx);
            this.graph_inputs.push(idx);
        }
        for out in outputs {
            let (an, _cn) = GraphOutput::create_nodes(&out.name.0);
            let idx = this.add_node(an);
            this.node_indices_by_name
                .insert(NodeName(out.name.0.to_owned()), idx);
            this.graph_outputs.push(idx);
        }

        this
    }
}

impl Graph<ControlRate> {
    pub fn new(
        name: Option<NodeName>,
        inputs: Vec<Input<ControlRate>>,
        outputs: Vec<Output>,
    ) -> Graph<ControlRate> {
        let mut this = Self {
            name: name.unwrap_or(NodeName("graph".to_owned())),
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            graph_inputs: Vec::default(),
            graph_outputs: Vec::default(),
        };
        for inp in inputs {
            let (_an, cn) = GraphInput::create_nodes(&inp.name.0);
            let idx = this.add_node(cn);
            this.node_indices_by_name
                .insert(NodeName(inp.name.0.to_owned()), idx);
            this.graph_inputs.push(idx);
        }
        for out in outputs {
            let (_an, cn) = GraphOutput::create_nodes(&out.name.0);
            let idx = this.add_node(cn);
            this.node_indices_by_name
                .insert(NodeName(out.name.0.to_owned()), idx);
            this.graph_outputs.push(idx);
        }

        this
    }
}

impl<T: SignalType + 'static> Graph<T>
where
    Self: Processor<T>,
{
    pub fn add_node(&mut self, node: Arc<Node<T>>) -> NodeIndex {
        let name = node.name.to_owned();
        let idx = self.digraph.add_node(node);
        self.node_indices_by_name.insert(name, idx);
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

    pub fn node_id_by_name(&self, name: &NodeName) -> Option<NodeIndex> {
        self.node_indices_by_name.get(name).copied()
    }

    pub fn process_graph(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        inputs: &FxHashMap<InputName, Signal<T>>,
        outputs: &mut FxHashMap<OutputName, Signal<T>>,
    ) {
        // early check for empty graph (nothing to do)
        if self.digraph.node_count() == 0 {
            return;
        }

        // copy the provided input values into each input node's input chache
        for (input_name, value) in inputs.iter() {
            let inp_idx = self.node_id_by_name(&input_name.clone().into()).unwrap();
            *self.digraph[inp_idx]
                .inputs_cache
                .write()
                .unwrap()
                .get_mut(&InputName::default())
                .unwrap() = *value;
        }

        // initialize a breadth-first search starting at the input nodes
        let mut starts = self.digraph.externals(Direction::Incoming);
        let mut bfs = Bfs::new(
            &self.digraph,
            starts
                .next()
                .expect("Graph::process(): graph has no input/source nodes"),
        );
        for node in starts {
            bfs.stack.push_back(node);
        }
        for node in self.graph_inputs.iter() {
            bfs.stack.push_back(*node);
        }

        // walk the BFS...
        while let Some(node_id) = bfs.next(&self.digraph) {
            // for each incoming connection into the visited node:
            // - grab the cached outputs from earlier in the graph
            // - copy them to the input cache of the currently visited node
            for edge in self.digraph.edges_directed(node_id, Direction::Incoming) {
                let out = {
                    self.digraph[edge.source()].outputs_cache.read().unwrap()
                        [&edge.weight().source_output]
                };
                *self.digraph[node_id]
                    .inputs_cache
                    .write()
                    .unwrap()
                    .get_mut(&edge.weight().sink_input)
                    .unwrap() = out;
            }

            // create a copy of the inputs from the cache (necessary because we mutably borrow `self` in the next step)
            let inps = self.digraph[node_id].inputs_cache.read().unwrap().clone();
            let node = &self.digraph[node_id];
            // run the processing logic for this node, which will store its results directly in our output cache
            node.processor.process(
                t,
                sample_rate,
                node.sibling_node.as_ref(),
                &inps,
                &mut node.outputs_cache.write().unwrap(),
            )
        }

        // copy the cached (and now updated) output values into the mutable passed outputs
        for (out_name, out) in outputs.iter_mut() {
            let node_idx = self.node_id_by_name(&out_name.clone().into()).unwrap();
            *out = self.digraph[node_idx].outputs_cache.read().unwrap()[&OutputName::default()];
        }
    }

    pub fn into_node(self) -> Node<T> {
        Node::from_graph(self)
    }
}

impl<T: SignalType> Processor<T> for Graph<T>
where
    Self: Send + Sync,
{
    fn process(
        &self,
        t: Scalar,
        sample_rate: Scalar,
        _sibling_node: Option<&Arc<<T as SignalType>::SiblingNode>>,
        inputs: &FxHashMap<InputName, Signal<T>>,
        outputs: &mut FxHashMap<OutputName, Signal<T>>,
    ) {
        self.process_graph(t, sample_rate, inputs, outputs)
    }
}

#[macro_export]
macro_rules! dual_graphs {
    {
        $name:literal
        @in {$($audio_inputs:literal = $ai_default_values:expr)*}
        @out {$($audio_outputs:literal)*}
        #in {$($control_inputs:literal = $ci_default_values:expr)*}
        #out {$($control_outputs:literal)*}
    } => {
        {
            let a_outs = vec![$(($crate::graph::Output { name: $crate::graph::OutputName($audio_outputs.to_owned()) })),*];
            let c_outs = vec![$(($crate::graph::Output { name: $crate::graph::OutputName($control_outputs.to_owned()) })),*];
            let a_ins = vec![$(($crate::graph::Input::new($audio_inputs, $crate::dsp::Signal::new_audio($ai_default_values)))),*];
            let c_ins = vec![$(($crate::graph::Input::new($control_inputs, $crate::dsp::Signal::new_control($ci_default_values)))),*];
            let ag = $crate::graph::Graph::<AudioRate>::new(Some($crate::graph::NodeName($name.to_owned())), a_ins, a_outs);
            let cg = $crate::graph::Graph::<ControlRate>::new(Some($crate::graph::NodeName($name.to_owned())), c_ins, c_outs);
            (ag, cg)
        }
    };
}

pub trait CreateNodes {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>);
}
