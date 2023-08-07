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

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display)]
pub struct InputName(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display)]
pub struct OutputName(pub String);

#[derive(Clone, Debug)]
pub struct Connection {
    pub source_output: OutputName,
    pub sink_input: InputName,
}

#[derive(Clone)]
pub struct Input<T: SignalType> {
    pub name: InputName,
    pub default: Signal<T>,
}

impl<T: SignalType> Input<T> {
    pub fn new(name: &str, default: Signal<T>) -> Self {
        Self {
            name: InputName(name.to_owned()),
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
    pub sibling_node: Option<Arc<T::SiblingNode>>,
    pub inputs: FxHashMap<InputName, Input<T>>,
    pub outputs: FxHashMap<OutputName, Output>,
    pub processor: ProcessorType<T>,
    inputs_cache: RwLock<FxHashMap<InputName, Signal<T>>>,
}

impl<T: SignalType + 'static> Node<T>
where
    Graph<T>: Processor<T>,
{
    pub fn new(
        inputs: FxHashMap<InputName, Input<T>>,
        outputs: FxHashMap<OutputName, Output>,
        processor: ProcessorType<T>,
        sibling_node: Option<Arc<T::SiblingNode>>,
    ) -> Self {
        Self {
            inputs_cache: RwLock::new(
                inputs
                    .iter()
                    .map(|(k, v)| (k.to_owned(), v.default))
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
            graph
                .graph_inputs
                .keys()
                .map(|name| (name.to_owned(), Input::new(&name.0, Signal::new(0.0))))
                .collect(),
            graph
                .graph_outputs
                .keys()
                .map(|name| {
                    (
                        name.to_owned(),
                        Output {
                            name: name.to_owned(),
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
    pub digraph: DiGraph<Arc<Node<T>>, Connection>,
    node_indices_by_name: FxHashMap<NodeName, NodeIndex>,
    output_cache: RwLock<FxHashMap<NodeIndex, FxHashMap<OutputName, Signal<T>>>>,
    graph_inputs: FxHashMap<InputName, NodeIndex>,
    graph_outputs: FxHashMap<OutputName, NodeIndex>,
}

impl Graph<AudioRate> {
    pub fn new(
        inputs: FxHashMap<InputName, Input<AudioRate>>,
        outputs: FxHashMap<OutputName, Output>,
    ) -> Graph<AudioRate> {
        let mut this = Self {
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            output_cache: RwLock::new(FxHashMap::default()),
            graph_inputs: FxHashMap::default(),
            graph_outputs: FxHashMap::default(),
        };
        for (inp_name, _inp) in inputs {
            let (an, _cn) = GraphInput::create_nodes();
            let idx = this.add_node(an, &inp_name.0);
            this.node_indices_by_name
                .insert(NodeName(inp_name.0.to_owned()), idx);
            this.graph_inputs.insert(inp_name, idx);
        }
        for (out_name, _out) in outputs {
            let (an, _cn) = GraphOutput::create_nodes();
            let idx = this.add_node(an, &out_name.0);
            this.node_indices_by_name
                .insert(NodeName(out_name.0.to_owned()), idx);
            this.graph_outputs.insert(out_name, idx);
        }

        this
    }
}

impl Graph<ControlRate> {
    pub fn new(
        inputs: FxHashMap<InputName, Input<ControlRate>>,
        outputs: FxHashMap<OutputName, Output>,
    ) -> Graph<ControlRate> {
        let mut this = Self {
            digraph: DiGraph::new(),
            node_indices_by_name: FxHashMap::default(),
            output_cache: RwLock::new(FxHashMap::default()),
            graph_inputs: FxHashMap::default(),
            graph_outputs: FxHashMap::default(),
        };
        for (inp_name, _inp) in inputs {
            let (_an, cn) = GraphInput::create_nodes();
            let idx = this.add_node(cn, &inp_name.0);
            this.node_indices_by_name
                .insert(NodeName(inp_name.0.to_owned()), idx);
            this.graph_inputs.insert(inp_name, idx);
        }
        for (out_name, _out) in outputs {
            let (_an, cn) = GraphOutput::create_nodes();
            let idx = this.add_node(cn, &out_name.0);
            this.node_indices_by_name
                .insert(NodeName(out_name.0.to_owned()), idx);
            this.graph_outputs.insert(out_name, idx);
        }

        this
    }
}

impl<T: SignalType + 'static> Graph<T>
where
    Self: Processor<T>,
{
    pub fn add_node(&mut self, node: Arc<Node<T>>, name: &str) -> NodeIndex {
        let outs = node
            .outputs
            .keys()
            .map(|out_name| (out_name.to_owned(), Signal::new(0.0)))
            .collect();
        let idx = self.digraph.add_node(node);
        self.output_cache.write().unwrap().insert(idx, outs);
        self.node_indices_by_name
            .insert(NodeName(name.to_owned()), idx);
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

    pub fn get_output_id(&self, name: &OutputName) -> Option<NodeIndex> {
        self.graph_outputs.get(name).copied()
    }

    pub fn get_input_id(&self, name: &InputName) -> Option<NodeIndex> {
        self.graph_inputs.get(name).copied()
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
        if self.digraph.node_count() == 0 {
            return;
        }

        for (input_name, value) in inputs.iter() {
            let inp_idx = self.graph_inputs[input_name];
            *self.digraph[inp_idx]
                .inputs_cache
                .write()
                .unwrap()
                .get_mut(&InputName("in".to_owned()))
                .unwrap() = *value;
        }

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
        for node in self.graph_inputs.values() {
            bfs.stack.push_back(*node);
        }

        while let Some(node_id) = bfs.next(&self.digraph) {
            for edge in self.digraph.edges_directed(node_id, Direction::Incoming) {
                let out = {
                    self.output_cache.read().unwrap()[&edge.source()][&edge.weight().source_output]
                };
                *self.digraph[node_id]
                    .inputs_cache
                    .write()
                    .unwrap()
                    .get_mut(&edge.weight().sink_input)
                    .unwrap() = out;
            }
            let inps = self.digraph[node_id].inputs_cache.read().unwrap().clone();
            let node = &self.digraph[node_id];
            node.processor.process(
                t,
                sample_rate,
                node.sibling_node.as_ref(),
                &inps,
                self.output_cache
                    .write()
                    .unwrap()
                    .get_mut(&node_id)
                    .unwrap(),
            )
        }

        for (out_name, out) in outputs.iter_mut() {
            let node_idx = self.graph_outputs[out_name];
            *out = self.output_cache.read().unwrap()[&node_idx][&OutputName("out".to_owned())];
        }
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

pub trait CreateNodes {
    fn create_nodes() -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>);
}
