use std::{
    collections::VecDeque,
    sync::{Arc, RwLock},
};

use miette::{Diagnostic, Result};
use petgraph::{dot::Dot, prelude::*};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::dsp::{
    graph_util::{GraphInput, GraphOutput},
    Processor, Signal, SignalRate,
};

#[derive(Debug, Error, Diagnostic)]
pub enum GraphError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("DSP error: {0}")]
    Dsp(#[from] crate::dsp::DspError),
    #[error("Node not found: {name}")]
    NodeNotFound { name: String },
    #[error("Couldn't acquire processor mutex lock")]
    MutexLock,
}

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

#[derive(Clone, PartialEq, Eq)]
pub struct Connection {
    pub source_output: usize,
    pub sink_input: usize,
}

impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.source_output, self.sink_input)
    }
}

#[derive(Clone)]
#[non_exhaustive]
pub struct Input {
    pub name: String,
    pub minimum: Option<Signal>,
    pub maximum: Option<Signal>,
    pub default: Option<Signal>,
    pub implicit: bool,
    pub is_ui: bool,
}

impl Input {
    pub fn new(name: &str, default: Option<Signal>) -> Self {
        Self {
            name: name.to_owned(),
            minimum: None,
            maximum: None,
            default,
            implicit: false,
            is_ui: false,
        }
    }

    pub fn new_ui(name: &str, minimum: Signal, maximum: Signal, default: Signal) -> Self {
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

pub enum ProcessorType
where
    Graph: Processor,
{
    Builtin(Box<RwLock<dyn Processor + Send + Sync>>),
    Subgraph(RwLock<Graph>),
    None,
}

impl ProcessorType
where
    Graph: Processor,
{
    pub fn process_sample(
        &self,
        buffer_idx: usize,
        signal_rate: SignalRate,
        inputs: &[Signal],
        outputs: &mut [Signal],
    ) -> Result<()> {
        match self {
            Self::Builtin(p) => p
                .write()
                .map_err(|_| GraphError::MutexLock)?
                .process_sample(buffer_idx, signal_rate, inputs, outputs),
            Self::Subgraph(p) => p
                .write()
                .map_err(|_| GraphError::MutexLock)?
                .process_sample(buffer_idx, signal_rate, inputs, outputs),
            Self::None => Ok(()),
        }
    }

    pub fn process_buffer(
        &self,
        signal_rate: SignalRate,
        inputs: &[Vec<Signal>],
        outputs: &mut [Vec<Signal>],
    ) -> Result<()> {
        match self {
            Self::Builtin(p) => p
                .write()
                .map_err(|_| GraphError::MutexLock)?
                .process_buffer(signal_rate, inputs, outputs),
            Self::Subgraph(p) => p
                .write()
                .map_err(|_| GraphError::MutexLock)?
                .process_buffer(signal_rate, inputs, outputs),
            Self::None => Ok(()),
        }
    }

    // ui_update is called from App::update which is allowed to panic
    #[allow(clippy::unwrap_used)]
    pub fn ui_update(&self, ui: &mut eframe::egui::Ui) {
        match self {
            Self::Builtin(p) => p.write().unwrap().ui_update(ui),
            Self::Subgraph(p) => p.write().unwrap().ui_update(ui),
            Self::None => {}
        }
    }
}

#[non_exhaustive]
pub struct Node
where
    Graph: Processor,
{
    pub name: NodeName,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub processor: ProcessorType,
}

impl std::fmt::Debug for Node
where
    Graph: Processor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name.0)
    }
}

impl Node
where
    Graph: Processor,
{
    pub fn new(
        name: NodeName,
        mut inputs: Vec<Input>,
        outputs: Vec<Output>,
        processor: ProcessorType,
    ) -> Self {
        if !inputs.iter().any(|i| i.name == "t") {
            let mut t = Input::new("t", None);
            t.implicit = true;
            inputs.push(t);
        }
        Self {
            name,
            inputs,
            outputs,
            processor,
        }
    }

    pub fn from_graph(graph: Graph) -> Self {
        Self::new(
            graph.name.clone(),
            graph
                .input_node_indices
                .iter()
                .map(|idx| {
                    let node = &graph.digraph[*idx];
                    Input::new(&node.name.0, node.inputs[0].default.clone())
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
            ProcessorType::Subgraph(RwLock::new(graph)),
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

pub struct Graph
where
    Self: Processor,
{
    pub name: NodeName,
    pub digraph: DiGraph<Arc<Node>, Connection>,
    node_indices_by_name: BTreeMap<String, NodeIndex>,
    pub input_node_indices: Vec<NodeIndex>,
    pub output_node_indices: Vec<NodeIndex>,
}

impl Graph {
    pub fn new(name: Option<NodeName>, mut inputs: Vec<Input>, outputs: Vec<Output>) -> Graph {
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
            digraph: DiGraph::new(),
            node_indices_by_name: BTreeMap::default(),
            input_node_indices: Vec::default(),
            output_node_indices: Vec::default(),
        };

        for inp in inputs {
            let an = GraphInput::create_node(&inp.name, inp.clone());
            let idx = this.add_node(an);
            this.node_indices_by_name.insert(inp.name.to_owned(), idx);
            this.input_node_indices.push(idx);
        }
        for out in outputs {
            let node = GraphOutput::create_node(&out.name);
            let idx = this.add_node(node);
            this.node_indices_by_name.insert(out.name.to_owned(), idx);
            this.output_node_indices.push(idx);
        }

        this
    }
}

impl Graph
where
    Self: Processor,
    Signal: std::fmt::Debug,
{
    pub fn write_dot(&self, name: &str) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(name).map_err(GraphError::Io)?;
        write!(f, "{:?}", Dot::with_config(&self.digraph, &[])).map_err(GraphError::Io)?;
        Ok(())
    }

    pub fn add_node(&mut self, node: Arc<Node>) -> NodeIndex {
        let name = node.name.to_owned();
        let idx = self.digraph.add_node(node);
        self.node_indices_by_name.insert(name.to_string(), idx);
        idx
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: Connection,
    ) -> EdgeIndex {
        let idx = self.digraph.add_edge(source, sink, connection);
        idx
    }

    pub fn node_id_by_name(&self, name: &str) -> Result<NodeIndex> {
        let id = self
            .node_indices_by_name
            .get(name)
            .copied()
            .ok_or_else(|| GraphError::NodeNotFound {
                name: name.to_owned(),
            })?;
        Ok(id)
    }

    pub fn process_graph(
        &mut self,
        signal_rate: SignalRate,
        inputs: &BTreeMap<NodeIndex, &Vec<Signal>>,
        outputs: &mut BTreeMap<NodeIndex, &mut Vec<Signal>>,
    ) -> Result<()> {
        macro_rules! assign {
            ($a:expr, $b:expr) => {
                $a[..$b.len()].clone_from_slice($b)
            };
        }

        // early check for empty graph (nothing to do)
        if self.digraph.node_count() == 0 {
            return Ok(());
        }

        let mut inputs_cache = BTreeMap::default();
        for (id, inp) in inputs {
            let inps = inputs_cache.entry(*id).or_insert(vec![
                vec![
                    Signal::Scalar(0.0);
                    signal_rate.buffer_len()
                ];
                self.digraph[*id].inputs.len()
            ]);
            assign!(inps[0], inp);
        }
        let mut outputs_cache = BTreeMap::default();

        // walk the BFS...
        let mut bfs = Bfs::new(&self.digraph, self.node_id_by_name("t")?);
        for node in self.digraph.externals(Direction::Incoming) {
            if !bfs.stack.contains(&node) {
                bfs.stack.push_back(node);
            }
        }
        // for layer in self.partitions.clone().iter() {
        while let Some(node_id) = bfs.next(&self.digraph) {
            // println!(
            //     "visiting node {} ({})",
            //     node_id.index(),
            //     self.digraph[node_id].name
            // );

            // for each incoming connection into the visited node:
            // - grab the cached outputs from earlier in the graph
            // - copy them to the input cache of the currently visited node
            for edge in self.digraph.edges_directed(node_id, Direction::Incoming) {
                let out = {
                    outputs_cache.entry(edge.source()).or_insert(vec![
                        vec![
                            Signal::Scalar(0.0);
                            signal_rate.buffer_len()
                        ];
                        self.digraph[edge.source()]
                            .outputs
                            .len()
                    ])
                };
                assign!(
                    inputs_cache.entry(node_id).or_insert(vec![
                        vec![
                            Signal::Scalar(0.0);
                            signal_rate.buffer_len()
                        ];
                        self.digraph[node_id].inputs.len()
                    ])[edge.weight().sink_input],
                    &out[edge.weight().source_output]
                );
            }

            // create a copy of the inputs from the cache (necessary because we mutably borrow `self` in the next step)
            let inps =
                inputs_cache.entry(node_id).or_insert(vec![
                    vec![
                        Signal::Scalar(0.0);
                        signal_rate.buffer_len()
                    ];
                    self.digraph[node_id].inputs.len()
                ]);

            // manually copy over `t` since it's implicit
            assign!(
                inps[self.digraph[node_id]
                    .input_named("t")
                    .ok_or(GraphError::Dsp(crate::dsp::DspError::NoInputNamed(
                        "t".into()
                    )))?],
                inputs[&self.node_id_by_name("t")?]
            );

            let outs =
                outputs_cache.entry(node_id).or_insert(vec![
                    vec![
                        Signal::Scalar(0.0);
                        signal_rate.buffer_len()
                    ];
                    self.digraph[node_id].outputs.len()
                ]);

            // run the processing logic for this node, which will store its results directly in our output cache
            self.digraph[node_id]
                .processor
                .process_buffer(signal_rate, inps, outs)?;
        }
        // }

        // copy the cached (and now updated) output values into the mutable passed outputs
        for (out_name, out) in outputs.iter_mut() {
            out.clone_from_slice(&outputs_cache[out_name][0]);
        }
        Ok(())
    }

    pub fn into_node(self) -> Node {
        Node::from_graph(self)
    }
}

impl Processor for Graph {
    fn process_sample(
        &mut self,
        _buffer_idx: usize,
        _signal_rate: SignalRate,
        _inputs: &[Signal],
        _outputs: &mut [Signal],
    ) -> Result<()> {
        unimplemented!()
    }
    fn process_buffer(
        &mut self,
        signal_rate: SignalRate,
        inputs: &[Vec<Signal>],
        outputs: &mut [Vec<Signal>],
    ) -> Result<()> {
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
        self.process_graph(signal_rate, &inputs, &mut outputs)
    }
}
