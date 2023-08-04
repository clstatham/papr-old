use std::sync::{Arc, RwLock};

use petgraph::prelude::*;
use rustc_hash::FxHashMap;
use tokio::sync::watch;

use crate::{
    dsp::{AudioSignal, ControlSignal, SignalImpl},
    Scalar, PI,
};

#[derive(Clone, Copy)]
pub struct AudioConnection {
    pub source_output_index: usize,
    pub sink_input_index: usize,
}

pub struct AudioInput {
    pub name: Option<String>,
    pub default: AudioSignal,
}

#[derive(Clone)]
pub struct AudioOutput {
    pub name: Option<String>,
}

pub struct AudioNode {
    pub control_node: Arc<ControlNode>,
    pub inputs: Vec<AudioInput>,
    pub outputs: Vec<AudioOutput>,
    pub processor: Box<dyn AudioProcessor>,
}

pub struct AudioGraph {
    digraph: DiGraph<AudioNode, AudioConnection>,
    input_cache: FxHashMap<NodeIndex, Vec<AudioSignal>>,
    output_cache: FxHashMap<NodeIndex, Vec<AudioSignal>>,
    dac_nodes: Vec<NodeIndex>,
}

impl AudioGraph {
    pub fn new() -> Self {
        Self {
            digraph: DiGraph::new(),
            input_cache: FxHashMap::default(),
            output_cache: FxHashMap::default(),
            dac_nodes: Vec::new(),
        }
    }

    pub fn add_dac(&mut self) -> NodeIndex {
        let idx = self.digraph.add_node(AudioNode {
            inputs: vec![AudioInput {
                name: Some("in".to_owned()),
                default: AudioSignal(0 as Scalar),
            }],
            outputs: vec![AudioOutput {
                name: Some("out".to_owned()),
            }],
            control_node: Arc::new(ControlNode {
                inputs: vec![],
                params: FxHashMap::default(),
                outputs: vec![],
                processor: Box::new(DummyControlNode),
            }),
            processor: Box::new(Dac),
        });
        self.input_cache
            .insert(idx, vec![AudioSignal(0 as Scalar); 1]);
        self.output_cache
            .insert(idx, vec![AudioSignal(0 as Scalar); 1]);
        self.dac_nodes.push(idx);
        idx
    }

    pub fn add_node(&mut self, node: AudioNode) -> NodeIndex {
        let n_outs = node.outputs.len();
        let ins = node.inputs.iter().map(|inp| inp.default).collect();
        let idx = self.digraph.add_node(node);
        self.input_cache.insert(idx, ins);
        self.output_cache
            .insert(idx, vec![AudioSignal(0 as Scalar); n_outs]);
        idx
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: AudioConnection,
    ) -> EdgeIndex {
        self.digraph.add_edge(source, sink, connection)
    }

    pub fn process_audio(
        &mut self,
        t: Scalar,
        inputs: &[AudioSignal],
        outputs: &mut [AudioSignal],
    ) {
        if self.digraph.node_count() == 0 {
            return;
        }
        let mut starts = self.digraph.externals(petgraph::Direction::Incoming);
        let mut bfs = Bfs::new(
            &self.digraph,
            starts
                .next()
                .expect("AudioGraph::process_audio(): graph has no input/source nodes"),
        );
        for node in starts {
            bfs.stack.push_back(node);
        }
        // todo: push back audio inputs from hardware (adc)

        while let Some(node_id) = bfs.next(&self.digraph) {
            // let inputs =
            for edge in self.digraph.edges_directed(node_id, Direction::Incoming) {
                let out = self.output_cache[&edge.source()][edge.weight().source_output_index];
                self.input_cache.get_mut(&node_id).unwrap()[edge.weight().sink_input_index] = out;
            }
            let node = &mut self.digraph[node_id];
            node.processor.process_audio(
                t,
                &self.input_cache[&node_id],
                &node.control_node,
                self.output_cache.get_mut(&node_id).unwrap(),
            )
        }

        for (i, dac_idx) in self.dac_nodes.iter().enumerate() {
            outputs[i] = self.output_cache[dac_idx][0];
        }
    }
}

pub struct ControlConnection {
    pub source_output_index: usize,
    pub sink_input_index: usize,
}

pub struct ControlInput {
    pub name: Option<String>,
    pub rx: RwLock<Option<watch::Receiver<ControlSignal>>>,
    pub default: ControlSignal,
}

impl ControlInput {
    pub fn new(name: Option<&str>, default: ControlSignal) -> Self {
        Self {
            name: name.map(|s| s.to_owned()),
            rx: RwLock::new(None),
            default,
        }
    }

    pub fn connect(&self, rx: watch::Receiver<ControlSignal>) {
        *self
            .rx
            .write()
            .expect("ControlInput::connect(): couldn't acquire self.rx write lock") = Some(rx);
    }
}

pub struct ControlOutput {
    pub name: Option<String>,
    pub tx: watch::Sender<ControlSignal>,
}

impl ControlOutput {
    pub fn new(name: Option<&str>) -> Self {
        let (tx, _rx) = watch::channel(ControlSignal(0 as Scalar));
        Self {
            name: name.map(|s| s.to_owned()),
            tx,
        }
    }
    pub fn subscribe(&self) -> watch::Receiver<ControlSignal> {
        self.tx.subscribe()
    }
}

pub struct ControlNode {
    pub inputs: Vec<ControlInput>,
    pub params: FxHashMap<String, ControlSignal>,
    pub outputs: Vec<ControlOutput>,
    pub processor: Box<dyn ControlProcessor>,
}

pub struct ControlGraph {
    digraph: DiGraph<Arc<ControlNode>, ControlConnection>,
}

impl ControlGraph {
    pub fn new() -> Self {
        Self {
            digraph: DiGraph::new(),
        }
    }

    pub fn add_node(&mut self, node: ControlNode) -> NodeIndex {
        self.digraph.add_node(Arc::new(node))
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        connection: ControlConnection,
    ) -> EdgeIndex {
        let rx = self.digraph[source].outputs[connection.source_output_index].subscribe();
        self.digraph[sink].inputs[connection.sink_input_index].connect(rx);
        self.digraph.add_edge(source, sink, connection)
    }

    pub fn process_control(&mut self, t: Scalar) {
        if self.digraph.node_count() == 0 {
            return;
        }
        let mut starts = self.digraph.externals(petgraph::Direction::Incoming);
        let mut bfs = Bfs::new(
            &self.digraph,
            starts
                .next()
                .expect("ControlGraph::process_control(): graph has no input/source nodes"),
        );
        for node in starts {
            bfs.stack.push_back(node);
        }

        while let Some(node) = bfs.next(&self.digraph) {
            let inputs = &self.digraph[node].inputs;
            let params = &self.digraph[node].params;
            let outputs = &self.digraph[node].outputs;
            self.digraph[node]
                .processor
                .process_control(t, inputs, params, outputs);
        }
    }
}

pub trait AudioProcessor
where
    Self: Send + Sync,
{
    fn process_audio(
        &mut self,
        t: Scalar,
        inputs: &[AudioSignal],
        control_node: &Arc<ControlNode>,
        outputs: &mut [AudioSignal],
    );
}

pub trait ControlProcessor
where
    Self: Send + Sync,
{
    fn process_control(
        &self,
        t: Scalar,
        inputs: &[ControlInput],
        params: &FxHashMap<String, ControlSignal>,
        outputs: &[ControlOutput],
    );
}

pub struct DummyControlNode;
impl ControlProcessor for DummyControlNode {
    fn process_control(
        &self,
        _t: Scalar,
        _inputs: &[ControlInput],
        _params: &FxHashMap<String, ControlSignal>,
        _outputs: &[ControlOutput],
    ) {
    }
}

pub struct DebugControlNode {
    pub name: &'static str,
}

impl ControlProcessor for DebugControlNode {
    fn process_control(
        &self,
        t: Scalar,
        inputs: &[ControlInput],
        _params: &FxHashMap<String, ControlSignal>,
        outputs: &[ControlOutput],
    ) {
        println!("Debug: {} (t={t})", self.name);

        for (inp, out) in inputs.iter().zip(outputs.iter()) {
            let val = inp
                .rx
                .read()
                .unwrap()
                .as_ref()
                .map(|rx| *rx.borrow())
                .unwrap_or(inp.default);
            println!(
                "{} = {}",
                inp.name.as_ref().unwrap_or(&"(unknown)".to_owned()),
                val.value()
            );
            out.tx.send_replace(val);
        }
    }
}

pub struct Dac;

impl AudioProcessor for Dac {
    fn process_audio(
        &mut self,
        t: Scalar,
        inputs: &[AudioSignal],
        control_node: &Arc<ControlNode>,
        outputs: &mut [AudioSignal],
    ) {
        // dbg!(inputs[0].value());
        outputs.copy_from_slice(inputs);
    }
}

pub struct SineOscA;
pub struct SineOscC;

impl AudioProcessor for SineOscA {
    fn process_audio(
        &mut self,
        t: Scalar,
        inputs: &[AudioSignal],
        control_node: &Arc<ControlNode>,
        outputs: &mut [AudioSignal],
    ) {
        outputs[0] = AudioSignal(Scalar::sin(t * PI * 2.0 * 440.0));
    }
}

impl ControlProcessor for SineOscC {
    fn process_control(
        &self,
        t: Scalar,
        inputs: &[ControlInput],
        params: &FxHashMap<String, ControlSignal>,
        outputs: &[ControlOutput],
    ) {
    }
}
