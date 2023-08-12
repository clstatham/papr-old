use std::sync::{Arc, RwLock};

use midir::{MidiInput, MidiInputConnection};
use midly::live::LiveEvent;

use crate::{
    dsp::{basic::Dummy, AudioRate, ControlRate, Processor, Signal},
    graph::{Node, NodeName, Output},
    Scalar,
};

const POLYPHONY: usize = 1;

pub struct NoteIn {
    conn_in: MidiInputConnection<()>,
    chan: crossbeam_channel::Receiver<Vec<u8>>,
    notes: [Option<(u8, u8)>; POLYPHONY],
}

impl NoteIn {
    pub fn new(name: &str) -> Self {
        let mut midi_in = MidiInput::new(&format!("PAPR MidiIn {name}")).unwrap();
        midi_in.ignore(midir::Ignore::None);
        let in_ports = midi_in.ports();
        let port = &in_ports[0];
        let (tx, rx) = crossbeam_channel::unbounded();
        let conn_in = midi_in
            .connect(
                port,
                name,
                move |_, message, _| {
                    tx.send(message.to_vec()).unwrap();
                },
                (),
            )
            .unwrap();
        Self {
            conn_in,
            chan: rx,
            notes: [None; POLYPHONY],
        }
    }
}

// FIXME: this might break stuff, it should be fine for now though
// since we only have one control thread and the audio thread doesn't
// access anything for this node
unsafe impl Sync for NoteIn {}

impl Processor<ControlRate> for NoteIn {
    fn process_sample(
        &mut self,
        buffer_idx: usize,
        sample_rate: crate::Scalar,
        sibling_node: Option<&std::sync::Arc<<ControlRate as crate::dsp::SignalRate>::SiblingNode>>,
        inputs: &[crate::dsp::Signal<ControlRate>],
        outputs: &mut [crate::dsp::Signal<ControlRate>],
    ) {
        if let Ok(msg) = self.chan.try_recv() {
            let msg = LiveEvent::parse(&msg).unwrap();
            if let LiveEvent::Midi {
                channel: _,
                message,
            } = msg
            {
                match message {
                    midly::MidiMessage::NoteOn { key, vel } => {
                        if let Some(idx) = self.notes.iter_mut().find_map(|idx| {
                            if idx.is_none() {
                                Some(idx)
                            } else {
                                None
                            }
                        }) {
                            *idx = Some((key.as_int(), vel.as_int()));
                        }
                    }
                    midly::MidiMessage::NoteOff { key, vel } => {
                        if let Some(idx) = self.notes.iter_mut().find(|idx| {
                            if let Some((k, _)) = idx {
                                *k == key.as_int()
                            } else {
                                false
                            }
                        }) {
                            *idx = None;
                        }
                    }
                    _ => {}
                }
            }
        }
        for i in 0..POLYPHONY {
            let (key, vel) = self.notes[i].unwrap_or((0, 0));
            outputs[i] = Signal::new(key as Scalar);
            outputs[i + POLYPHONY] = Signal::new(vel as Scalar / 127.0);
        }
    }
}

impl NoteIn {
    pub fn create_nodes(
        name: &str,
        audio_buffer_len: usize,
    ) -> (Arc<Node<AudioRate>>, Arc<Node<ControlRate>>) {
        let this = Arc::new(RwLock::new(Self::new(name)));
        let cn = Arc::new(Node::new(
            NodeName::new(name),
            1,
            vec![],
            (0..POLYPHONY)
                .map(|i| Output {
                    name: format!("note{}", i),
                })
                .chain((0..POLYPHONY).map(|i| Output {
                    name: format!("vel{}", i),
                }))
                .collect(),
            crate::graph::ProcessorType::Builtin(this),
            None,
        ));
        let an = Arc::new(Node::new(
            NodeName::new(name),
            audio_buffer_len,
            vec![],
            vec![],
            crate::graph::ProcessorType::Builtin(Arc::new(RwLock::new(Dummy))),
            None,
        ));
        (an, cn)
    }
}
