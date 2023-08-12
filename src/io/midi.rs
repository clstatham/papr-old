use std::sync::{Arc, Mutex, RwLock};

use midir::{MidiInput, MidiInputConnection};
use midly::live::LiveEvent;
use spin::Once;

use crate::{
    dsp::{Processor, Signal, SignalRate},
    graph::{Node, NodeName, Output},
    Scalar,
};

pub static MIDI_CHAN: Once<crossbeam_channel::Receiver<Vec<u8>>> = Once::INIT;

const POLYPHONY: usize = 1;

pub struct MidiContext {
    _conn_in: MidiInputConnection<()>,
}

impl MidiContext {
    pub fn new(name: &str) -> Self {
        let mut midi_in = MidiInput::new(name).unwrap();
        midi_in.ignore(midir::Ignore::None);
        let in_ports = midi_in.ports();
        let port = &in_ports[0];
        let (tx, rx) = crossbeam_channel::unbounded();
        let _conn_in = midi_in
            .connect(
                port,
                name,
                move |_, message, _| {
                    tx.send(message.to_vec()).unwrap();
                },
                (),
            )
            .unwrap();
        MIDI_CHAN.call_once(|| rx);
        Self { _conn_in }
    }
}

pub struct NoteIn {
    notes: [Option<(u8, u8)>; POLYPHONY],
}

impl Processor for NoteIn {
    fn process_control_sample(
        &mut self,
        _buffer_idx: usize,
        _sample_rate: Scalar,
        _inputs: &[Signal],
        outputs: &mut [Signal],
    ) {
        if let Ok(msg) = MIDI_CHAN.get().unwrap().try_recv() {
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
                    midly::MidiMessage::NoteOff { key, vel: _ } => {
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
    pub fn create_node(name: &str, audio_buffer_len: usize) -> Arc<Node> {
        let this = Box::new(RwLock::new(Self {
            notes: [None; POLYPHONY],
        }));

        Arc::new(Node::new(
            NodeName::new(name),
            SignalRate::Control,
            audio_buffer_len,
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
        ))
    }
}
