use std::sync::Arc;



use crate::{
    graph::{AudioNode, ControlNode, ControlOutput},
    Scalar,
};

pub mod basic;
pub mod generators;

pub trait SignalImpl {
    fn value(self) -> Scalar;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct AudioSignal(pub Scalar);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ControlSignal(pub Scalar);

impl SignalImpl for AudioSignal {
    fn value(self) -> Scalar {
        self.0
    }
}

impl SignalImpl for ControlSignal {
    fn value(self) -> Scalar {
        self.0
    }
}

impl From<Scalar> for AudioSignal {
    fn from(value: Scalar) -> Self {
        Self(value)
    }
}

impl From<Scalar> for ControlSignal {
    fn from(value: Scalar) -> Self {
        Self(value)
    }
}

pub trait AudioProcessor
where
    Self: Send + Sync,
{
    fn into_node(self) -> AudioNode;

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
    fn process_control(&self, t: Scalar, inputs: &[ControlSignal], outputs: &[ControlOutput]);
}
