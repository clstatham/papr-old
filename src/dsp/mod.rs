use crate::Scalar;

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
