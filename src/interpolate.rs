use ndarray::{Array, Array1, Dimension, s};

trait IsMonotonic {
    fn is_monotonic(&self) -> bool {
        self.is_rising() || self.is_falling()
    }
    fn is_strict_monotonic(&self)->bool{
        self.is_strict_rising() || self.is_strict_falling()
    }
    fn is_rising(&self) -> bool;
    fn is_falling(&self) -> bool;
    fn is_strict_rising(&self) -> bool;
    fn is_strict_falling(&self) -> bool;
}

impl<T: PartialOrd> IsMonotonic for Array1<T> {
    /// is this array monotonic rising
    fn is_rising(&self) -> bool {
        self.len() > 1 &&
        !self
            .windows(2)
            .into_iter()
            .map(|items| {
                items.get(0).unwrap_or_else(|| unreachable!())
                    <= items.get(1).unwrap_or_else(|| unreachable!())
            })
            .any(|b| !b)
    }
    /// is this array monotonic falling
    fn is_falling(&self) -> bool {
        self.len() > 1 &&
        !self
            .windows(2)
            .into_iter()
            .map(|items| {
                items.get(0).unwrap_or_else(|| unreachable!())
                    >= items.get(1).unwrap_or_else(|| unreachable!())
            })
            .any(|b| !b)
    }

    fn is_strict_rising(&self) -> bool {
        self.len() > 1 &&
        !self
            .windows(2)
            .into_iter()
            .map(|items| {
                items.get(0).unwrap_or_else(|| unreachable!())
                    < items.get(1).unwrap_or_else(|| unreachable!())
            })
            .any(|b| !b)
    }

    fn is_strict_falling(&self) -> bool {
        self.len() > 1 &&
        !self
            .windows(2)
            .into_iter()
            .map(|items| {
                items.get(0).unwrap_or_else(|| unreachable!())
                    > items.get(1).unwrap_or_else(|| unreachable!())
            })
            .any(|b| !b)
    }
}
