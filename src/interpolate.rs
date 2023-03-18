use ndarray::{s, Array, Array1, Dimension};

trait IsMonotonic {
    fn is_monotonic(&self) -> bool {
        self.is_rising() || self.is_falling()
    }
    fn is_strict_monotonic(&self) -> bool {
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
        self.len() > 1
            && !self
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
        self.len() > 1
            && !self
                .windows(2)
                .into_iter()
                .map(|items| {
                    items.get(0).unwrap_or_else(|| unreachable!())
                        >= items.get(1).unwrap_or_else(|| unreachable!())
                })
                .any(|b| !b)
    }

    fn is_strict_rising(&self) -> bool {
        self.len() > 1
            && !self
                .windows(2)
                .into_iter()
                .map(|items| {
                    items.get(0).unwrap_or_else(|| unreachable!())
                        < items.get(1).unwrap_or_else(|| unreachable!())
                })
                .any(|b| !b)
    }

    fn is_strict_falling(&self) -> bool {
        self.len() > 1
            && !self
                .windows(2)
                .into_iter()
                .map(|items| {
                    items.get(0).unwrap_or_else(|| unreachable!())
                        > items.get(1).unwrap_or_else(|| unreachable!())
                })
                .any(|b| !b)
    }
}

#[cfg(test)]
mod test {
    use std::{cell::BorrowError, fmt::Debug, result};

    use ndarray::{array, Array1};

    use crate::interpolate::IsMonotonic;

    #[derive(Debug, PartialEq, Eq)]
    struct MonotonicTestResult {
        pub monotonic: bool,
        pub strict: bool,
        pub rising: bool,
        pub falling: bool,
        pub strict_rising: bool,
        pub strict_falling: bool,
    }

    impl MonotonicTestResult {
        fn strict_rising() -> Self {
            Self {
                monotonic: true,
                strict: true,
                rising: true,
                falling: false,
                strict_rising: true,
                strict_falling: false,
            }
        }
        fn strict_falling() -> Self {
            Self {
                monotonic: true,
                strict: true,
                rising: false,
                falling: true,
                strict_rising: false,
                strict_falling: true,
            }
        }
        fn rising() -> Self {
            Self {
                monotonic: true,
                strict: false,
                rising: true,
                falling: false,
                strict_rising: false,
                strict_falling: false,
            }
        }
        fn falling() -> Self {
            Self {
                monotonic: true,
                strict: false,
                rising: false,
                falling: true,
                strict_rising: false,
                strict_falling: false,
            }
        }
        fn not_monotonic() -> Self {
            Self {
                monotonic: false,
                strict: false,
                rising: false,
                falling: false,
                strict_rising: false,
                strict_falling: false,
            }
        }
    }

    #[derive(Debug)]
    struct MonotonicTestsCase<T: IsMonotonic> {
        pub data: T,
        pub expected: MonotonicTestResult,
    }

    fn test_monotonic_data<T: IsMonotonic + Debug>(
        MonotonicTestsCase { data, expected }: MonotonicTestsCase<T>,
    ) {
        let result = MonotonicTestResult {
            monotonic: data.is_monotonic(),
            strict: data.is_strict_monotonic(),
            rising: data.is_rising(),
            falling: data.is_falling(),
            strict_rising: data.is_strict_rising(),
            strict_falling: data.is_strict_falling(),
        };
        assert_eq!(result, expected, "IsMonotonic test failed: \nexpected result {expected:?}\n found result {result:?}\n for data {data:?} ")
    }

    #[test]
    fn monotonic_int() {
        test_monotonic_data(MonotonicTestsCase {
            data: array![1, 2, 2, 3, 4, 4],
            expected: MonotonicTestResult::rising(),
        });
        test_monotonic_data(MonotonicTestsCase {
            data: array![1, 2, 3, 4],
            expected: MonotonicTestResult::strict_rising(),
        });
        test_monotonic_data(MonotonicTestsCase {
            data: array![4, 4, 3, 2, 2, 1, -1, -2, -2],
            expected: MonotonicTestResult::falling(),
        });
        test_monotonic_data(MonotonicTestsCase {
            data: array![4, 3, 1, -2],
            expected: MonotonicTestResult::strict_falling(),
        });
        test_monotonic_data(MonotonicTestsCase {
            data: array![1, 2, 1],
            expected: MonotonicTestResult::not_monotonic(),
        });
        test_monotonic_data(MonotonicTestsCase {
            data: array![1],
            expected: MonotonicTestResult::not_monotonic(),
        });
    }
}
