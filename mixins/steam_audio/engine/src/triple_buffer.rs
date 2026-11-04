//! Hands data from one thread to another without either waiting for the other (upstream `triple_buffer.h`), e.g.
//! impulse responses from the simulation thread to the audio thread.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Set in the shared state when the shared slot holds data the reader has not taken yet.
const NEW_DATA: u8 = 4;

/// Three slots of `T`: one the writer fills, one the reader reads, and one in between (upstream `TripleBuffer<T>`).
///
/// Upstream is one object whose `writeBuffer` and `readBuffer` two threads use; here it is split into a
/// [`TripleBufferWriter`] and a [`TripleBufferReader`], which can be sent to different threads. Each slot sits behind
/// a `Mutex` only to satisfy the borrow checker: the protocol gives each slot to one side at a time, so the locks are
/// never contended.
#[derive(Debug)]
struct Shared<T> {
    slots: [Mutex<T>; 3],
    /// Index of the shared slot, plus [`NEW_DATA`].
    state: AtomicU8,
}

/// The writing side of a triple buffer.
#[derive(Debug)]
pub struct TripleBufferWriter<T> {
    shared: Arc<Shared<T>>,
    write: u8,
}

/// The reading side of a triple buffer.
#[derive(Debug)]
pub struct TripleBufferReader<T> {
    shared: Arc<Shared<T>>,
    read: u8,
}

/// A triple buffer whose three slots are made by `init` (upstream `initBuffers`).
pub fn triple_buffer<T>(
    mut init: impl FnMut() -> T,
) -> (TripleBufferWriter<T>, TripleBufferReader<T>) {
    let shared = Arc::new(Shared {
        slots: [Mutex::new(init()), Mutex::new(init()), Mutex::new(init())],
        state: AtomicU8::new(1),
    });
    (
        TripleBufferWriter {
            shared: Arc::clone(&shared),
            write: 0,
        },
        TripleBufferReader { shared, read: 2 },
    )
}

fn lock<T>(slot: &Mutex<T>) -> MutexGuard<'_, T> {
    slot.lock().unwrap_or_else(|e| e.into_inner())
}

impl<T> TripleBufferWriter<T> {
    /// The slot being written (upstream `writeBuffer`).
    pub fn write_buffer(&mut self) -> MutexGuard<'_, T> {
        lock(&self.shared.slots[usize::from(self.write)])
    }

    /// Publishes the write slot and takes the shared slot to write into next (upstream `commitWriteBuffer`).
    ///
    /// As upstream, nothing happens while the reader has not taken the previous commit: the reader then gets that
    /// older data, and the writer keeps writing into the same slot.
    pub fn commit_write_buffer(&mut self) {
        let state = self.shared.state.load(Ordering::Acquire);
        if state & NEW_DATA == 0 {
            // Only this side changes the state while NEW_DATA is clear.
            self.shared
                .state
                .store(self.write | NEW_DATA, Ordering::Release);
            self.write = state;
        }
    }
}

impl<T> TripleBufferReader<T> {
    /// The slot being read (upstream `readBuffer`).
    pub fn read_buffer(&mut self) -> MutexGuard<'_, T> {
        lock(&self.shared.slots[usize::from(self.read)])
    }

    /// Takes the latest commit, if there is one the reader has not taken yet, and returns whether it did (upstream
    /// `updateReadBuffer`).
    pub fn update_read_buffer(&mut self) -> bool {
        let state = self.shared.state.load(Ordering::Acquire);
        if state & NEW_DATA == 0 {
            return false;
        }
        // Only this side changes the state while NEW_DATA is set.
        self.shared.state.store(self.read, Ordering::Release);
        self.read = state & !NEW_DATA;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hands_over_commits() {
        let (mut writer, mut reader) = triple_buffer(|| 0);
        assert!(!reader.update_read_buffer());

        *writer.write_buffer() = 1;
        writer.commit_write_buffer();
        assert!(reader.update_read_buffer());
        assert_eq!(*reader.read_buffer(), 1);
        assert!(!reader.update_read_buffer());
        assert_eq!(*reader.read_buffer(), 1);
    }

    #[test]
    fn commit_is_dropped_until_the_previous_one_is_read() {
        let (mut writer, mut reader) = triple_buffer(|| 0);
        *writer.write_buffer() = 1;
        writer.commit_write_buffer();
        *writer.write_buffer() = 2;
        writer.commit_write_buffer();
        assert!(reader.update_read_buffer());
        assert_eq!(*reader.read_buffer(), 1);

        // The writer kept its slot, still holding 2.
        assert_eq!(*writer.write_buffer(), 2);
        writer.commit_write_buffer();
        assert!(reader.update_read_buffer());
        assert_eq!(*reader.read_buffer(), 2);
    }

    #[test]
    fn values_arrive_in_order_across_threads() {
        let (mut writer, mut reader) = triple_buffer(|| 0u32);
        let handle = std::thread::spawn(move || {
            let mut next = 1;
            while next <= 10_000 {
                *writer.write_buffer() = next;
                let before = writer.write;
                writer.commit_write_buffer();
                if writer.write != before {
                    next += 1;
                }
            }
        });
        let mut last = 0;
        while last < 10_000 {
            if reader.update_read_buffer() {
                let value = *reader.read_buffer();
                assert!(value > last);
                last = value;
            }
        }
        handle.join().unwrap();
    }
}
