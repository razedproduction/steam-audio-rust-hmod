//! Runs the jobs of a [`JobGraph`] on several threads (upstream `thread_pool.{h,cpp}`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};
use std::thread;

use crate::job_graph::JobGraph;

/// A fixed number of threads that process [`JobGraph`]s (upstream `ThreadPool`).
///
/// Upstream keeps its threads alive between calls. Jobs here may borrow from the caller, so each call to
/// [`process`](Self::process) runs them on scoped threads that end with the call; which thread runs which job is
/// decided the same way, by taking the next job in order whenever a thread is free.
#[derive(Debug)]
pub struct ThreadPool {
    num_threads: usize,
    cancel: AtomicBool,
}

impl ThreadPool {
    /// # Panics
    ///
    /// If `num_threads` is 0.
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0, "a thread pool needs at least one thread");
        Self {
            num_threads,
            cancel: AtomicBool::new(false),
        }
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Runs every job of `graph`, on threads `0..num_threads`, and returns when they are done. The graph is empty
    /// afterwards.
    ///
    /// After [`cancel`](Self::cancel), each thread runs at most one more job (which sees the flag set), as upstream.
    pub fn process(&self, graph: &mut JobGraph<'_>) {
        self.process_with_progress(graph, |_| {});
    }

    /// [`process`](Self::process), calling `progress` on the calling thread with the fraction of jobs completed so far:
    /// once at the start and after every job (`0 / 0` when the graph is empty, as upstream).
    pub fn process_with_progress(&self, graph: &mut JobGraph<'_>, mut progress: impl FnMut(f32)) {
        let num_jobs = graph.jobs.len();
        let jobs = Mutex::new(graph.jobs.drain(..));
        let jobs_completed = AtomicUsize::new(0);
        let (sender, receiver) = mpsc::channel::<()>();

        let report = |progress: &mut dyn FnMut(f32)| {
            progress(jobs_completed.load(Ordering::SeqCst) as f32 / num_jobs as f32);
        };
        // Before any thread starts, so the first report is always 0.
        report(&mut progress);

        thread::scope(|scope| {
            for thread_id in 0..self.num_threads {
                let sender = sender.clone();
                let jobs = &jobs;
                let jobs_completed = &jobs_completed;
                scope.spawn(move || {
                    loop {
                        // The lock is held only to take the job.
                        let job = jobs.lock().unwrap_or_else(|e| e.into_inner()).next();
                        let Some(job) = job else { break };
                        job(thread_id, &self.cancel);

                        if self.cancel.load(Ordering::SeqCst) {
                            break;
                        }

                        jobs_completed.fetch_add(1, Ordering::SeqCst);
                        // The receiver lives until every sender is gone.
                        let _ = sender.send(());
                    }
                });
            }
            drop(sender);

            for () in receiver {
                report(&mut progress);
            }
        });
    }

    /// Asks running and future jobs to stop. The flag is never cleared, as upstream.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_job_runs_once_in_order_on_a_valid_thread() {
        for num_threads in [1, 3] {
            let pool = ThreadPool::new(num_threads);
            let order = Mutex::new(Vec::new());
            let mut graph = JobGraph::new();
            for i in 0..50 {
                let order = &order;
                graph.add_job(move |thread_id, _| {
                    assert!(thread_id < num_threads);
                    order.lock().unwrap().push(i);
                });
            }
            pool.process(&mut graph);
            assert!(graph.is_empty());
            drop(graph);
            let mut order = order.into_inner().unwrap();
            if num_threads == 1 {
                assert_eq!(order, (0..50).collect::<Vec<_>>());
            }
            order.sort_unstable();
            assert_eq!(order, (0..50).collect::<Vec<_>>());
        }
    }

    #[test]
    fn progress_is_reported_up_to_one() {
        let pool = ThreadPool::new(2);
        let mut graph = JobGraph::new();
        for _ in 0..8 {
            graph.add_job(|_, _| {});
        }
        let mut reports = Vec::new();
        pool.process_with_progress(&mut graph, |p| reports.push(p));
        assert!(reports.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(reports.first(), Some(&0.0));
        assert_eq!(reports.last(), Some(&1.0));
    }

    #[test]
    fn cancelled_pool_runs_at_most_one_job_per_thread() {
        let pool = ThreadPool::new(2);
        pool.cancel();
        let runs = AtomicUsize::new(0);
        let mut graph = JobGraph::new();
        for _ in 0..10 {
            graph.add_job(|_, cancel| {
                assert!(cancel.load(Ordering::SeqCst));
                runs.fetch_add(1, Ordering::SeqCst);
            });
        }
        pool.process(&mut graph);
        assert!(runs.load(Ordering::SeqCst) <= 2);
        assert!(graph.is_empty());
    }
}
