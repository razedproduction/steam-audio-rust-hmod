//! A list of jobs for a [`ThreadPool`](crate::thread_pool::ThreadPool) to run (upstream `job.h` and
//! `job_graph.{h,cpp}`).

use std::fmt;
use std::sync::atomic::AtomicBool;

/// A job: called once with the index of the thread running it and the pool's cancellation flag (upstream
/// `JobCallback`). Long jobs should return early once the flag is set.
pub type Job<'a> = Box<dyn FnOnce(usize, &AtomicBool) + Send + 'a>;

/// Jobs that a [`ThreadPool`](crate::thread_pool::ThreadPool) runs in the order they were added, each on whichever
/// thread is free next (upstream `JobGraph`). All jobs are added before the graph is processed.
///
/// Processing takes the jobs out of the graph, so it is empty afterwards (upstream keeps them until `reset`, but never
/// runs them again).
#[derive(Default)]
pub struct JobGraph<'a> {
    pub(crate) jobs: Vec<Job<'a>>,
}

impl<'a> JobGraph<'a> {
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Number of jobs waiting to be processed (upstream `getNumJobs`).
    pub fn num_jobs(&self) -> usize {
        self.jobs.len()
    }

    /// Removes every job.
    pub fn reset(&mut self) {
        self.jobs.clear();
    }

    pub fn add_job(&mut self, job: impl FnOnce(usize, &AtomicBool) + Send + 'a) {
        self.jobs.push(Box::new(job));
    }
}

impl fmt::Debug for JobGraph<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JobGraph")
            .field("num_jobs", &self.jobs.len())
            .finish()
    }
}
