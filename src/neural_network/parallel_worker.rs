use crossbeam::thread::ScopedJoinHandle;
use std::ops::Range;

pub fn split_tasks_into_chunks<T>(tasks: &[T]) -> Vec<&[T]> {
    let cpus = num_cpus::get();
    let input_size = tasks.len();
    let tasks_per_thread = (input_size as f64 / cpus as f64).ceil() as usize;
    let mut task_ranges: Vec<&[T]> = Vec::with_capacity(cpus);
    for cpu in 0..cpus {
        let from = cpu * tasks_per_thread;
        let to = std::cmp::min((cpu + 1) * tasks_per_thread, input_size);
        task_ranges.push(&tasks[from..to]);
    }
    task_ranges
}

pub fn split_range_into_chunks(range: &Range<usize>) -> Vec<Range<usize>> {
    let cpus = num_cpus::get();
    let input_size = range.end - range.start;
    let tasks_per_thread = (input_size as f64 / cpus as f64).ceil() as usize;
    let mut task_ranges: Vec<Range<usize>> = Vec::with_capacity(cpus);
    for cpu in 0..cpus {
        let from = cpu * tasks_per_thread;
        let to = std::cmp::min((cpu + 1) * tasks_per_thread, input_size);
        task_ranges.push(from..to);
    }

    task_ranges
}

pub fn map<I, O, W>(input: &[I], f: W) -> Result<Vec<O>, ()>
where
    I: Send + Sync,
    O: Send + Sync,
    W: (Fn(&I) -> O) + Send + Sync + Copy,
{
    let jobs = split_tasks_into_chunks(input);

    crossbeam::scope(|s| {
        let mut threads: Vec<ScopedJoinHandle<Vec<O>>> = Vec::with_capacity(jobs.len());

        for job in jobs {
            let thread = s.spawn(move |_| {
                let mut out = Vec::with_capacity(job.len());
                for item in job {
                    out.push(f(item));
                }
                out
            });
            threads.push(thread);
        }

        let mut output: Vec<O> = Vec::with_capacity(input.len());
        for thread in threads {
            match thread.join() {
                Ok(data) => {
                    output.extend(data);
                }
                Err(_) => return Err(()),
            };
        }
        Ok(output)
    })
    .unwrap()
}

pub fn map_range<O, W>(range: &Range<usize>, f: W) -> Result<Vec<O>, ()>
where
    O: Send + Sync,
    W: (Fn(usize) -> O) + Send + Sync + Copy,
{
    let jobs = split_range_into_chunks(range);

    crossbeam::scope(|s| {
        let mut threads: Vec<ScopedJoinHandle<Vec<O>>> = Vec::with_capacity(jobs.len());

        for job in jobs {
            let thread = s.spawn(move |_| {
                let mut out = Vec::with_capacity(job.len());
                for item in job {
                    out.push(f(item));
                }
                out
            });
            threads.push(thread);
        }

        let mut output: Vec<O> = Vec::with_capacity(range.len());
        for thread in threads {
            match thread.join() {
                Ok(data) => {
                    output.extend(data);
                }
                Err(_) => return Err(()),
            };
        }
        Ok(output)
    })
    .unwrap()
}
