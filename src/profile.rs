use crate::{kernel, Context};
use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
#[derive(Clone, Copy)]
struct KernelStatsIntermediate {
    n_launches: u64,
    total_time: f64, // nano seconds
    max_time: f64,   // nano seconds
}
#[derive(Clone, Copy, Debug)]
pub struct KernelStats {
    pub n_launches: u64,
    pub total_time_msec: f64,
    pub max_time_msec: f64,
    pub avg_time_msec: f64,
}
#[derive(Clone)]
struct KernelRecord {
    stats: KernelStatsIntermediate,
    queries: Vec<(u32, u32)>,
}
pub struct Profiler {
    ctx: Context,
    hash_map: HashMap<String, KernelRecord>,
    query_pool: vk::QueryPool,
    fences: VecDeque<Rc<crate::Fence>>,
    query_counter: u32,
    query_count: u32,
}

impl Profiler {
    pub fn all_stats(&self) -> Vec<(String, KernelStats)> {
        self.hash_map
            .iter()
            .map(|(kernel_name, record)| {
                let stats = &record.stats;
                (
                    kernel_name.clone(),
                    KernelStats {
                        n_launches: stats.n_launches,
                        avg_time_msec: (stats.total_time / stats.n_launches as f64) / 1e6,
                        total_time_msec: stats.total_time / 1e6,
                        max_time_msec: stats.max_time / 1e6,
                    },
                )
            })
            .collect()
    }
    pub fn stats_of(&self, kernel_name: &str) -> Option<KernelStats> {
        let stats = &self.hash_map.get(&String::from(kernel_name))?.stats;
        Some(KernelStats {
            n_launches: stats.n_launches,
            avg_time_msec: (stats.total_time / stats.n_launches as f64) / 1e6,
            total_time_msec: stats.total_time / 1e6,
            max_time_msec: stats.max_time / 1e6,
        })
    }
    pub fn new(ctx: &Context, query_count: u32) -> Self {
        unsafe {
            let query_pool = ctx
                .device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::builder()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(query_count),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();

            ctx.device.reset_query_pool(query_pool, 0, query_count);

            Self {
                fences: VecDeque::new(),
                ctx: ctx.clone(),
                hash_map: HashMap::new(),
                query_count,
                query_counter: 0,
                query_pool,
            }
        }
    }
    pub fn poll(&mut self) {
        for fence in &self.fences {
            fence.wait();
        }
        self.fences.clear();
        for (_, record) in &mut self.hash_map {
            let queries: Vec<(u32, u32)> = std::mem::replace(&mut record.queries, vec![]);
            record.stats.n_launches += queries.len() as u64;
            for queries in &queries {
                assert!(queries.1 == queries.0 + 1);
                unsafe {
                    let mut buf = [0u64; 2];
                    self.ctx
                        .device
                        .get_query_pool_results(
                            self.query_pool,
                            queries.0,
                            2,
                            &mut buf,
                            vk::QueryResultFlags::TYPE_64,
                        )
                        .unwrap();
                    let period = self.ctx.limits.timestamp_period as f64;
                    let exec_time = (buf[1] - buf[0]) as f64 * period;
                    record.stats.max_time = record.stats.max_time.max(exec_time);
                    record.stats.total_time += exec_time;
                }
            }
        }

        unsafe {
            self.ctx
                .device
                .reset_query_pool(self.query_pool, 0, self.query_count);
        }
        self.query_counter = 0;
    }
    pub fn profile<F: FnOnce() -> ()>(
        &mut self,
        cmd_encoder: &crate::CommandEncoder,
        kernel_name: &str,
        // stage: vk::PipelineStageFlags,
        f: F,
    ) {
        if self.query_counter + 2 > self.query_count {
            panic!("maximum of {} queries exceeded", self.query_count);
        }
        let kernel_name = String::from(kernel_name);
        unsafe {
            self.ctx.device.cmd_write_timestamp(
                cmd_encoder.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                self.query_counter,
            );

            f();
            self.ctx.device.cmd_write_timestamp(
                cmd_encoder.command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                self.query_counter + 1,
            );

            let fence = cmd_encoder.get_fence();
            self.fences.push_back(fence);
            if !self.hash_map.contains_key(&kernel_name) {
                self.hash_map.insert(
                    kernel_name.clone(),
                    KernelRecord {
                        stats: KernelStatsIntermediate {
                            n_launches: 0,
                            total_time: 0.0,
                            max_time: 0.0,
                        },
                        queries: vec![],
                    },
                );
            }
            {
                let record = self.hash_map.get_mut(&kernel_name).unwrap();
                record
                    .queries
                    .push((self.query_counter, self.query_counter + 1));
            }

            self.query_counter += 2;
        }
    }
}
