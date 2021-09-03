use std::{
    cell::RefCell,
    collections::HashMap,
    iter::FromIterator,
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, Weak},
};

use ash::vk;

use crate::{CommandEncoder, Context, Fence};

struct TaskGraphInner {
    nodes: Vec<Arc<TaskGraphNodeInner>>,
    exec_order: Option<GraphExecutionOrder>,
}
pub struct TaskGraph {
    inner: Arc<TaskGraphInner>,
}
impl TaskGraph {
    pub fn builder() -> TaskGraphBuilder {
        TaskGraphBuilder {
            graph: TaskGraph {
                inner: Arc::new(TaskGraphInner {
                    nodes: vec![],
                    exec_order: None,
                }),
            },
        }
    }
}
pub type TaskFunc = dyn FnOnce(&mut CommandEncoder) -> () + 'static;
struct TaskGraphNodeInner {
    depends: Vec<Weak<TaskGraphNodeInner>>,
    graph: Weak<TaskGraphInner>,
    task_func: Box<TaskFunc>,
}

pub struct TaskGraphNode<'a> {
    inner: Weak<TaskGraphNodeInner>,
    phantom: PhantomData<&'a mut u32>,
}
pub struct TaskGraphBuilder {
    graph: TaskGraph,
}
impl TaskGraphBuilder {
    pub fn build(self) -> TaskGraph {
        let mut graph = self.graph;
        let order = Some(GraphExecutionOrder::new(&graph));
        {
            let inner = Arc::get_mut(&mut graph.inner).unwrap();
            inner.exec_order = order;
        }
        graph
    }
    pub fn add<'a, F: FnOnce(&mut CommandEncoder) -> () + 'static>(
        &'a mut self,
        depends: &[&TaskGraphNode<'a>],
        f: F,
    ) -> TaskGraphNode<'a> {
        for depend in depends {
            let inner = Weak::upgrade(&depend.inner).unwrap();
            if !inner.graph.ptr_eq(&Arc::downgrade(&self.graph.inner)) {
                panic!("mixed tasks of different graphs");
            }
        }
        let inner = TaskGraphNodeInner {
            depends: depends.iter().map(|x| x.inner.clone()).collect(),
            graph: Arc::downgrade(&self.graph.inner),
            task_func: Box::new(f),
        };
        let inner = Arc::new(inner);
        {
            let graph = Arc::get_mut(&mut self.graph.inner).unwrap();
            graph.nodes.push(inner.clone());
        }
        TaskGraphNode::<'a> {
            inner: Arc::downgrade(&inner),
            phantom: PhantomData {},
        }
    }
}

struct GraphExecutionOrder {
    order: Vec<Vec<Arc<TaskGraphNodeInner>>>,
}
impl GraphExecutionOrder {
    fn new(graph: &TaskGraph) -> Self {
        enum Mark {
            Permenant,
            Temp,
        }
        let mut markers: HashMap<u64, Mark> = HashMap::new();
        let mut sorted = vec![];
        fn visit(
            node: &Arc<TaskGraphNodeInner>,
            markers: &mut HashMap<u64, Mark>,
            sorted: &mut Vec<Arc<TaskGraphNodeInner>>,
        ) {
            let id = node.as_ref() as *const TaskGraphNodeInner as u64;
            if let Some(mark) = markers.get(&id) {
                match mark {
                    Mark::Permenant => {
                        return;
                    }
                    Mark::Temp => {
                        panic!("graph is not DAG");
                    }
                }
            }
            markers.insert(id, Mark::Temp);
            for dep in &node.depends {
                let dep = Weak::upgrade(dep).unwrap();
                visit(&dep, markers, sorted);
            }
            *markers.get_mut(&id).unwrap() = Mark::Permenant;
            sorted.push(node.clone());
        }
        for node in &graph.inner.nodes {
            let id = node.as_ref() as *const TaskGraphNodeInner as u64;
            if let None = markers.get(&id) {
                visit(node, &mut markers, &mut sorted);
            }
        }
        let mut node_to_idx = HashMap::new();
        for (i, node) in sorted.iter().enumerate() {
            let id = node.as_ref() as *const TaskGraphNodeInner as u64;
            node_to_idx.insert(id, i);
        }
        let mut ranks = vec![-1i32; sorted.len()];
        for (i, node) in sorted.iter().enumerate() {
            if node.depends.is_empty() {
                ranks[i] = 0;
            } else {
                ranks[i] = node
                    .depends
                    .iter()
                    .map(|dep| -> i32 {
                        let dep = Weak::upgrade(dep).unwrap();
                        let id = dep.as_ref() as *const TaskGraphNodeInner as u64;
                        let idx = node_to_idx.get(&id).unwrap();
                        assert!(ranks[*idx] >= 0);
                        ranks[*idx]
                    })
                    .max()
                    .unwrap();
            }
        }
        let mut order: Vec<Vec<Arc<TaskGraphNodeInner>>> = vec![];
        for (i, node) in sorted.iter().enumerate() {
            let rank = ranks[i];
            while rank >= order.len() as i32 {
                order.push(vec![]);
            }
            order[rank as usize].push(node.clone());
        }
        Self { order }
    }
}
pub struct Executor {
    ctx: Context,
}

impl Executor {
    pub fn execute(&self, graph: &TaskGraph) {}
}
