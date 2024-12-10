from metaflow import (
    FlowSpec,
    step,
    Parameter,
    kubernetes,
    gpu_profile,
    pypi,
    metaflow_ray,
    current,
    Task,
    Run,
    schedule,
    retry
)
from metaflow.profilers import gpu_profile

N_PARALLEL = 2
N_GPU_PER_WORKER = 1

@schedule(cron="*/5 * * * *")
class RayTorchTrain(FlowSpec):

    epochs = Parameter("epochs", default=3)
    global_batch_size = Parameter("batch_size", default=32)
    learning_rate = Parameter("learning_rate", default=1e-3)
    upstream_task_pathspec = Parameter(
        "from-task",
        default=None,
        help="A task pathspec like flow_name/run_id/step_name/task_id containing a .results artifact with a Ray checkpoint.",
    )
    upstream_run_pathspec = Parameter(
        "from-run",
        default=None,
        help="A run pathspec like flow_name/run_id containing a .results artifact with a Ray checkpoint.",
    )

    @step
    def start(self):
        self.next(self.train, num_parallel=N_PARALLEL)

    @retry(times=3)
    @metaflow_ray(all_nodes_started_timeout = 60 * 5)
    @pypi(
        packages={
            "ray[train]": "2.39.0",
            "torch": "2.5.1",
            "torchvision": "0.20.1",
            "numpy": "2.1.3",
        }
    )
    @gpu_profile(interval=1)
    @kubernetes(gpu=N_GPU_PER_WORKER, compute_pool="obp-gpu")
    @step
    def train(self):
        from my_ray_module import train_fashion_mnist

        hyperparameters = dict(
            epochs=self.epochs,
            global_batch_size=self.global_batch_size,
            learning_rate=self.learning_rate,
        )
        args = dict(
            num_workers=N_PARALLEL*N_GPU_PER_WORKER,
            use_gpu=True,
            checkpoint_storage_path=current.ray_storage_path,
            **hyperparameters
        )
        if self.upstream_task_pathspec is not None and self.upstream_task_pathspec != "null":
            t = Task(self.upstream_task_pathspec)
            args['checkpoint'] = t.data.result.checkpoint
        elif self.upstream_run_pathspec is not None and self.upstream_run_pathspec != "null":
            r = Run(self.upstream_run_pathspec)
            args['checkpoint'] = r.data.result.checkpoint
        else: 
            print('Training from newly initialized')

        self.result = train_fashion_mnist(**args)
        self.next(self.join)

    @pypi(packages={"ray[train]": "2.39.0"})
    @kubernetes
    @step
    def join(self, inputs):
        for i in inputs:
            try:
                self.result = i.result
            except:
                pass
        self.next(self.end)

    @pypi(packages={"ray[train]": "2.39.0"})
    @kubernetes
    @step
    def end(self):
        print(self.result)


if __name__ == "__main__":
    RayTorchTrain()

