from metaflow import (
    FlowSpec,
    step,
    Parameter,
    kubernetes,
    gpu_profile,
    pypi,
    metaflow_ray,
    card,
    current,
    trigger_on_finish,
    Task,
    Run,
)
from metaflow.cards import Table, Artifact, Markdown, Image

N_GPU = 1

@trigger_on_finish(flow="RayTorchTrain")
class RayTorchEval(FlowSpec):

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
    upstream_namespace = Parameter(
        "from-namespace",
        default=None, # TODO: Select this default carefully, based on where you deployed the upstream flow.
        help="Specify this if the upstream task or run with the Ray checkpoint is in a different Metaflow namespace.",
    )
    batch_size = Parameter("batch_size", default=512)
    n_error_samples = 50

    def _get_checkpoint(self):
        try:
            checkpoint = current.trigger.run.data.result.checkpoint
        except AttributeError as e:
            if self.upstream_task_pathspec is not None and self.upstream_task_pathspec != "null":
                t = Task(self.upstream_task_pathspec)
                checkpoint = t.data.result.checkpoint
            elif self.upstream_run_pathspec is not None and self.upstream_run_pathspec != "null":
                r = Run(self.upstream_run_pathspec)
                checkpoint = r.data.result.checkpoint
            else:
                raise ValueError(
                    "If this run is not being triggered by RayTorchTrain, you must specify an upstream run or task id."
                )
        return checkpoint

    @card(type='blank', id="error_analysis")
    @gpu_profile(interval=1)
    @kubernetes(gpu=N_GPU, compute_pool="obp-gpu")
    @pypi(
        packages={
            "ray[train]": "2.39.0",
            "torch": "2.5.1",
            "torchvision": "0.20.1",
            "numpy": "2.1.3",
            "pandas": "2.2.3",
            "matplotlib": "3.9.2"
        }
    )
    @step
    def start(self):
        from my_ray_module import get_dataloaders, TorchPredictor, get_labels_map
        import ray
        import torch
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        from ray.data.context import DataContext
        ctx = DataContext.get_current()
        ctx.enable_tensor_extension_casting = False

        self.upstream_checkpoint = self._get_checkpoint()
        ds = get_dataloaders(batch_size=self.batch_size, val_only=True, as_ray_ds=True)

        result = ds.map_batches(
            TorchPredictor(checkpoint=self.upstream_checkpoint, cpu_only=not torch.cuda.is_available()),
            concurrency=N_GPU,
            batch_size=self.batch_size,
            num_gpus=N_GPU
        ).take_all()
        self.predictions = pd.concat([ds.to_pandas(), pd.DataFrame(result)], axis=1)
        self.misclassifications = self.predictions.where(
            self.predictions.labels != self.predictions.predicted_values
        ).dropna()

        labels_map = get_labels_map()
        sample = self.misclassifications.sample(self.n_error_samples)
        current.card['error_analysis'].append(
            Markdown(f'### Misclassifications {self.misclassifications.shape[0]} out of {self.predictions.shape[0]}')
        )

        table_data = []
        for idx, row in sample.iterrows(): 

            features_fig, features_ax = plt.subplots()
            features_ax.imshow(row.features.reshape(28, 28), cmap='gray')
            features_ax.axis('off')

            image_artifact = Image.from_matplotlib(features_fig)
            plt.close(features_fig)

            logits_fig, logits_ax = plt.subplots(figsize=(6, 4))
            categories = list(labels_map.values())
            logits_ax.barh(categories, row.logits)
            logits_ax.set_title("Logits")
            logits_ax.set_xlabel("Value")
            logits_ax.set_ylabel("Category")
            logits_ax.spines[['right', 'top']].set_visible(False)
            plt.tight_layout()

            for bar, value in zip(logits_ax.patches, row.logits):
                logits_ax.text(value, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va='center')

            logits_artifact = Image.from_matplotlib(logits_fig)
            plt.close(logits_fig)

            table_data.append([
                image_artifact,
                labels_map[int(row.labels)], 
                labels_map[int(row.predicted_values)], 
                logits_artifact
            ])

        current.card['error_analysis'].append(
            Table(
                headers=["Image", "True label", "Predicted label", "Logits"],
                data=table_data
            )
        )

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayTorchEval()

