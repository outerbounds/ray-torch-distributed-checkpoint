## Setup

```bash
pip install outerbounds metaflow-ray==0.1.3
```

## Manual mode
To run training from scratch, without any pre-trained weights:
```bash
python train_flow.py --environment=fast-bakery run
```

Checkpoints are persisted in the Metaflow datastore, in a location unique to each task runtime. 
Notice the `current.ray_storage_path` variable exposed in Metaflow step annotated with `@metaflow_ray`.
This variable can be passed to Ray's `RunConfig(storage_path=...)` value.

To run training from a previous run's checkpoint - which you can find in the Outerbounds UI or CLI output:
```bash
python train_flow.py --environment=fast-bakery run --from-run RayTorchTrain/<YOUR_RUN_ID>
```

To run an evaluation pipeline from the training run's checkpoint:
```bash
python eval_flow.py --environment=fast-bakery evaluate --from-run RayTorchTrain/<YOUR_RUN_ID>
```

## Automate on the product orchestrator

Deploy the training workflow:
```bash
python train_flow.py --environment=fast-bakery argo-workflows create
```

Deploy the evaluation workflow:
```bash
python eval_flow.py --environment=fast-bakery argo-workflows create
```

To trigger the workflow, you can go the Deployments page in the Outerbounds UI and find this workflow.
Alternatively, you can run this command:
```bash
python train_flow.py --environment=fast-bakery argo-workflows trigger
```

The evaluation workflow will be triggered automatically after the training workflow completes.