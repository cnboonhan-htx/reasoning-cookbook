# reasoning-cookbook


## Design
Current thoughts on the design and pipeline.

1. Download, Deploy [`nvidia/Cosmos-Reason2-8B`](https://huggingface.co/nvidia/Cosmos-Reason2-8B) for inference on Phase1A
2. Deploy [`SimWorld`](https://simworld.readthedocs.io/en/latest/) environment on PC for agent rollout and evuation. Test out how to move agent and interact with objects in the environment.
3. Explore [`TRL`](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/README.md) and and [`cosmos-rl`](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/cosmos_rl/README.md) techniques to fine-tune model. Get a sense of datasets required and collect some new datasets / transform public datasets.
4. Come up with some SOP for use cases. Get a sense of the appropriate "datasets" required, and the post-processing required to align with datasets in 3.
5. Establish automation pipeline for training 3. through P1A and other clusters.
6. Establish automation pipeline for evaluation output from 5.
7. Deploy the necessary infrastructure to enable experiment tracking. (WanDB)
8. As needed, modify the `Simworld` to add assets objects to carry out SOP.
