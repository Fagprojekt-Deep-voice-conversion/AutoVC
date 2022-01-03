import wandb
import time

# We will use ray to launch our runs in parallel
# for demonstration purposes. You can orchestrate
# your parallel runs however you want.
import ray

ray.init()

artifact_type = "dataset"
artifact_name = "parallel-artifact"
table_name = "distributed_table"
parts_path = "parts"
num_parallel = 5

# Each batch of parallel writers should have its own
# unique group name.
group_name = "writer-group-{}".format(round(time.time()))

@ray.remote
def train(i):
  """
  Our writer job. Each writer will add one image to the artifact.
  """
  with wandb.init(group=group_name, entity = "deep_voice_inc") as run:
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    
    # Add data to a wandb table. In this case we use example data
    table = wandb.Table(columns=["a", "b", "c"], data=[[i, i*2, 2**i]])
    
    # Add the table to folder in the artifact
    artifact.add(table, "{}/table_{}".format(parts_path, i))
    
    # Upserting the artifact creates or appends data to the artifact
    run.upsert_artifact(artifact)
  
# Launch your runs in parallel
result_ids = [train.remote(i) for i in range(num_parallel)]

# Join on all the writers to make sure their files have
# been added before finishing the artifact. 
ray.get(result_ids)

# Once all the writers are done writing, finish the artifact
# to mark it ready.
with wandb.init(group=group_name, entity = "deep_voice_inc") as run:
  artifact = wandb.Artifact(artifact_name, type=artifact_type)
  
  # Create a "PartitionTable" pointing to the folder of tables
  # and add it to the artifact.
  artifact.add(wandb.data_types.PartitionedTable(parts_path), table_name)
  
  # Finish artifact finalizes the artifact, disallowing future "upserts"
  # to this version.
  run.finish_artifact(artifact)