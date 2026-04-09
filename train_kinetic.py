import kinetic
from model import get_model
from data_loader import get_dataset

def train():
    TRAIN_PATH = "gs://your-bucket/lulc_samples/train"
    
    job = kinetic.create_job(
        model_fn=get_model,
        dataset_fn=lambda: get_dataset(TRAIN_PATH),
        tpu_type="v5lite-pod-8", # TPU TYPE
        epochs=50
    )
    
    job.run()

if __name__ == "__main__":
    train()