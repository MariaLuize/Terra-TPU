import kinetic
# from model import get_model
from model import UNetModel
from data_loader import get_dataset

def train():
    TRAIN_PATH = "gs://your-bucket/lulc_samples/train"
    model_instance = UNetModel(input_shape=[None, None, len(optical_bands + optical_indices)], dropout_rate=dropout, loss=loss, metrics_list=metrics)
    model          = model_instance.get_model()

    
    job = kinetic.create_job(
        model_fn=model,
        dataset_fn=lambda: get_dataset(TRAIN_PATH),
        tpu_type="v5lite-pod-8", # TPU TYPE
        epochs=50
    )
    
    job.run()

if __name__ == "__main__":
    train()