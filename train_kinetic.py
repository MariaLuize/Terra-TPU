import os
import kinetic
from dotenv import load_dotenv
from data_loader import BANDS

load_dotenv()

PROJECT_ID  = os.getenv("KINETIC_PROJECT")
# ZONE        = os.getenv("KINETIC_ZONE")
# ZONE        = "us-west1-c"
# ACCELERATOR = os.getenv("TPU_TYPE", "v5litepod-4")
ACCELERATOR = os.getenv("TPU_TYPE")

BUCKET_NAME     = os.getenv("BUCKET_NAME")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
VAL_DATA_PATH   = os.getenv("VAL_DATA_PATH")


@kinetic.run(accelerator=ACCELERATOR, project=PROJECT_ID)
def train_on_tpu(bucket=BUCKET_NAME, train_folder=TRAIN_DATA_PATH, val_folder=VAL_DATA_PATH):
    import tensorflow as tf
    import os
    from model import UNetModel
    from data_loader import get_dataset, BANDS
    

    TRAIN_PATH = f"gs://{bucket}/{train_folder}"
    VAL_PATH   = f"gs://{bucket}/{val_folder}"
    
    # Kinetic define automaticamente KINETIC_OUTPUT_DIR na TPU
    output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/terra_tpu")
    checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.keras")
    log_dir         = os.path.join(output_dir, "metrics")

    model_instance = UNetModel(
        input_shape=[256, 256, len(BANDS)],
        dropout_rate=0.3, 
        loss='binary_crossentropy', 
        metrics_list=['RootMeanSquaredError', 'BinaryIoU']
    )
    model = model_instance.get_model()

    # 2. Carrega os datasets
    # Usamos batch 64/128 para alta performance na TPU
    BATCH_SIZE = 64
    TRAIN_SIZE = 65218
    VAL_SIZE   = 16212
    
    ds_train = get_dataset(TRAIN_PATH, batch_size=BATCH_SIZE, is_training=True)
    ds_val   = get_dataset(VAL_PATH, batch_size=BATCH_SIZE, is_training=False)

    # 3. Callbacks (Sintaxe TF 2.x/Keras 3)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_binary_io_u',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    # 4. O Treino propriamente dito
    print(f"Lendo dados de: {TRAIN_PATH}")
    print("Iniciando fit...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=50,
        steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
        validation_steps=VAL_SIZE // BATCH_SIZE,  
        callbacks=callbacks
    )

    return f"Treino finalizado! Modelo salvo em: {checkpoint_path}"

if __name__ == "__main__":
    # Ao chamar a função, o Kinetic intercepta e manda para o Google Cloud
    result = train_on_tpu()
    print(result)