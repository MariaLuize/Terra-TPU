# import os
# import kinetic
# from dotenv import load_dotenv
# from model import UNetModel
# from data_loader import get_dataset, BANDS
# import tensorflow as tf

# load_dotenv()

# def train():
#     bucket       = os.getenv("BUCKET_NAME")
#     train_folder = os.getenv("TRAIN_DATA_PATH")
#     val_folder   = os.getenv("VAL_DATA_PATH")
#     tpu_type     = os.getenv("TPU_TYPE", "v5lite-pod-8")
    
#     GLOBAL_BATCH_SIZE = 64 
#     TRAIN_PATH = f"gs://{bucket}/{train_folder}"
#     VAL_PATH = f"gs://{bucket}/{val_folder}"
#     # CHECKPOINT_PATH = f"gs://{bucket}/models/terra_tpu_v1.keras"
#     output_dir = os.environ.get("KINETIC_OUTPUT_DIR", "/tmp/terra_tpu")

#     checkpoint_path = os.path.join(output_dir, "checkpoints", "cp-{epoch:04d}.keras")
#     log_dir         = os.path.join(output_dir, "metrics")
#     final_model_path = os.path.join(output_dir, "final", "terra_tpu_final.keras")
    
#     # input como (256, 256, 6). Como as TPUs trabalham melhor com tamanhos estáticos, certifique-se de que no train.py você está passando exatamente o número de bandas (6) e o tamanho (256).
#     model_instance = UNetModel(
#         input_shape=[256, 256, len(BANDS)],
#         dropout_rate=0.3, 
#         loss='binary_crossentropy', 
#         metrics_list=['RootMeanSquaredError', 'BinaryIoU']
#     )
    
#     # O Kinetic precisa de uma FUNÇÃO que retorne o modelo, 
#     # ou o próprio objeto do modelo compilado.
#     model = model_instance.get_model()
    
#     callbacks = [
#         # Salva a cada 5 épocas conforme solicitado
#         tf.keras.callbacks.ModelCheckpoint(
#             filepath=checkpoint_path,
#             verbose=1,
#             save_weights_only=False,
#             save_freq='epoch',
#         ),
#         # Logs para o TensorBoard (na pasta 'metrics' conforme o layout recomendado)
#         tf.keras.callbacks.TensorBoard(
#             log_dir=log_dir,
#             update_freq='epoch'
#         )
#     ]
#     job = kinetic.create_job(
#         model_fn=lambda: model, 
#         dataset_fn=lambda: get_dataset(TRAIN_PATH, batch_size=GLOBAL_BATCH_SIZE, is_training=True), #Para uma TPU v5lite-pod-8 (que tem 8 núcleos), o batch size ideal costuma ser um múltiplo de 8 ou 128. Isso preenche a memória de cada núcleo da TPU e permite que o compilador XLA otimize as operações de matriz. Com um batch de 10, a TPU ficaria ociosa na maior parte do tempo.
#         validation_data=lambda: get_dataset(VAL_PATH, batch_size=GLOBAL_BATCH_SIZE, is_training=False), #As TPUs funcionam melhor com "Static Shapes" (formatos fixos). Se o treino é 64 e a validação é 1, a TPU precisa recompilar ou reajustar o grafo toda vez que muda de fase, o que gera um atraso enorme. Usar o mesmo batch simplifica a vida do hardware.
#         tpu_type=tpu_type,
#         epochs=50,
#         callback=callbacks
#     )

#     print(f"Iniciando treino na TPU tipo: {tpu_type}")
#     print(f"Dados de treino: {TRAIN_PATH} e de validacao {VAL_PATH}")
#     job.run()
#     model.save(final_model_path)
#     return f"Treinamento concluído. Modelo final: {final_model_path}"

# if __name__ == "__main__":
#     train()


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
    # IMPORTANTE: Imports pesados ficam DENTRO da função para a TPU usar o hardware dela
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

    # 1. Instancia o modelo e compila (DENTRO da função)
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