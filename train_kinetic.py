import os
import kinetic
from dotenv import load_dotenv
from model import UNetModel
from data_loader import get_dataset, BANDS

load_dotenv()

def train():
    bucket       = os.getenv("BUCKET_NAME")
    train_folder = os.getenv("TRAIN_DATA_PATH")
    val_folder   = os.getenv("VAL_DATA_PATH")
    tpu_type     = os.getenv("TPU_TYPE", "v5lite-pod-8")
    
    GLOBAL_BATCH_SIZE = 64 
    TRAIN_PATH = f"gs://{bucket}/{train_folder}"
    VAL_PATH = f"gs://{bucket}/{val_folder}"

    model_instance = UNetModel(
        input_shape=[None, None, len(BANDS)],
        dropout_rate=0.3, 
        loss='binary_crossentropy', 
        metrics_list=['RootMeanSquaredError', 'BinaryIoU']
    )
    
    # O Kinetic precisa de uma FUNÇÃO que retorne o modelo, 
    # ou o próprio objeto do modelo compilado.
    model = model_instance.get_model()

    job = kinetic.create_job(
        model_fn=lambda: model, 
        dataset_fn=lambda: get_dataset(TRAIN_PATH, batch_size=GLOBAL_BATCH_SIZE, is_training=True), #Para uma TPU v5lite-pod-8 (que tem 8 núcleos), o batch size ideal costuma ser um múltiplo de 8 ou 128. Isso preenche a memória de cada núcleo da TPU e permite que o compilador XLA otimize as operações de matriz. Com um batch de 10, a TPU ficaria ociosa na maior parte do tempo.
        validation_data=lambda: get_dataset(VAL_PATH, batch_size=GLOBAL_BATCH_SIZE, is_training=False), #As TPUs funcionam melhor com "Static Shapes" (formatos fixos). Se o treino é 64 e a validação é 1, a TPU precisa recompilar ou reajustar o grafo toda vez que muda de fase, o que gera um atraso enorme. Usar o mesmo batch simplifica a vida do hardware.
        tpu_type=tpu_type,
        epochs=50
    )

    print(f"Iniciando treino na TPU tipo: {tpu_type}")
    print(f"Dados de treino: {TRAIN_PATH} e de validacao {VAL_PATH}")
    job.run()

if __name__ == "__main__":
    train()
    