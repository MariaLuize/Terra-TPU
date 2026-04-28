# test_setup.py
from data_loader import get_dataset
import os
from dotenv import load_dotenv

load_dotenv()
path = f"gs://{os.getenv('BUCKET_NAME')}/{os.getenv('TRAIN_DATA_PATH')}"

try:
    ds = get_dataset(path, batch_size=2)
    for img, mask in ds.take(1):
        print("✅ Sucesso! Dados lidos do GCS.")
        print(f"Shape Imagem: {img.shape}") # Esperado: (2, 256, 256, 6)
except Exception as e:
    print(f"❌ Erro ao ler do GCS: {e}")