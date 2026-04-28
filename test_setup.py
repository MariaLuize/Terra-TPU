import os
import tensorflow as tf
from dotenv import load_dotenv
from data_loader import get_dataset, BANDS

load_dotenv()
bucket       = os.getenv("BUCKET_NAME")
train_folder = os.getenv("TRAIN_DATA_PATH")
TRAIN_PATH   = f"gs://{bucket}/{train_folder}"

print(f"Verificando acesso ao bucket: {TRAIN_PATH}")

def test_pipeline():
    # try:
        # 2. Tenta instanciar o dataset (batch pequeno para teste)
        # Se você estiver local, certifique-se de estar logada no gcloud:
        # gcloud auth application-default login
    dataset = get_dataset(TRAIN_PATH, batch_size=2, is_training=False)
    
    # 3. Pega apenas um exemplo
    for img, mask in dataset.take(1):
        print("\n✅ SUCESSO: Pipeline de dados funcionando!")
        print(f"---")
        print(f"📊 Shape da Imagem (Input): {img.shape}")
        print(f"📊 Shape da Máscara (Label): {mask.shape}")
        print(f"---")
        print(f"✨ Valor Máximo da Imagem: {tf.reduce_max(img).numpy():.4f} (Esperado: ~1.0 ou menos)")
        print(f"✨ Valor Mínimo da Imagem: {tf.reduce_min(img).numpy():.4f} (Esperado: >= 0.0)")
        print(f"✨ Bandas detectadas: {img.shape[-1]} (Esperado: {len(BANDS)})")
        
        # Verificação de sanidade dos labels
        unique_labels = tf.unique(tf.reshape(mask, [-1]))[0].numpy()
        print(f"🏷️  Valores únicos no Label: {unique_labels}")
            
    # except Exception as e:
    #     print(f"\n❌ ERRO no setup: {e}")
    #     print("\n💡 Dica: Verifique se você rodou 'gcloud auth application-default login'")
    #     print("💡 Dica: Verifique se o caminho no .env não tem espaços ou barras extras.")

if __name__ == "__main__":
    test_pipeline()