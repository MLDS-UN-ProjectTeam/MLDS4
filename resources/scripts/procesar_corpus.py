import re
import time
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from tqdm import tqdm  

import nltk
import kagglehub
import pandas as pd

from emoji import demojize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from textblob import download_corpora


try:
    import contractions
    from langdetect import detect, DetectorFactory
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "contractions", "langdetect"])
    import contractions
    from langdetect import detect, DetectorFactory

# Descargar recursos necesarios
nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
download_corpora.download_all()

# Instanciar herramientas
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

OUTPUT_CSV = "./dataset_limpio.csv"
TEXT_COLUMN = "statement"
TIMEOUT_SEGUNDOS = 5

# Flag opcional para activar corrección
CORREGIR_ORTOGRAFIA = True

# Función para detectar idioma
def detectar_idioma(texto):
    try:
        return detect(str(texto))
    except:
        return "unknown"

def limpiar_texto(texto):
    # 0. Asegurar que sea string
    texto = str(texto)

    # 1. Expandir contracciones ("I'm" -> "I am")
    texto = contractions.fix(texto)

    # 2. Demojizar 
    texto = demojize(texto)

    # 3. Reemplazar URLs, menciones, hashtags
    texto = re.sub(r"http\S+|www\S+|https\S+", " ", texto)
    texto = re.sub(r"@\w+", " ", texto)
    texto = re.sub(r"#(\w+)", r"\1", texto)

    # 4. Reemplazar puntuación por espacios
    texto = re.sub(r"[^a-zA-Z]", " ", texto)

    # 5. Minúsculas
    texto = texto.lower()

    # 6. Tokenización
    tokens = word_tokenize(texto)

    # 7. Filtro de tokens
    tokens_filtrados = [
        token for token in tokens
        if token not in stop_words and len(token) > 1 and token.isalpha()
    ]

    # 8. Corrección ortográfica opcional
    if CORREGIR_ORTOGRAFIA:
        tokens_corregidos = []
        for token in tokens_filtrados:
            palabra_corregida = str(TextBlob(token).correct())
            tokens_corregidos.append(palabra_corregida)
    else:
        tokens_corregidos = tokens_filtrados

    # 9. Lematización
    tokens_lemmatizados = [lemmatizer.lemmatize(token) for token in tokens_corregidos]

    # 10. Retornar lista final de tokens
    return tokens_lemmatizados

def timeout_en_limpieza(texto):
    with ThreadPoolExecutor(max_workers=1) as executor:
        futuro = executor.submit(limpiar_texto, texto)
        try:
            return futuro.result(timeout=TIMEOUT_SEGUNDOS)
        except TimeoutError:
            logging.warning("⏱ Timeout en limpieza: " + str(texto[:40] + "\n"))
            return []
        except Exception as e:
            logging.error(f"❌ Error al limpiar texto: {texto[:40]}... -> {str(e)} \n")
            return []

# Descarga la última versión del dataset
path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")

print("Path to dataset files:", path)
# Carga la data del dataset
path = '/Users/diegof/.cache/kagglehub/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/versions/1/Combined Data.csv'
df = pd.read_csv(path)

def procesar_en_paralelo(funcion, lista_textos, num_procesos=None):
    if num_procesos is None:
        from os import cpu_count
        num_procesos = cpu_count()
    with mp.Pool(processes=num_procesos) as pool:
        resultados = list(tqdm(pool.imap(funcion, lista_textos), total=len(lista_textos)))
    return resultados

if __name__ == "__main__":
    start = time.time()
    try:
        logging.info("🚀 Inicio del procesamiento")
        print("📦 Cargando datos...")
        print(f"✅ {len(df)} documentos cargados.")
        logging.info(f"{len(df)} documentos cargados.")

        print("🌍 Detectando idioma...")
        df["lang"] = df[TEXT_COLUMN].apply(detectar_idioma)
        df_ingles = df[df["lang"] == "en"].copy()
        print(f"✅ Documentos en inglés: {len(df_ingles)}")

        print("🧠 Procesando en paralelo con timeout por fila...")
        textos = df_ingles[TEXT_COLUMN].tolist()
        textos_limpios = procesar_en_paralelo(timeout_en_limpieza, textos, num_procesos=11)

        print("💾 Guardando resultados...")
        df_ingles["clean_tokens"] = textos_limpios
        df_ingles.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Archivo guardado como: {OUTPUT_CSV}")
        print(f"✅ Archivo guardado como: {OUTPUT_CSV}")
    except Exception as e:
        print("❌ Error durante el procesamiento:")
        print(e)
        logging.error(f"Error general del script: {str(e)}")
    finally:
        end = time.time()
        elapsed = (end - start) / 60
        print(f"⏱ Tiempo total: {elapsed:.2f} minutos")
        logging.info(f"Tiempo total: {elapsed:.2f} minutos")