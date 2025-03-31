'''from kafka import KafkaConsumer
from google.cloud import bigquery
import json
import os

# Configurar BigQuery
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\GAMIC\PORTFOLIO\CLOTHING RETAIL - PRS POWERED BY LLM\credentials.json"
client = bigquery.Client()
dataset_id = "fashion_retail_dataset"
table_id = "interactions"

Q_PROJECT_ID = "clothing-retail-prs-llm"

# Configurar el consumidor de Kafka
consumer = KafkaConsumer(
    'interactions_topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# --- PRUEBA MANUAL ---
# (Ejecútalo una vez, luego comenta o elimina esta sección)

# Datos de prueba estáticos (ajusta los valores según tu esquema)
test_data = {
    "interaction_id": 999999,
    "customer_id": 100,
    "product_id": 50,
    "event_type": "test",
    "event_timestamp": "2025-01-20T15:30:45.123456Z",  # Usa una fecha en formato ISO
    "session_id": "test_session_123"
}

# Referencia completa de la tabla (proyecto.dataset.tabla)
table_ref = f"{Q_PROJECT_ID}.{dataset_id}.{table_id}"

# Insertar el dato de prueba
errors = client.insert_rows_json(
    table_ref,
    [test_data]  # Debe ser una lista de diccionarios
)

if not errors:
    print("¡Prueba exitosa! Dato insertado en BigQuery.")
else:
    print(f"Error en la prueba: {errors}")



def transform_data(data):
    return {
        "interaction_id": data["interaction_id"],  
        "customer_id": data["customer_id"],        
        "product_id": data["product_id"],          
        "event_type": data["event_type"],
        "event_timestamp": data["event_timestamp"],
        "session_id": data["session_id"]
    }

for message in consumer:
    try:
        data = message.value
        transformed_data = transform_data(data)
        # Insertar en BigQuery...
    except Exception as e:
        print(f"[ERROR] Fallo al procesar mensaje: {e}")

if not errors:
    print(f"Dato insertado: {transformed_data}")
else:
    print(f"Error detallado: {errors}") '''

from kafka import KafkaConsumer
from google.cloud import bigquery
import json
import os
import logging

# Configurar logging para depuración
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configurar BigQuery
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\GAMIC\PORTFOLIO\CLOTHING RETAIL - PRS POWERED BY LLM\credentials.json"
client = bigquery.Client()
dataset_id = "fashion_retail_dataset"
table_id = "interactions"
Q_PROJECT_ID = "clothing-retail-prs-llm"

# Referencia completa de la tabla en BigQuery
TABLE_REF = f"{Q_PROJECT_ID}.{dataset_id}.{table_id}"

# Configurar el consumidor de Kafka
def create_kafka_consumer():
    try:
        consumer = KafkaConsumer(
            'interactions_topic',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',  # Leer desde el inicio del topic
            enable_auto_commit=False
        )
        logging.info("Conexión a Kafka establecida correctamente")
        return consumer
    except Exception as e:
        logging.error(f"Error al conectar a Kafka: {e}")
        raise

# Transformar datos para BigQuery
def transform_data(data):
    try:
        transformed = {
            "interaction_id": int(data["interaction_id"]),
            "customer_id": int(data["customer_id"]),
            "product_id": int(data["product_id"]),
            "event_type": str(data["event_type"]),
            "event_timestamp": data["event_timestamp"],  # BigQuery maneja ISO 8601
            "session_id": str(data["session_id"])
        }
        logging.debug("Datos transformados: %s", transformed)
        return transformed
    except KeyError as e:
        logging.error(f"Campo faltante en los datos: {e}")
        raise
    except Exception as e:
        logging.error(f"Error transformando datos: {e}")
        raise

# Insertar datos en BigQuery
def insert_to_bigquery(data):
    try:
        errors = client.insert_rows_json(TABLE_REF, [data])
        if errors:
            logging.error("Error insertando en BigQuery: %s", errors)
        else:
            logging.info("Dato insertado correctamente")
    except Exception as e:
        logging.error(f"Error fatal al insertar en BigQuery: {e}")
        raise

# Procesar mensajes
def main():
    consumer = create_kafka_consumer()
    try:
        for message in consumer:
            logging.info("--------------------------------------------------")
            logging.info("Mensaje recibido de Kafka:")
            logging.info("Topic: %s, Partición: %s, Offset: %s", 
                        message.topic, message.partition, message.offset)
            
            try:
                # Paso 1: Obtener datos del mensaje
                raw_data = message.value
                logging.debug("Dato en bruto: %s", raw_data)
                
                # Paso 2: Transformar datos
                transformed_data = transform_data(raw_data)
                
                # Paso 3: Insertar en BigQuery
                insert_to_bigquery(transformed_data)
                
            except Exception as e:
                logging.error("Error procesando mensaje: %s", e, exc_info=True)
                
    except KeyboardInterrupt:
        logging.info("Deteniendo el consumer...")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()