from kafka import KafkaProducer
from faker import Faker
import json
import time
import random
from datetime import datetime  


fake = Faker()

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


event_types = ['view', 'click', 'add_to_cart', 'purchase']

def generate_interaction():
    return {
        "interaction_id": fake.unique.random_number(digits=8),
        "customer_id": random.randint(1, 5005),  
        "product_id": random.randint(1, 500),     
        "event_type": random.choice(event_types),
        "event_timestamp": datetime.now().isoformat(),
        "session_id": fake.uuid4()
    }

max_interactions = 20  
for _ in range(max_interactions):
    data = generate_interaction()
    producer.send('interactions_topic', value=data)
    print(f"Enviado: {data}")
    time.sleep(random.uniform(0.1, 1))  

print("¡Se han generado 20 interacciones!") 