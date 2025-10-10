import random
from locust import HttpUser, task, between

class MLModelUser(HttpUser):
    """
    User class that simulates requests to the ML model monitoring service.
    It tests both the prediction endpoint and the health check endpoint.
    """
    host = "http://localhost:8000"
    
    wait_time = between(0.5, 2.5)

    @task(10)
    def predict_passenger(self):
        """
        Sends a POST request to the /invocations endpoint with random but valid data.
        """
        payload = {
            "home_planet": random.choice(["Europa", "Earth", "Mars"]),
            "deck": random.choice(["B", "F", "G", "E", "C", "D", "A", "T"]),
            "side": random.choice(["P", "S"]),
            "age": round(random.uniform(1.0, 80.0), 1),
            "cabin_num": float(random.randint(0, 2000)),
            "room_service": float(random.randint(0, 5000)),
            "food_court": float(random.randint(0, 5000)),
            "shopping_mall": float(random.randint(0, 5000)),
            "spa": float(random.randint(0, 5000)),
            "vr_deck": float(random.randint(0, 5000))
        }

        with self.client.post("/invocations", json=payload, name="/invocations", catch_response=True) as response:
            if not response.ok:
                response.failure(f"Request failed with status {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Sends a GET request to the /health endpoint.
        """
        self.client.get("/health", name="/health")

