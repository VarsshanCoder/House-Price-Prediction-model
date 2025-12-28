from fastapi.testclient import TestClient
from house_price_prediction.api.app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to House Price Prediction API"}

def test_predict_price():
    payload = {
        "area": 1500,
        "location": "CollgCr",
        "bedrooms": 3,
        "bathrooms": 2,
        "amenities": ["parking", "pool"]
    }
    response = client.post("/predict-price", json=payload)
    if response.status_code != 200:
        print(response.text)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert "confidence_range" in data
    assert isinstance(data["predicted_price"], float)
    print(f"Prediction: {data['predicted_price']}")

if __name__ == "__main__":
    test_read_root()
    test_predict_price()
    print("All tests passed!")
