# House Price Prediction System

## 🎯 Project Overview
This project builds a complete, production-ready House Price Prediction System. It estimates property prices using features such as area, location, bedrooms, bathrooms, and amenities. The system includes a robust Machine Learning pipeline, a REST API, and a modern user interface.

## 🏗 Architecture
The project follows a modular architecture:
- **Data**: Raw training and test data.
- **Preprocessing**: Cleaning, encoding, and scaling pipelines.
- **Features**: Feature engineering logic.
- **Models**: Model registry and definitions (XGBoost, etc.).
- **Training**: Training pipeline with cross-validation.
- **Evaluation**: Metrics and visualization.
- **Explainability**: SHAP analysis for model transparency.
- **Inference**: Prediction logic for new data.
- **API**: FastAPI backend.
- **Frontend**: HTML/CSS/JS user interface.
- **Deployment**: Docker configuration.

## 🚀 Setup & Usage

### Prerequisites
- Python 3.9+
- Docker (optional)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r house_price_prediction/requirements.txt
   pip install fastapi uvicorn
   ```

### Quick Start (Run Everything)
To run the full pipeline (training, evaluation, explainability, tests) in one go:
```bash
python run_pipeline.py
```

### Training the Model
To train the model manually:
```bash
python -m house_price_prediction.training.train_pipeline
```
This will save the trained pipeline to `house_price_prediction/output/models/`.

### Hyperparameter Tuning
To tune the RandomForest model using GridSearchCV:
```bash
python -m house_price_prediction.training.tune_model
```
This will save the best found model to `house_price_prediction/output/models/RandomForest_tuned_pipeline.joblib`.

### Running the API
Start the FastAPI server:
```bash
uvicorn house_price_prediction.api.app:app --reload
```
The API will be available at `http://localhost:8000`.
Docs: `http://localhost:8000/docs`.

### Using the Frontend
The frontend is served directly by the API.
Open your browser to: `http://localhost:8000/app/index.html`

Alternatively, you can open `house_price_prediction/frontend/index.html` directly in your browser.

![Frontend Interface](https://github.com/VarsshanCoder/House-Price-Prediction-model/blob/main/house_price_prediction/frontend_screenshot.png)

## 📊 Model Performance
The current best model (XGBoost) achieves:
- **R² Score**: ~0.90
- **RMSE**: ~26,000
- **MAE**: ~16,000

## 🔍 Explainability
We use SHAP (SHapley Additive exPlanations) to understand model predictions.
Check `house_price_prediction/output/explainability/` for summary and force plots.

## 🗺️ Visualization & Spatial Insights
We analyze spatial trends by plotting average house prices per neighborhood.
Check `house_price_prediction/output/evaluation/price_by_neighborhood.png` for insights.

**Why Spatial Data Matters:**
Location is often the single most important factor in real estate. By analyzing price trends across different neighborhoods (spatial clusters), the model captures:
- **Socio-economic factors**: School districts, crime rates, and income levels often correlate with location.
- **Proximity to amenities**: Distance to city centers, parks, and transport hubs.
- **Neighborhood prestige**: Historical value and desirability.
Including location data significantly improves prediction accuracy by accounting for these latent variables that physical house features (like sqft) cannot explain alone.

## 🐳 Deployment
Build the Docker image:
```bash
docker build -t house-price-predictor -f house_price_prediction/deployment/Dockerfile .
```
Run the container:
```bash
docker run -p 8000:8000 house-price-predictor
```
## ☁️ Cloud Deployment

### Render / Railway (PaaS)
1. Push this repository to GitHub.
2. Connect your repository to Render/Railway.
3. Set the Build Command: `pip install -r house_price_prediction/requirements.txt`
4. Set the Start Command: `uvicorn house_price_prediction.api.app:app --host 0.0.0.0 --port $PORT`

### AWS / GCP (Containerized)
1. Build the Docker image.
2. Push the image to ECR (AWS) or GCR (GCP).
3. Deploy using ECS/App Runner (AWS) or Cloud Run (GCP).
   ```bash
   # Example for GCP Cloud Run
   gcloud run deploy house-price-predictor --image gcr.io/PROJECT-ID/house-price-predictor --platform managed
   ```

## ⚖️ Ethical Considerations
- **Bias**: The model is trained on historical data which may contain biases (e.g., redlining effects in location data). Care must be taken when using it for decision-making.
- **Fairness**: We should regularly evaluate model performance across different demographic groups (proxied by location) to ensure fairness.
- **Transparency**: SHAP values are provided to explain individual predictions, helping users understand *why* a price was estimated.
- **Privacy**: The dataset is anonymized, but in production, user data must be handled securely according to GDPR/CCPA.

## 🔮 Future Improvements
- Integrate interactive map-based visualization.
- Add more advanced feature engineering.
- Deploy to cloud (AWS/GCP).
- Implement A/B testing for model updates.
