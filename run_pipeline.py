import os
import sys
import subprocess
from house_price_prediction.logger import get_logger

logger = get_logger(__name__)

def run_command(command, description):
    logger.info(f"Starting: {description}")
    try:
        # Use sys.executable to ensure we use the same python interpreter
        cmd = [sys.executable, '-m'] + command.split(' ')
        subprocess.check_call(cmd)
        logger.info(f"Completed: {description}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(str(e))
        sys.exit(1)

def main():
    logger.info("=== Starting House Price Prediction Pipeline ===")
    
    # 1. Train Models
    run_command("house_price_prediction.training.train_pipeline", "Model Training")
    
    # 2. Evaluate Best Model (Lasso)
    run_command("house_price_prediction.evaluation.run_evaluation", "Model Evaluation")
    
    # 3. Explainability (SHAP)
    run_command("house_price_prediction.explainability.shap_analysis", "SHAP Analysis")
    
    # 4. Run Tests
    run_command("house_price_prediction.tests.test_api", "API Tests")
    
    logger.info("=== Pipeline Completed Successfully ===")
    logger.info("You can now start the API with: uvicorn house_price_prediction.api.app:app --reload")

if __name__ == "__main__":
    main()
