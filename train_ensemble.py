"""Train multiple models with different seeds for ensemble."""
from train import main

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 2024]
    
    trained_models = []
    trained_submissions = []
    
    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model with seed {seed}")
        logger.info(f"{'='*60}\n")
        
        model_path, submission_path = main(seed=seed)
        trained_models.append(model_path)
        trained_submissions.append(submission_path)
        
        logger.info(f"\nModel saved: {model_path}")
        logger.info(f"Submission saved: {submission_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"All {len(seeds)} models trained successfully!")
    logger.info(f"{'='*60}")
    logger.info("\nTrained models:")
    for model in trained_models:
        logger.info(f"  - {model}")
    logger.info("\nNow run create_ensemble_submission.py to combine predictions")
