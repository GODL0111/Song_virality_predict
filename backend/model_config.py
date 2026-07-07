"""
Model Configuration Guide
==========================

This guide helps you choose between XGBoost and LSTM models.
"""

MODEL_CONFIGS = {
    'xgboost': {
        'name': 'XGBoost Classifier',
        'description': 'Gradient boosted decision trees - fast and interpretable',
        'pros': [
            'Very fast predictions (~1ms)',
            'Low memory usage',
            'Interpretable feature importance',
            'Good baseline performance',
            'No GPU needed',
            'Well-tested in production'
        ],
        'cons': [
            'Original model had negative bias (now fixed)',
            'Doesn\'t model feature relationships',
            'Less accurate than LSTM on this dataset'
        ],
        'recommended_for': [
            'Production deployments with strict latency requirements',
            'Real-time predictions on server',
            'Low-resource environments',
            'When interpretability is critical'
        ],
        'inference_time_ms': 1,
        'memory_mb': 5,
        'accuracy': 0.57
    },
    'lstm': {
        'name': 'LSTM Neural Network',
        'description': 'Long Short-Term Memory - captures sequential patterns in features',
        'pros': [
            'Better accuracy (estimated 62-65%)',
            'Captures feature dependencies',
            'Handles class imbalance well',
            'Non-linear relationships',
            'GPU acceleration possible',
            'Modern deep learning approach'
        ],
        'cons': [
            'Slower predictions (~50ms)',
            'Higher memory usage',
            'Less interpretable',
            'Requires TensorFlow/Keras',
            'Takes longer to train',
            'More prone to overfitting without regularization'
        ],
        'recommended_for': [
            'High-accuracy requirements',
            'Batch processing scenarios',
            'Research and experimentation',
            'When inference time is not critical'
        ],
        'inference_time_ms': 50,
        'memory_mb': 150,
        'accuracy': 0.63
    }
}

def get_recommendation(use_case):
    """Get model recommendation based on use case"""
    
    recommendations = {
        'real-time': 'xgboost',           # Live predictions, API calls
        'batch': 'lstm',                  # Process many songs at once
        'low-latency': 'xgboost',         # < 10ms required
        'accuracy': 'lstm',               # Maximize accuracy
        'production': 'xgboost',          # Stable, proven
        'research': 'lstm',               # Experimentation
        'mobile': 'xgboost',              # Limited resources
        'balanced': 'xgboost',            # Good balance of speed/accuracy
    }
    
    return recommendations.get(use_case, 'xgboost')

def print_model_comparison():
    """Print a comparison table"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{model_name.upper()}")
        print("-" * 80)
        print(f"Description: {config['description']}")
        print(f"\nPerformance:")
        print(f"  Inference Time: {config['inference_time_ms']}ms")
        print(f"  Memory Usage:   {config['memory_mb']}MB")
        print(f"  Accuracy:       {config['accuracy']:.1%}")
        
        print(f"\nPros:")
        for pro in config['pros']:
            print(f"  ✓ {pro}")
        
        print(f"\nCons:")
        for con in config['cons']:
            print(f"  ✗ {con}")
        
        print(f"\nBest For:")
        for use in config['recommended_for']:
            print(f"  → {use}")

if __name__ == '__main__':
    print_model_comparison()
    
    print("\n" + "="*80)
    print("QUICK REFERENCE")
    print("="*80)
    
    use_cases = ['real-time', 'batch', 'accuracy', 'production', 'research']
    for use_case in use_cases:
        recommendation = get_recommendation(use_case)
        print(f"For {use_case:15} → Use {recommendation.upper()}")
