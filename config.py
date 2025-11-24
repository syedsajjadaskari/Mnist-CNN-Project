class Config:
    # Image parameters
    IMG_SIZE = (224, 224)
    IMG_SHAPE = (*IMG_SIZE, 3)
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 3 
    FINE_TUNE_EPOCHS = 3
    LEARNING_RATE = 0.0001
    FINE_TUNE_LR = 0.00001
    
    # Data split ratios
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model parameters
    DROPOUT_RATE = 0.5
    DENSE_UNITS = 512
    
    # Paths
    MODEL_SAVE_PATH = 'models/best_model.h5'
    LOG_DIR = 'logs/fit'
    CHECKPOINT_DIR = 'models/checkpoints'
    PLOT_DIR = 'plots'
    
    # Dataset selection (choose one)
    DATASET = 'cifar10'  # Options: 'cifar10', 'cifar100', 'fashion_mnist'
    
    # Other settings
    SEED = 42
    VERBOSE = 1