import tensorflow as tf
import tensorflow_datasets as tfds
from config import Config

class DataLoader:
    """Load and prepare datasets from TensorFlow"""
    
    def __init__(self, dataset_name=Config.DATASET):
        self.dataset_name = dataset_name
        self.num_classes = None
        self.class_names = None
        
    def load_data(self):
        """Load dataset from TensorFlow Datasets"""
        
        print(f"\nLoading {self.dataset_name} dataset...")
        
        if self.dataset_name == 'cifar10':
            (train_ds, val_ds, test_ds), info = tfds.load(
                'cifar10',
                split=['train[:80%]', 'train[80%:]', 'test'],
                as_supervised=True,
                with_info=True
            )
            self.num_classes = 10
            self.class_names = info.features['label'].names
            
        elif self.dataset_name == 'cifar100':
            (train_ds, val_ds, test_ds), info = tfds.load(
                'cifar100',
                split=['train[:80%]', 'train[80%:]', 'test'],
                as_supervised=True,
                with_info=True
            )
            self.num_classes = 100
            self.class_names = info.features['label'].names
            
        elif self.dataset_name == 'fashion_mnist':
            (train_ds, val_ds, test_ds), info = tfds.load(
                'fashion_mnist',
                split=['train[:80%]', 'train[80%:]', 'test'],
                as_supervised=True,
                with_info=True
            )
            self.num_classes = 10
            self.class_names = info.features['label'].names
            
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
        
        return train_ds, val_ds, test_ds
    
    @staticmethod
    def preprocess_image(image, label):
        """Resize and normalize image"""
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, Config.IMG_SIZE)
        image = image / 255.0
        return image, label
    
    @staticmethod
    def augment_image(image, label):
        """Apply data augmentation"""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label
    
    def prepare_dataset(self, train_ds, val_ds, test_ds):
        """Prepare datasets with preprocessing and augmentation"""
        
        # Training dataset with augmentation
        train_ds = (train_ds
                   .map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                   .map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                   .shuffle(1000)
                   .batch(Config.BATCH_SIZE)
                   .prefetch(tf.data.AUTOTUNE))
        
        # Validation dataset
        val_ds = (val_ds
                 .map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(Config.BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))
        
        # Test dataset
        test_ds = (test_ds
                  .map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(Config.BATCH_SIZE)
                  .prefetch(tf.data.AUTOTUNE))
        
        return train_ds, val_ds, test_ds