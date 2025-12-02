"""
RNN-based Image Filter Model
Uses LSTM to process images sequentially and apply artistic filters
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List
import cv2

class RNNImageFilter:
    """
    RNN model for applying artistic filters to images
    Processes images row-by-row using LSTM layers
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 3)):
        """
        Initialize RNN Image Filter
        
        Args:
            input_shape: Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.height, self.width, self.channels = input_shape
        self.model = None
        
    def build_model(self, filter_type: str = 'sketch') -> keras.Model:
        """
        Build RNN model for image filtering
        
        Args:
            filter_type: Type of filter to apply
            
        Returns:
            Compiled Keras model
        """
        # Input: Image as sequence of rows
        inputs = layers.Input(shape=(self.height, self.width * self.channels))
        
        # Bidirectional LSTM to process image rows
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        )(inputs)
        
        x = layers.Dropout(0.3)(x)
        
        # Second LSTM layer for feature refinement
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True)
        )(x)
        
        x = layers.Dropout(0.2)(x)
        
        # Dense layers for transformation
        x = layers.TimeDistributed(
            layers.Dense(256, activation='relu')
        )(x)
        
        x = layers.TimeDistributed(
            layers.Dense(128, activation='relu')
        )(x)
        
        # Output layer - reconstruct filtered image
        outputs = layers.TimeDistributed(
            layers.Dense(self.width * self.channels, activation='sigmoid')
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=f'rnn_filter_{filter_type}')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for RNN input
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        # Resize image
        image = cv2.resize(image, (self.width, self.height))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Reshape to (height, width * channels)
        image_seq = image.reshape(self.height, self.width * self.channels)
        
        return image_seq
    
    def postprocess_image(self, output: np.ndarray) -> np.ndarray:
        """
        Postprocess RNN output to image
        
        Args:
            output: RNN output array
            
        Returns:
            Image array
        """
        # Reshape back to image
        image = output.reshape(self.height, self.width, self.channels)
        
        # Clip values to [0, 1]
        image = np.clip(image, 0, 1)
        
        # Convert back to uint8
        image = (image * 255).astype(np.uint8)
        
        return image


class ImageFilterPresets:
    """
    Traditional image processing filters that can be used
    with or without RNN enhancement
    """
    
    @staticmethod
    def apply_sketch_filter(image: np.ndarray) -> np.ndarray:
        """Apply pencil sketch filter"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blur = 255 - blurred
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        return sketch
    
    @staticmethod
    def apply_watercolor_filter(image: np.ndarray) -> np.ndarray:
        """Apply watercolor painting filter"""
        # Bilateral filter for edge-preserving smoothing
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply median blur for painting effect
        watercolor = cv2.medianBlur(filtered, 7)
        
        # Enhance colors
        hsv = cv2.cvtColor(watercolor, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Increase saturation
        watercolor = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return np.clip(watercolor, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_oil_painting_filter(image: np.ndarray) -> np.ndarray:
        """Apply oil painting filter"""
        # Convert to BGR for OpenCV
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Apply oil painting effect
        oil = cv2.xphoto.oilPainting(bgr, 7, 1)
        
        # Convert back to RGB
        oil = cv2.cvtColor(oil, cv2.COLOR_BGR2RGB)
        
        return oil
    
    @staticmethod
    def apply_cartoon_filter(image: np.ndarray) -> np.ndarray:
        """Apply cartoon filter"""
        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 9
        )
        
        # Color quantization
        data = np.float32(image).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(image.shape)
        
        # Combine edges and colors
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        cartoon = cv2.bitwise_and(quantized, edges_colored)
        
        return cartoon
    
    @staticmethod
    def apply_vintage_filter(image: np.ndarray) -> np.ndarray:
        """Apply vintage/sepia filter"""
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        
        vintage = cv2.transform(image, kernel)
        vintage = np.clip(vintage, 0, 255).astype(np.uint8)
        
        # Add slight vignette
        rows, cols = image.shape[:2]
        X_kernel = cv2.getGaussianKernel(cols, cols/2)
        Y_kernel = cv2.getGaussianKernel(rows, rows/2)
        kernel = Y_kernel * X_kernel.T
        mask = kernel / kernel.max()
        
        vintage = vintage.astype(np.float32)
        for i in range(3):
            vintage[:, :, i] = vintage[:, :, i] * mask
        
        return np.clip(vintage, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_edge_enhancement(image: np.ndarray) -> np.ndarray:
        """Apply edge enhancement filter"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend with original
        enhanced = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
        
        return enhanced
    
    @staticmethod
    def apply_emboss_filter(image: np.ndarray) -> np.ndarray:
        """Apply emboss filter"""
        kernel = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        embossed = cv2.filter2D(gray, -1, kernel)
        embossed = embossed + 128  # Add offset
        embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2RGB)
        
        return np.clip(embossed, 0, 255).astype(np.uint8)


class RNNFilterTrainer:
    """
    Trainer for RNN-based image filters
    Learns to transform images using sequential processing
    """
    
    def __init__(self, model: RNNImageFilter):
        """
        Initialize trainer
        
        Args:
            model: RNNImageFilter instance
        """
        self.model = model
        self.history = None
    
    def prepare_training_data(
        self, 
        original_images: List[np.ndarray],
        filtered_images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from image pairs
        
        Args:
            original_images: List of original images
            filtered_images: List of filtered target images
            
        Returns:
            X_train, y_train arrays
        """
        X_train = []
        y_train = []
        
        for orig, filt in zip(original_images, filtered_images):
            X_train.append(self.model.preprocess_image(orig))
            y_train.append(self.model.preprocess_image(filt))
        
        return np.array(X_train), np.array(y_train)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 8,
        validation_split: float = 0.2
    ):
        """
        Train the RNN model
        
        Args:
            X_train: Training input images
            y_train: Training target images
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation data split ratio
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        self.history = self.model.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
