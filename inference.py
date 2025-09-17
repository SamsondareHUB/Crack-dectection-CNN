import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt

class CrackDetectionInference:
    def _init_(self, model_path, img_height=224, img_width=224):
        """
        Initialize the inference class
        
        Args:
            model_path (str): Path to the trained model
            img_height (int): Input image height
            img_width (int): Input image width
        """
        self.img_height = img_height
        self.img_width = img_width
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """
        Load the trained model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_single_image(self, image_path, show_image=False):
        """
        Predict crack for a single image
        
        Args:
            image_path (str): Path to the image
            show_image (bool): Whether to display the image
            
        Returns:
            tuple: (prediction_text, confidence_score, prediction_probability)
        """
        # Preprocess image
        processed_img = self.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(processed_img, verbose=0)
        confidence = prediction[0][0]
        
        # Interpret prediction
        if confidence > 0.5:
            prediction_text = "CRACK DETECTED"
            confidence_score = confidence
        else:
            prediction_text = "NO CRACK DETECTED"
            confidence_score = 1 - confidence
        
        # Display image if requested
        if show_image:
            self.display_prediction(image_path, prediction_text, confidence_score)
        
        return prediction_text, confidence_score, confidence
    
    def display_prediction(self, image_path, prediction_text, confidence_score):
        """
        Display image with prediction
        
        Args:
            image_path (str): Path to the image
            prediction_text (str): Prediction result text
            confidence_score (float): Confidence score
        """
        # Load and display image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        
        # Color code the prediction
        color = 'red' if 'CRACK' in prediction_text else 'green'
        plt.title(f'{prediction_text}\nConfidence: {confidence_score:.2%}', 
                 fontsize=16, color=color, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def batch_predict(self, image_folder, output_file=None):
        """
        Predict cracks for multiple images in a folder
        
        Args:
            image_folder (str): Path to folder containing images
            output_file (str): Path to save results (optional)
            
        Returns:
            list: List of prediction results
        """
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {image_folder}")
            return []
        
        results = []
        print(f"Processing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files, 1):
            try:
                prediction_text, confidence_score, raw_confidence = self.predict_single_image(str(image_file))
                
                result = {
                    'image_name': image_file.name,
                    'prediction': prediction_text,
                    'confidence': confidence_score,
                    'raw_probability': raw_confidence
                }
                results.append(result)
                
                print(f"[{i}/{len(image_files)}] {image_file.name}: {prediction_text} ({confidence_score:.2%})")
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                continue
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results, output_file):
        """
        Save prediction results to file
        
        Args:
            results (list): List of prediction results
            output_file (str): Output file path
        """
        try:
            with open(output_file, 'w') as f:
                f.write("Image Name,Prediction,Confidence,Raw Probability\n")
                for result in results:
                    f.write(f"{result['image_name']},{result['prediction']},"
                           f"{result['confidence']:.4f},{result['raw_probability']:.4f}\n")
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Crack Detection Inference')
    parser.add_argument('--model', type=str, default='crack_detection_model.h5',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--folder', type=str, help='Path to folder with images for batch prediction')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    parser.add_argument('--show', action='store_true', help='Show image with prediction')
    
    args = parser.parse_args()
    
    # Initialize inference
    try:
        detector = CrackDetectionInference(args.model)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Single image prediction
    if args.image:
        try:
            prediction, confidence, _ = detector.predict_single_image(args.image, args.show)
            print(f"\nImage: {args.image}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.2%}")
        except Exception as e:
            print(f"Error predicting image: {e}")
    
    # Batch prediction
    elif args.folder:
        try:
            results = detector.batch_predict(args.folder, args.output)
            if results:
                crack_count = sum(1 for r in results if 'CRACK' in r['prediction'])
                print(f"\nSummary: {crack_count}/{len(results)} images contain cracks")
        except Exception as e:
            print(f"Error in batch prediction: {e}")
    
    else:
        print("Please specify --image or --folder")
        parser.print_help()

if _name_ == "_main_":
    main()