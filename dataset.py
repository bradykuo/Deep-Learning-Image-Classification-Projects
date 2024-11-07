import os
import shutil
import random
from pathlib import Path
import glob
import tqdm

def create_directory_structure(base_path):
    """Create the required directory structure"""
    # Main dataset directories
    directories = [
        'Dataset_OpenCvDl_Hw2_Q5/train/Cat',
        'Dataset_OpenCvDl_Hw2_Q5/train/Dog',
        'Dataset_OpenCvDl_Hw2_Q5/val/Cat',
        'Dataset_OpenCvDl_Hw2_Q5/val/Dog',
        'inference_dataset/Cat',
        'inference_dataset/Dog'
    ]
    
    for directory in directories:
        Path(os.path.join(base_path, directory)).mkdir(parents=True, exist_ok=True)

def prepare_dataset(kaggle_dataset_path, output_base_path):
    """Prepare the dataset according to specifications"""
    
    # Create directories
    create_directory_structure(output_base_path)
    
    # Get all cat and dog images
    cat_path = os.path.join(kaggle_dataset_path, 'PetImages', 'Cat')
    dog_path = os.path.join(kaggle_dataset_path, 'PetImages', 'Dog')
    
    # Check if directories exist
    if not os.path.exists(cat_path):
        print(f"Error: Cat directory not found at {cat_path}")
        return
    if not os.path.exists(dog_path):
        print(f"Error: Dog directory not found at {dog_path}")
        return
    
    # Get all cat and dog images
    cat_images = glob.glob(os.path.join(cat_path, '*.jpg'))
    dog_images = glob.glob(os.path.join(dog_path, '*.jpg'))
    
    # Print initial counts
    print(f"Found {len(cat_images)} cat images and {len(dog_images)} dog images")
    
    if len(cat_images) == 0 or len(dog_images) == 0:
        print("Error: No images found. Please check the path and file structure.")
        return
    
    # Remove corrupted images
    cat_images = [img for img in cat_images if os.path.getsize(img) > 0]
    dog_images = [img for img in dog_images if os.path.getsize(img) > 0]
    
    print(f"After removing corrupted images: {len(cat_images)} cat images and {len(dog_images)} dog images")
    
    # Check if we have enough images
    total_images_needed = 16200 + 1800 + 10  # training + validation + inference
    if len(cat_images) < total_images_needed/2 or len(dog_images) < total_images_needed/2:
        print(f"Error: Not enough images. Need at least {total_images_needed/2} images of each class.")
        print(f"Found only {len(cat_images)} cat images and {len(dog_images)} dog images.")
        return
    
    # Create imbalanced training set (70% dogs, 30% cats)
    num_train_total = 16200
    num_train_dogs = int(num_train_total * 0.7)
    num_train_cats = num_train_total - num_train_dogs
    
    # Validation set (equal distribution)
    num_val_total = 1800
    num_val_each = num_val_total // 2
    
    # Inference set
    num_inference = 5  # 5 each for cats and dogs
    
    # Randomly select images
    random.shuffle(cat_images)
    random.shuffle(dog_images)
    
    # Split images into sets
    train_cats = cat_images[:num_train_cats]
    train_dogs = dog_images[:num_train_dogs]
    
    val_cats = cat_images[num_train_cats:num_train_cats + num_val_each]
    val_dogs = dog_images[num_train_dogs:num_train_dogs + num_val_each]
    
    inference_cats = cat_images[-num_inference:]
    inference_dogs = dog_images[-num_inference:]
    
    # Function to copy images
    def copy_images(image_list, destination_dir, desc):
        for i, src in enumerate(tqdm.tqdm(image_list, desc=desc)):
            dst = os.path.join(destination_dir, f"{i+1}.jpg")
            shutil.copy2(src, dst)
    
    # Copy images to respective directories
    copy_images(train_cats, os.path.join(output_base_path, 'Dataset_OpenCvDl_Hw2_Q5/train/Cat'), 'Copying training cats')
    copy_images(train_dogs, os.path.join(output_base_path, 'Dataset_OpenCvDl_Hw2_Q5/train/Dog'), 'Copying training dogs')
    copy_images(val_cats, os.path.join(output_base_path, 'Dataset_OpenCvDl_Hw2_Q5/val/Cat'), 'Copying validation cats')
    copy_images(val_dogs, os.path.join(output_base_path, 'Dataset_OpenCvDl_Hw2_Q5/val/Dog'), 'Copying validation dogs')
    copy_images(inference_cats, os.path.join(output_base_path, 'inference_dataset/Cat'), 'Copying inference cats')
    copy_images(inference_dogs, os.path.join(output_base_path, 'inference_dataset/Dog'), 'Copying inference dogs')

def print_dataset_stats(base_path):
    """Print statistics about the prepared dataset"""
    def count_images(directory):
        return len([f for f in os.listdir(directory) if f.endswith('.jpg')])
    
    # Count images in each directory
    train_cats = count_images(os.path.join(base_path, 'Dataset_OpenCvDl_Hw2_Q5/train/Cat'))
    train_dogs = count_images(os.path.join(base_path, 'Dataset_OpenCvDl_Hw2_Q5/train/Dog'))
    val_cats = count_images(os.path.join(base_path, 'Dataset_OpenCvDl_Hw2_Q5/val/Cat'))
    val_dogs = count_images(os.path.join(base_path, 'Dataset_OpenCvDl_Hw2_Q5/val/Dog'))
    inf_cats = count_images(os.path.join(base_path, 'inference_dataset/Cat'))
    inf_dogs = count_images(os.path.join(base_path, 'inference_dataset/Dog'))
    
    print("\nDataset Statistics:")
    print(f"Training Set (Total: {train_cats + train_dogs}):")
    print(f"  - Cats: {train_cats} ({train_cats/(train_cats + train_dogs)*100:.1f}%)")
    print(f"  - Dogs: {train_dogs} ({train_dogs/(train_cats + train_dogs)*100:.1f}%)")
    print(f"\nValidation Set (Total: {val_cats + val_dogs}):")
    print(f"  - Cats: {val_cats}")
    print(f"  - Dogs: {val_dogs}")
    print(f"\nInference Set (Total: {inf_cats + inf_dogs}):")
    print(f"  - Cats: {inf_cats}")
    print(f"  - Dogs: {inf_dogs}")

def main():
    # Set paths
    base_path = "/Users/brady/Desktop/程式專案/Train a Cat-Dog Classifier Using ResNet50"
    
    # Verify the base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
        
    # The base path should contain the PetImages folder
    kaggle_dataset_path = base_path
    output_base_path = base_path
    
    print(f"Looking for images in: {os.path.join(kaggle_dataset_path, 'PetImages')}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    prepare_dataset(kaggle_dataset_path, output_base_path)
    
    # Only print stats if the dataset was successfully created
    if os.path.exists(os.path.join(output_base_path, 'Dataset_OpenCvDl_Hw2_Q5/train/Cat')):
        print_dataset_stats(output_base_path)
        print("\nDataset preparation completed!")
    else:
        print("\nDataset preparation failed!")

if __name__ == "__main__":
    main()