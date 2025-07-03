import os
import shutil
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


class YOLOv8ClassFolderCombiner:
    def __init__(self, existing_dataset_path: str, new_classes_folder: str, output_path: str):
        self.existing_dataset_path = Path(existing_dataset_path)
        self.new_classes_folder = Path(new_classes_folder)
        self.output_path = Path(output_path)
        
        # Validate input paths
        if not self.existing_dataset_path.exists():
            raise FileNotFoundError(f"Existing dataset path does not exist: {existing_dataset_path}")
        if not self.new_classes_folder.exists():
            raise FileNotFoundError(f"New classes folder does not exist: {new_classes_folder}")
        
        # Load existing dataset configuration
        self.existing_config = self._load_yaml_config(self.existing_dataset_path / "data.yaml")
        self.existing_classes = self._normalize_classes(self.existing_config['names'])
        
        # Discover new classes from folder structure
        self.new_classes = self._discover_new_classes()
        
        # Create combined class mapping
        self.combined_classes, self.new_class_mapping = self._create_class_mapping()
        
    def _load_yaml_config(self, yaml_path: Path) -> Dict:
        """Load YAML configuration file"""
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'names' not in config:
                raise ValueError(f"'names' key not found in {yaml_path}")
            
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {yaml_path}: {e}")
    
    def _normalize_classes(self, classes) -> Dict[int, str]:
        """Normalize classes to dict format {id: name}"""
        if isinstance(classes, dict):
            return classes
        elif isinstance(classes, list):
            return {i: name for i, name in enumerate(classes)}
        else:
            raise ValueError(f"Unsupported classes format: {type(classes)}")
    
    def _discover_new_classes(self) -> Dict[str, Path]:
        """Discover new classes from folder structure"""
        new_classes = {}
        
        for class_folder in self.new_classes_folder.iterdir():
            if class_folder.is_dir():
                images_dir = class_folder / "images"
                labels_dir = class_folder / "labels"
                
                if images_dir.exists() and labels_dir.exists():
                    class_name = class_folder.name
                    new_classes[class_name] = class_folder
                    print(f"Found new class: '{class_name}' at {class_folder}")
                else:
                    print(f"Warning: Skipping {class_folder} - missing images or labels directory")
        
        if not new_classes:
            raise ValueError("No valid class folders found in new classes directory")
        
        return new_classes
    
    def _create_class_mapping(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Create combined class mapping"""
        combined_classes = self.existing_classes.copy()
        new_class_mapping = {}
        
        # Get set of existing class names for overlap detection
        existing_class_names = set(self.existing_classes.values())
        
        # Add new classes that don't overlap with existing ones
        next_class_id = max(self.existing_classes.keys()) + 1 if self.existing_classes else 0
        
        for class_name in self.new_classes.keys():
            if class_name in existing_class_names:
                # Overlapping class - map to existing class ID
                existing_id = next(k for k, v in self.existing_classes.items() if v == class_name)
                new_class_mapping[class_name] = existing_id
                print(f"  Mapping overlapping class '{class_name}' to existing ID: {existing_id}")
            else:
                # New unique class - add to combined classes
                combined_classes[next_class_id] = class_name
                new_class_mapping[class_name] = next_class_id
                print(f"  Adding new class '{class_name}' with ID: {next_class_id}")
                next_class_id += 1
        
        return combined_classes, new_class_mapping
    
    def _get_existing_image_label_pairs(self, split: str) -> List[Tuple[Path, Path]]:
        """Get pairs of image and label files from existing dataset for a given split"""
        # Try both directory structures
        images_dir = self.existing_dataset_path / "images" / split
        labels_dir = self.existing_dataset_path / "labels" / split
        
        if not images_dir.exists():
            images_dir = self.existing_dataset_path / split / "images"
            labels_dir = self.existing_dataset_path / split / "labels"
        
        pairs = []
        if images_dir.exists() and labels_dir.exists():
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            
            for img_file in images_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        pairs.append((img_file, label_file))
                    else:
                        print(f"  Warning: Label file not found for {img_file.name}")
        else:
            print(f"  Warning: Images or labels directory not found for {split} split")
        
        return pairs
    
    def _get_new_class_image_label_pairs(self, class_name: str) -> List[Tuple[Path, Path, str]]:
        """Get pairs of image and label files for a new class"""
        class_folder = self.new_classes[class_name]
        images_dir = class_folder / "images"
        labels_dir = class_folder / "labels"
        
        pairs = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    pairs.append((img_file, label_file, class_name))
                else:
                    print(f"  Warning: Label file not found for {img_file.name} in class {class_name}")
        
        return pairs
    
    def _sample_dataset(self, pairs: List[Tuple], percentage: float) -> List[Tuple]:
        """Sample a percentage of pairs"""
        if percentage >= 1.0:
            return pairs
        if percentage <= 0.0:
            return []
        
        sample_size = max(1, int(len(pairs) * percentage))
        return random.sample(pairs, min(sample_size, len(pairs)))
    
    def _update_existing_label_file(self, label_path: Path) -> List[str]:
        """Read existing label file (already has class IDs)"""
        updated_lines = []
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            parts = line.split()
                            if len(parts) < 5:  # YOLO format: class_id x_center y_center width height
                                print(f"  Warning: Invalid annotation format in {label_path}:{line_num}")
                                continue
                            updated_lines.append(line)
                        except (ValueError, IndexError) as e:
                            print(f"  Warning: Error parsing line {line_num} in {label_path}: {e}")
        except Exception as e:
            print(f"  Error reading label file {label_path}: {e}")
        
        return updated_lines
    
    def _update_new_class_label_file(self, label_path: Path, class_name: str) -> List[str]:
        """Update label file by adding class ID (currently only has x, y, w, h)"""
        updated_lines = []
        class_id = self.new_class_mapping[class_name]
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            parts = line.split()
                            if len(parts) < 4:  # Should have x_center y_center width height
                                print(f"  Warning: Invalid annotation format in {label_path}:{line_num}")
                                continue
                            
                            # Add class ID at the beginning
                            updated_line = f"{class_id} {' '.join(parts)}"
                            updated_lines.append(updated_line)
                        except (ValueError, IndexError) as e:
                            print(f"  Warning: Error parsing line {line_num} in {label_path}: {e}")
        except Exception as e:
            print(f"  Error reading label file {label_path}: {e}")
        
        return updated_lines
    
    def _copy_existing_files(self, pairs: List[Tuple[Path, Path]], 
                           output_images_dir: Path, output_labels_dir: Path, prefix: str = ""):
        """Copy existing dataset files"""
        for img_path, label_path in pairs:
            try:
                # Generate new filename with prefix to avoid conflicts
                new_filename = f"{prefix}{img_path.name}" if prefix else img_path.name
                
                # Copy image file
                shutil.copy2(img_path, output_images_dir / new_filename)
                
                # Copy label file (no modification needed for existing dataset)
                updated_labels = self._update_existing_label_file(label_path)
                new_label_path = output_labels_dir / f"{Path(new_filename).stem}.txt"
                
                if updated_labels:
                    with open(new_label_path, 'w', encoding='utf-8') as f:
                        for label_line in updated_labels:
                            f.write(label_line + '\n')
                else:
                    new_label_path.touch()
                    
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
    
    def _copy_new_class_files(self, pairs: List[Tuple[Path, Path, str]], 
                            output_images_dir: Path, output_labels_dir: Path):
        """Copy new class files and update labels with class IDs"""
        for img_path, label_path, class_name in pairs:
            try:
                # Generate new filename with class prefix
                new_filename = f"{class_name}_{img_path.name}"
                
                # Copy image file
                shutil.copy2(img_path, output_images_dir / new_filename)
                
                # Update and copy label file (add class ID)
                updated_labels = self._update_new_class_label_file(label_path, class_name)
                new_label_path = output_labels_dir / f"{Path(new_filename).stem}.txt"
                
                if updated_labels:
                    with open(new_label_path, 'w', encoding='utf-8') as f:
                        for label_line in updated_labels:
                            f.write(label_line + '\n')
                else:
                    new_label_path.touch()
                    
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
    
    def _distribute_new_classes_to_splits(self, split_ratios: Dict[str, float] = None) -> Dict[str, List]:
        """Distribute new class images across splits"""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.15, 'test': 0.05}
        
        # Normalize ratios to sum to 1.0
        total_ratio = sum(split_ratios.values())
        split_ratios = {k: v/total_ratio for k, v in split_ratios.items()}
        
        split_data = {split: [] for split in split_ratios.keys()}
        
        for class_name in self.new_classes.keys():
            # Get all pairs for this class
            class_pairs = self._get_new_class_image_label_pairs(class_name)
            
            if not class_pairs:
                print(f"  Warning: No valid pairs found for class {class_name}")
                continue
            
            # Shuffle for random distribution
            random.shuffle(class_pairs)
            
            # Distribute across splits
            start_idx = 0
            for split, ratio in split_ratios.items():
                if ratio > 0:
                    split_size = int(len(class_pairs) * ratio)
                    if split == list(split_ratios.keys())[-1]:  # Last split gets remainder
                        split_pairs = class_pairs[start_idx:]
                    else:
                        split_pairs = class_pairs[start_idx:start_idx + split_size]
                    
                    split_data[split].extend(split_pairs)
                    start_idx += split_size
                    
                    print(f"  Assigned {len(split_pairs)} samples from '{class_name}' to {split}")
        
        return split_data
    
    def combine_datasets(self, existing_percentage: float = 0.2, 
                        split_ratios: Dict[str, float] = None):
        """Combine existing dataset with new class folders"""
        print(f"Combining datasets:")
        print(f"Existing dataset: {self.existing_dataset_path}")
        print(f"New classes folder: {self.new_classes_folder}")
        print(f"Output: {self.output_path}")
        print(f"Using {existing_percentage*100:.1f}% of existing dataset")
        print(f"\nClass mapping:")
        print(f"Existing classes: {dict(sorted(self.existing_classes.items()))}")
        print(f"New classes: {list(self.new_classes.keys())}")
        print(f"Combined classes: {dict(sorted(self.combined_classes.items()))}")
        
        # Create output directory structure
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Distribute new class data across splits
        print(f"\nDistributing new class data across splits...")
        new_class_split_data = self._distribute_new_classes_to_splits(split_ratios)
        
        # Determine which splits to process
        available_splits = set()
        for split in ['train', 'val', 'valid', 'test']:
            if self._get_existing_image_label_pairs(split) or split in new_class_split_data:
                available_splits.add(split)
        
        # Map 'valid' to 'val' for consistency
        if 'valid' in available_splits:
            available_splits.remove('valid')
            available_splits.add('val')
        
        print(f"Processing splits: {sorted(available_splits)}")
        
        for split in sorted(available_splits):
            print(f"\nProcessing {split} split...")
            
            # Create output directories
            output_images_dir = self.output_path / split / "images"
            output_labels_dir = self.output_path / split / "labels"
            output_images_dir.mkdir(parents=True, exist_ok=True)
            output_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Process existing dataset
            existing_pairs = self._get_existing_image_label_pairs(split)
            if existing_pairs:
                sampled_existing_pairs = self._sample_dataset(existing_pairs, existing_percentage)
                print(f"  Existing dataset: {len(sampled_existing_pairs)}/{len(existing_pairs)} samples")
                
                self._copy_existing_files(
                    sampled_existing_pairs, output_images_dir, output_labels_dir, "existing_"
                )
            else:
                print(f"  Existing dataset: No samples found for {split}")
            
            # Process new class data
            if split in new_class_split_data and new_class_split_data[split]:
                print(f"  New classes: {len(new_class_split_data[split])} samples")
                self._copy_new_class_files(
                    new_class_split_data[split], output_images_dir, output_labels_dir
                )
            else:
                print(f"  New classes: No samples for {split}")
        
        # Create combined YAML configuration
        self._create_combined_yaml()
        
        print(f"\nDataset combination complete!")
        print(f"Output directory: {self.output_path.absolute()}")
    
    def _create_combined_yaml(self):
        """Create YAML configuration for combined dataset"""
        combined_config = self.existing_config.copy()
        
        # Update paths
        combined_config['path'] = str(self.output_path.absolute())
        combined_config['train'] = 'train/images'
        combined_config['val'] = 'val/images'
        
        # Add test split if it exists
        if (self.output_path / "test" / "images").exists():
            combined_config['test'] = 'test/images'
        elif 'test' in combined_config:
            del combined_config['test']
        
        # Update classes
        combined_config['nc'] = len(self.combined_classes)
        combined_config['names'] = [self.combined_classes[i] for i in sorted(self.combined_classes.keys())]
        
        # Remove any dataset-specific keys
        keys_to_remove = ['roboflow']
        for key in keys_to_remove:
            if key in combined_config:
                del combined_config[key]
        
        # Save combined YAML
        yaml_path = self.output_path / "data.yaml"
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(combined_config, f, default_flow_style=False, allow_unicode=True)
            print(f"Created combined configuration: {yaml_path}")
        except Exception as e:
            print(f"Error creating YAML file: {e}")


def combine_yolo_datasets(
    existing_dataset_path: str,
    new_classes_folder: str,
    output_path: str,
    existing_percentage: float = 0.2,
    split_ratios: Optional[Dict[str, float]] = None,
    random_seed: int = 42
) -> bool:
    """
    Combine existing YOLOv8 dataset with new class folders.
    
    Args:
        existing_dataset_path (str): Path to existing YOLOv8 dataset directory
        new_classes_folder (str): Path to new classes folder
        output_path (str): Output path for combined dataset
        existing_percentage (float): Percentage of existing dataset to include (0.0-1.0)
        split_ratios (dict): Ratios for train/val/test splits (e.g., {'train': 0.8, 'val': 0.15, 'test': 0.05})
        random_seed (int): Random seed for reproducible sampling
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        >>> from yolo_dataset_combiner import combine_yolo_datasets
        >>> 
        >>> success = combine_yolo_datasets(
        ...     existing_dataset_path="./existing_dataset",
        ...     new_classes_folder="./new_classes",
        ...     output_path="./combined_dataset",
        ...     existing_percentage=0.3,
        ...     split_ratios={'train': 0.8, 'val': 0.15, 'test': 0.05},
        ...     random_seed=42
        ... )
        >>> 
        >>> if success:
        ...     print("Dataset combination successful!")
        ... else:
        ...     print("Dataset combination failed!")
    
    New classes folder structure should be:
        new_classes/
        ├── class1/
        │   ├── images/
        │   └── labels/
        ├── class2/
        │   ├── images/
        │   └── labels/
        └── ...
    """
    # Validate inputs
    if not (0 <= existing_percentage <= 1):
        print("Error: existing_percentage must be between 0 and 1")
        return False
    
    if split_ratios is None:
        split_ratios = {'train': 0.8, 'val': 0.15, 'test': 0.05}
    
    # Normalize ratios
    total_ratio = sum(split_ratios.values())
    if total_ratio <= 0:
        print("Error: Sum of split ratios must be greater than 0")
        return False
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    try:
        # Create combiner and combine datasets
        combiner = YOLOv8ClassFolderCombiner(
            existing_dataset_path, new_classes_folder, output_path
        )
        combiner.combine_datasets(existing_percentage, split_ratios)
        print("\n✅ Dataset combination completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False