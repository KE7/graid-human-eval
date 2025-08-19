"""
Dataset loading and sampling utilities for GRAID human evaluation.

This module handles loading HuggingFace datasets and performing stratified sampling
by question type to ensure balanced evaluation across all question categories.
"""

import hashlib
import logging
from typing import Dict, List, Tuple, Any
import random

from datasets import load_dataset

logger = logging.getLogger(__name__)


class DatasetSampler:
    """
    Handles dataset loading and stratified sampling for human evaluation.
    
    This class loads GRAID datasets from HuggingFace Hub and provides functionality
    to sample a specified number of questions per question type using deterministic
    seeding based on evaluator name.
    """
    
    def __init__(self, dataset_name: str = "kd7/graid-bdd100k-ground-truth"):
        """
        Initialize the dataset sampler.
        
        Args:
            dataset_name: HuggingFace dataset identifier
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.question_types = []
        
    def load_dataset(self) -> None:
        """
        Load dataset from HuggingFace Hub and discover all question types.
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        logger.info("Performing question type discovery...")
        
        try:
            # Always download full dataset for efficient unique() operation
            self.dataset = load_dataset(self.dataset_name, streaming=False)
            
            # Use train split by default, fallback to first available split
            if 'train' in self.dataset:
                split_data = self.dataset['train']
                split_name = 'train'
            else:
                split_name = list(self.dataset.keys())[0]
                split_data = self.dataset[split_name]
                logger.info(f"Using split '{split_name}' (train not available)")
            
            # Use the efficient unique() method to get ALL question types
            logger.info(f"Using Dataset.unique() method on {split_name} split...")
            unique_types = split_data.unique('question_type')
            self.question_types = sorted(unique_types)
            
            logger.info(f"✅ Discovered {len(self.question_types)} question types using Dataset.unique()")
            logger.info(f"Question types: {self.question_types}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise
    
    def get_question_types(self) -> List[str]:
        """Get list of discovered question types."""
        if not self.question_types:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        return self.question_types.copy()
    
    def sample_questions(self, username: str, n_per_type: int = 10) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Sample n questions per question type using deterministic seeding.
        
        Args:
            username: Evaluator name for deterministic seeding
            n_per_type: Number of questions to sample per type
            
        Returns:
            List of (dataset_index, sample) tuples for evaluation
        """
        if not self.dataset or not self.question_types:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Generate deterministic seed from username
        seed = self._generate_seed(username)
        random.seed(seed)
        
        logger.info(f"Sampling {n_per_type} questions per type for user '{username}' (seed: {seed})")
        
        sampled_questions = []
        
        # Use full dataset for sampling (always available since we download it)
        logger.info("Sampling from full dataset")
        
        # Use train split or first available
        if 'train' in self.dataset:
            split_data = self.dataset['train']
        else:
            split_name = list(self.dataset.keys())[0]
            split_data = self.dataset[split_name]
        
        # Use the unique 'id' field for efficient sampling
        for question_type in self.question_types:
            logger.info(f"Sampling {n_per_type} questions for type '{question_type}'...")
            
            # Filter dataset for this question type
            filtered_data = split_data.filter(
                lambda sample: sample['question_type'] == question_type
            )
            
            if len(filtered_data) == 0:
                logger.warning(f"No questions found for type '{question_type}'")
                continue
            
            # Shuffle and sample n_per_type
            shuffled_data = filtered_data.shuffle(seed=seed + hash(question_type))
            n_to_sample = min(n_per_type, len(shuffled_data))
            sampled_subset = shuffled_data.select(range(n_to_sample))
            
            # Use the unique 'id' field as the identifier
            for sample in sampled_subset:
                record_id = sample.get('id')
                if record_id is None:
                    logger.error(f"Sample missing 'id' field: {sample.keys()}")
                    continue
                sampled_questions.append((record_id, sample))
            
            logger.info(f"Sampled {n_to_sample} questions for type '{question_type}' "
                       f"(from {len(filtered_data)} available)")
        
        # Shuffle the final list to mix question types
        random.shuffle(sampled_questions)
        
        logger.info(f"Total sampled questions: {len(sampled_questions)}")
        return sampled_questions
    
    def _generate_seed(self, username: str) -> int:
        """Generate deterministic seed from username hash."""
        # Create hash of username for reproducible randomization
        hash_obj = hashlib.sha256(username.encode('utf-8'))
        # Use first 8 bytes as integer seed
        seed = int.from_bytes(hash_obj.digest()[:8], byteorder='big')
        # Ensure positive 32-bit integer for random.seed()
        return seed % (2**31)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the loaded dataset."""
        if not self.dataset:
            return {}
        
        # Use train split or first available
        if 'train' in self.dataset:
            split_data = self.dataset['train']
            split_name = 'train'
        else:
            split_name = list(self.dataset.keys())[0]
            split_data = self.dataset[split_name]
        
        # Get dataset info (always full dataset since we download it)
        try:
            total_samples = len(split_data)
            features = list(split_data.features.keys()) if hasattr(split_data, 'features') else []
        except Exception:
            total_samples = "Unknown"
            features = []
        
        return {
            'dataset_name': self.dataset_name,
            'split_used': split_name,
            'total_samples': total_samples,
            'question_types': self.question_types,
            'features': features
        }
