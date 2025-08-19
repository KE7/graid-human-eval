"""
Evaluation state management for GRAID human evaluation.

This module handles saving and loading evaluation progress, allowing evaluators
to resume their work across browser sessions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class EvalStateManager:
    """
    Manages evaluation state and progress for human evaluators.
    
    This class handles saving evaluation responses, tracking progress,
    and enabling resume functionality across browser sessions.
    """
    
    def __init__(self, username: str, base_dir: str = "./eval_sessions"):
        """
        Initialize state manager for a specific evaluator.
        
        Args:
            username: Evaluator identifier
            base_dir: Directory to store evaluation session files
        """
        self.username = username
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create safe filename from username
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).lower()
        self.session_file = self.base_dir / f"eval_session_{safe_username}.json"
        
        # Initialize session data
        self.session_data = {
            'username': username,
            'created_at': datetime.now().isoformat(),
            'dataset_info': {},
            'sampled_questions': [],
            'responses': [],
            'current_index': 0,
            'completed': False
        }
        
        # Load existing session if available
        self._load_session()
    
    def _load_session(self) -> None:
        """Load existing evaluation session from file."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    loaded_data = json.load(f)
                
                # Merge with defaults to handle version changes
                self.session_data.update(loaded_data)
                logger.info(f"Loaded existing session for {self.username} with {len(self.session_data['responses'])} responses")
                
            except Exception as e:
                logger.warning(f"Failed to load session file: {e}. Starting fresh session.")
    
    def _save_session(self) -> None:
        """Save current session state to file."""
        try:
            # Update timestamp
            self.session_data['updated_at'] = datetime.now().isoformat()
            
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def _serialize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize sample data for JSON storage, excluding non-serializable objects.
        
        Args:
            sample: Original sample dictionary from dataset
            
        Returns:
            Serializable dictionary with image and other non-JSON objects removed
        """
        # Create a copy and remove non-serializable fields
        serializable_sample = {}
        
        # Fields to exclude from serialization (images, complex objects)
        exclude_fields = {'image'}
        
        for key, value in sample.items():
            if key not in exclude_fields:
                # Check if value is JSON serializable
                try:
                    import json
                    json.dumps(value)
                    serializable_sample[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values
                    logger.debug(f"Skipping non-serializable field '{key}' of type {type(value)}")
                    continue
        
        return serializable_sample
    
    def initialize_evaluation(self, dataset_info: Dict[str, Any], 
                            sampled_questions: List[Tuple[int, Dict[str, Any]]]) -> None:
        """
        Initialize a new evaluation session.
        
        Args:
            dataset_info: Information about the dataset being evaluated
            sampled_questions: List of (dataset_index, sample) tuples
        """
        # Only initialize if not already done
        if not self.session_data['sampled_questions']:
            self.session_data['dataset_info'] = dataset_info
            self.session_data['sampled_questions'] = [
                {'dataset_index': idx, 'sample': self._serialize_sample(sample)} 
                for idx, sample in sampled_questions
            ]
            self.session_data['current_index'] = 0
            self.session_data['responses'] = []
            self.session_data['completed'] = False
            
            self._save_session()
            logger.info(f"Initialized evaluation session with {len(sampled_questions)} questions")
    
    def save_response(self, question_index: int, response_data: Dict[str, Any]) -> None:
        """
        Save a response for a specific question.
        
        Args:
            question_index: Index in the sampled questions list
            response_data: Dictionary containing the evaluation response
        """
        # Add metadata to response
        full_response = {
            'question_index': question_index,
            'dataset_index': self.session_data['sampled_questions'][question_index]['dataset_index'],
            'timestamp': datetime.now().isoformat(),
            **response_data
        }
        
        # Check if response already exists for this question
        existing_idx = None
        for i, resp in enumerate(self.session_data['responses']):
            if resp['question_index'] == question_index:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # Update existing response
            self.session_data['responses'][existing_idx] = full_response
            logger.debug(f"Updated response for question {question_index}")
        else:
            # Add new response
            self.session_data['responses'].append(full_response)
            logger.debug(f"Saved new response for question {question_index}")
        
        # Update current index to next question
        self.session_data['current_index'] = question_index + 1
        
        # Check if evaluation is complete
        if self.session_data['current_index'] >= len(self.session_data['sampled_questions']):
            self.session_data['completed'] = True
            logger.info(f"Evaluation completed for {self.username}")
        
        self._save_session()
    
    def get_current_question(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """
        Get the current question to be evaluated.
        
        Returns:
            Tuple of (question_index, question_data) or None if completed
        """
        if self.is_completed():
            return None
        
        current_idx = self.session_data['current_index']
        if current_idx < len(self.session_data['sampled_questions']):
            question_data = self.session_data['sampled_questions'][current_idx]
            return current_idx, question_data['sample']
        
        return None
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current evaluation progress.
        
        Returns:
            Dictionary with progress information
        """
        total_questions = len(self.session_data['sampled_questions'])
        completed_responses = len(self.session_data['responses'])
        current_index = self.session_data['current_index']
        
        return {
            'total_questions': total_questions,
            'completed_responses': completed_responses,
            'current_index': current_index,
            'progress_percent': (completed_responses / total_questions * 100) if total_questions > 0 else 0,
            'is_completed': self.is_completed()
        }
    
    def is_completed(self) -> bool:
        """Check if evaluation is completed."""
        return self.session_data.get('completed', False)
    
    def can_resume(self) -> bool:
        """Check if there's an existing session that can be resumed."""
        return (len(self.session_data['sampled_questions']) > 0 and 
                not self.is_completed())
    
    def get_responses_for_export(self) -> List[Dict[str, Any]]:
        """
        Get all responses formatted for export.
        
        Returns:
            List of response dictionaries ready for export
        """
        export_responses = []
        
        for response in self.session_data['responses']:
            # Get the original question data
            question_idx = response['question_index']
            if question_idx < len(self.session_data['sampled_questions']):
                question_data = self.session_data['sampled_questions'][question_idx]['sample']
                
                export_response = {
                    'evaluator': self.username,
                    'dataset_name': self.session_data['dataset_info'].get('dataset_name', 'unknown'),
                    'dataset_index': response['dataset_index'],
                    'question_type': question_data.get('question_type', 'unknown'),
                    'question': question_data.get('question', ''),
                    'answer': question_data.get('answer', ''),
                    'source_id': question_data.get('source_id', ''),
                    'is_question_valid': response.get('is_question_valid', ''),
                    'is_answer_correct': response.get('is_answer_correct', ''),
                    'difficulty_rating': response.get('difficulty_rating', ''),
                    'time_spent_seconds': response.get('time_spent_seconds', 0),
                    'comments': response.get('comments', ''),
                    'timestamp': response['timestamp']
                }
                export_responses.append(export_response)
        
        return export_responses
    
    def export_to_file(self, format: str = 'jsonl') -> str:
        """
        Export responses to file.
        
        Args:
            format: Export format ('jsonl' or 'csv')
            
        Returns:
            Path to the exported file
        """
        responses = self.get_responses_for_export()
        
        # Create export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_username = "".join(c for c in self.username if c.isalnum() or c in ('-', '_')).lower()
        
        if format.lower() == 'csv':
            import csv
            export_file = self.base_dir / f"eval_results_{safe_username}_{timestamp}.csv"
            
            if responses:
                with open(export_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=responses[0].keys())
                    writer.writeheader()
                    writer.writerows(responses)
        else:
            # Default to JSONL
            export_file = self.base_dir / f"eval_results_{safe_username}_{timestamp}.jsonl"
            
            with open(export_file, 'w', encoding='utf-8') as f:
                for response in responses:
                    f.write(json.dumps(response) + '\n')
        
        logger.info(f"Exported {len(responses)} responses to {export_file}")
        return str(export_file)
    
    def reset_session(self) -> None:
        """Reset the evaluation session (start over)."""
        if self.session_file.exists():
            # Backup current session before reset
            backup_file = self.session_file.with_suffix('.backup.json')
            self.session_file.rename(backup_file)
            logger.info(f"Backed up session to {backup_file}")
        
        # Reset session data
        self.session_data = {
            'username': self.username,
            'created_at': datetime.now().isoformat(),
            'dataset_info': {},
            'sampled_questions': [],
            'responses': [],
            'current_index': 0,
            'completed': False
        }
        
        logger.info(f"Reset evaluation session for {self.username}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        progress = self.get_progress()
        
        # Calculate question type distribution
        question_type_counts = {}
        for q_data in self.session_data['sampled_questions']:
            qtype = q_data['sample'].get('question_type', 'unknown')
            question_type_counts[qtype] = question_type_counts.get(qtype, 0) + 1
        
        return {
            'username': self.username,
            'dataset_name': self.session_data['dataset_info'].get('dataset_name', 'unknown'),
            'total_questions': progress['total_questions'],
            'completed_responses': progress['completed_responses'],
            'progress_percent': progress['progress_percent'],
            'is_completed': progress['is_completed'],
            'question_type_distribution': question_type_counts,
            'created_at': self.session_data.get('created_at', ''),
            'updated_at': self.session_data.get('updated_at', '')
        }
