"""
Data export utilities for GRAID human evaluation results.

This module provides functionality to export evaluation results in various formats
and generate summary statistics from collected human evaluations.
"""

import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Handles export and analysis of human evaluation results.
    
    This class provides functionality to export evaluation data in multiple formats
    and generate summary statistics for analysis.
    """
    
    def __init__(self, output_dir: str = "./eval_exports"):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_responses(self, responses: List[Dict[str, Any]], 
                        username: str, format: str = 'jsonl') -> str:
        """
        Export evaluation responses to file.
        
        Args:
            responses: List of response dictionaries
            username: Evaluator identifier
            format: Export format ('jsonl', 'csv', or 'json')
            
        Returns:
            Path to exported file
        """
        if not responses:
            logger.warning("No responses to export")
            return ""
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).lower()
        
        if format.lower() == 'csv':
            return self._export_csv(responses, safe_username, timestamp)
        elif format.lower() == 'json':
            return self._export_json(responses, safe_username, timestamp)
        else:
            return self._export_jsonl(responses, safe_username, timestamp)
    
    def _export_jsonl(self, responses: List[Dict[str, Any]], 
                     username: str, timestamp: str) -> str:
        """Export responses as JSONL format."""
        export_file = self.output_dir / f"eval_results_{username}_{timestamp}.jsonl"
        
        with open(export_file, 'w', encoding='utf-8') as f:
            for response in responses:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(responses)} responses to JSONL: {export_file}")
        return str(export_file)
    
    def _export_csv(self, responses: List[Dict[str, Any]], 
                   username: str, timestamp: str) -> str:
        """Export responses as CSV format."""
        export_file = self.output_dir / f"eval_results_{username}_{timestamp}.csv"
        
        if responses:
            fieldnames = responses[0].keys()
            with open(export_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(responses)
        
        logger.info(f"Exported {len(responses)} responses to CSV: {export_file}")
        return str(export_file)
    
    def _export_json(self, responses: List[Dict[str, Any]], 
                    username: str, timestamp: str) -> str:
        """Export responses as JSON format."""
        export_file = self.output_dir / f"eval_results_{username}_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'evaluator': username,
                'export_timestamp': datetime.now().isoformat(),
                'total_responses': len(responses)
            },
            'responses': responses
        }
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(responses)} responses to JSON: {export_file}")
        return str(export_file)
    
    def generate_summary_report(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from evaluation responses.
        
        Args:
            responses: List of response dictionaries
            
        Returns:
            Dictionary containing summary statistics
        """
        if not responses:
            return {'error': 'No responses to analyze'}
        
        # Basic counts
        total_responses = len(responses)
        
        # Question validity analysis
        validity_counts = Counter(r.get('is_question_valid', 'Unknown') for r in responses)
        validity_rate = validity_counts.get('Yes', 0) / total_responses if total_responses > 0 else 0
        
        # Answer correctness analysis
        correctness_counts = Counter(r.get('is_answer_correct', 'Unknown') for r in responses)
        correctness_rate = correctness_counts.get('Yes', 0) / total_responses if total_responses > 0 else 0
        
        # Difficulty analysis
        difficulty_ratings = [r.get('difficulty_rating', 0) for r in responses if r.get('difficulty_rating')]
        avg_difficulty = sum(difficulty_ratings) / len(difficulty_ratings) if difficulty_ratings else 0
        
        # Time analysis
        time_spent = [r.get('time_spent_seconds', 0) for r in responses if r.get('time_spent_seconds')]
        avg_time = sum(time_spent) / len(time_spent) if time_spent else 0
        
        # Question type analysis
        question_type_stats = defaultdict(lambda: {
            'count': 0,
            'valid_questions': 0,
            'correct_answers': 0,
            'avg_difficulty': 0,
            'avg_time': 0
        })
        
        for response in responses:
            qtype = response.get('question_type', 'Unknown')
            stats = question_type_stats[qtype]
            
            stats['count'] += 1
            
            if response.get('is_question_valid') == 'Yes':
                stats['valid_questions'] += 1
            
            if response.get('is_answer_correct') == 'Yes':
                stats['correct_answers'] += 1
            
            if response.get('difficulty_rating'):
                stats['avg_difficulty'] += response['difficulty_rating']
            
            if response.get('time_spent_seconds'):
                stats['avg_time'] += response['time_spent_seconds']
        
        # Calculate averages for question types
        for qtype, stats in question_type_stats.items():
            if stats['count'] > 0:
                stats['validity_rate'] = stats['valid_questions'] / stats['count']
                stats['correctness_rate'] = stats['correct_answers'] / stats['count']
                stats['avg_difficulty'] = stats['avg_difficulty'] / stats['count']
                stats['avg_time'] = stats['avg_time'] / stats['count']
        
        # Dataset analysis
        dataset_names = set(r.get('dataset_name', 'Unknown') for r in responses)
        evaluators = set(r.get('evaluator', 'Unknown') for r in responses)
        
        summary = {
            'overview': {
                'total_responses': total_responses,
                'datasets_evaluated': list(dataset_names),
                'evaluators': list(evaluators),
                'evaluation_period': {
                    'start': min(r.get('timestamp', '') for r in responses if r.get('timestamp')),
                    'end': max(r.get('timestamp', '') for r in responses if r.get('timestamp'))
                }
            },
            'quality_metrics': {
                'question_validity': {
                    'rate': validity_rate,
                    'counts': dict(validity_counts)
                },
                'answer_correctness': {
                    'rate': correctness_rate,
                    'counts': dict(correctness_counts)
                },
                'average_difficulty': avg_difficulty,
                'average_time_seconds': avg_time
            },
            'question_type_analysis': dict(question_type_stats),
            'recommendations': self._generate_recommendations(question_type_stats, validity_rate, correctness_rate)
        }
        
        return summary
    
    def _generate_recommendations(self, question_type_stats: Dict[str, Dict], 
                                validity_rate: float, correctness_rate: float) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Overall quality recommendations
        if validity_rate < 0.8:
            recommendations.append(f"Low question validity rate ({validity_rate:.1%}). Review question generation logic.")
        
        if correctness_rate < 0.8:
            recommendations.append(f"Low answer correctness rate ({correctness_rate:.1%}). Review answer generation or annotation quality.")
        
        # Question type specific recommendations
        for qtype, stats in question_type_stats.items():
            if stats['validity_rate'] < 0.7:
                recommendations.append(f"Question type '{qtype}' has low validity rate ({stats['validity_rate']:.1%})")
            
            if stats['correctness_rate'] < 0.7:
                recommendations.append(f"Question type '{qtype}' has low correctness rate ({stats['correctness_rate']:.1%})")
            
            if stats['avg_difficulty'] > 4.0:
                recommendations.append(f"Question type '{qtype}' is rated as very difficult (avg: {stats['avg_difficulty']:.1f}/5)")
        
        if not recommendations:
            recommendations.append("Overall evaluation results look good! No major issues identified.")
        
        return recommendations
    
    def export_summary_report(self, responses: List[Dict[str, Any]], 
                            filename: Optional[str] = None) -> str:
        """
        Export summary report to JSON file.
        
        Args:
            responses: List of response dictionaries
            filename: Optional custom filename
            
        Returns:
            Path to exported summary file
        """
        summary = self.generate_summary_report(responses)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_summary_{timestamp}.json"
        
        summary_file = self.output_dir / filename
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported summary report to: {summary_file}")
        return str(summary_file)
    
    def combine_multiple_evaluations(self, eval_files: List[str]) -> List[Dict[str, Any]]:
        """
        Combine multiple evaluation files into a single dataset.
        
        Args:
            eval_files: List of paths to evaluation files (JSONL or JSON)
            
        Returns:
            Combined list of all responses
        """
        combined_responses = []
        
        for file_path in eval_files:
            try:
                file_path_obj = Path(file_path)
                
                if file_path_obj.suffix == '.jsonl':
                    with open(file_path_obj, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                combined_responses.append(json.loads(line))
                
                elif file_path_obj.suffix == '.json':
                    with open(file_path_obj, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            combined_responses.extend(data)
                        elif isinstance(data, dict) and 'responses' in data:
                            combined_responses.extend(data['responses'])
                
                logger.info(f"Loaded {len(combined_responses)} responses from {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to load evaluation file {file_path}: {e}")
        
        logger.info(f"Combined {len(combined_responses)} total responses from {len(eval_files)} files")
        return combined_responses
