# GRAID Human Evaluation Tool

A Gradio-based interface for human evaluation of GRAID-generated visual question-answering datasets.

## Overview

This tool allows human evaluators to assess the quality of automatically generated VQA pairs by:
- Viewing original images alongside annotated versions with bounding boxes
- Evaluating question validity and answer correctness
- Rating question difficulty
- Providing optional feedback comments
- Tracking evaluation progress with resume capability

## Features

- **Stratified Sampling**: Automatically samples n questions per question type for balanced evaluation
- **Deterministic Seeding**: Uses evaluator name hash for reproducible question selection
- **Visual Comparison**: Side-by-side display of original and annotated images
- **Progress Tracking**: Resume evaluation sessions across browser restarts
- **Multiple Export Formats**: Download results as JSONL, CSV, or JSON
- **Time Tracking**: Automatically records time spent per question
- **Difficulty Rating**: 5-point Likert scale for question difficulty assessment

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have access to the GRAID dataset on HuggingFace Hub (default: `kd7/graid-bdd100k-ground-truth`)

## Usage

### Basic Usage

Run the evaluation interface:
```bash
python gradio_eval_app.py
```

Then open your browser to `http://localhost:7860`

### Custom Dataset

To evaluate a different dataset, modify the `dataset_name` in `gradio_eval_app.py`:
```python
dataset_name = "your-username/your-dataset-name"
```

### Evaluation Workflow

1. **Setup**: Enter your name and specify questions per type (default: 10)
2. **Evaluate**: For each question:
   - View original image (left) and annotated image (right)
   - Read the question and provided answer
   - Answer: "Is the question valid/meaningful?" (Yes/No/Unclear)
   - Answer: "Given highlighted objects, is the answer correct?" (Yes/No/Unclear)
   - Rate difficulty (1=Very Easy, 5=Very Hard)
   - Add optional comments
   - Submit or skip
3. **Download**: Get your evaluation results file when complete

### Resume Sessions

The tool automatically saves progress. If you close the browser and return later:
- Enter the same name
- Click "Start Evaluation"
- You'll be prompted to resume from where you left off

## File Structure

```
human_evals/
├── gradio_eval_app.py      # Main Gradio application
├── dataset_sampler.py      # Dataset loading and sampling
├── visualization_utils.py  # Image visualization with bounding boxes
├── eval_state_manager.py   # Session state and progress tracking
├── data_export.py         # Results export and analysis
├── requirements.txt       # Python dependencies
└── README.md             # This file

# Generated during evaluation:
eval_sessions/             # Session state files
├── eval_session_username.json
└── ...

eval_exports/             # Exported results
├── eval_results_username_timestamp.jsonl
└── ...
```

## Output Format

### JSONL Export (default)
Each line contains one evaluation response:
```json
{
  "evaluator": "john_doe",
  "dataset_name": "kd7/graid-bdd100k-ground-truth", 
  "dataset_index": 12345,
  "question_type": "HowMany",
  "question": "How many cars are in the image?",
  "answer": "3",
  "source_id": "bdd100k_image_001.jpg",
  "is_question_valid": "Yes",
  "is_answer_correct": "No", 
  "difficulty_rating": 3,
  "time_spent_seconds": 15.7,
  "comments": "Only 2 cars clearly visible",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### CSV Export
Same data in CSV format with headers for spreadsheet analysis.

### JSON Export
Structured JSON with metadata and responses array.

## Configuration

### Sampling Parameters
- **Questions per type**: Default 10, adjustable in UI (1-50)
- **Random seed**: Generated from evaluator name hash for reproducibility
- **Question types**: Automatically discovered from dataset

### Visual Settings
- **Colors**: Unique color per object category
- **Labels**: Include confidence scores when available
- **Font**: Automatic font selection with fallbacks

## Technical Details

### Dataset Requirements
The tool expects HuggingFace datasets with these columns:
- `image`: PIL Image or compatible format
- `annotations`: List of COCO-style annotation dicts with `bbox`, `category`, `score`
- `question`: Question text string
- `answer`: Answer text string  
- `question_type`: Question category string
- `source_id`: Original image identifier

### State Management
- Session files stored in `./eval_sessions/`
- Browser-based progress tracking
- Automatic cleanup of completed sessions

### Performance
- Lazy image loading for memory efficiency
- Batch processing for large datasets
- Responsive UI with progress indicators

## Troubleshooting

### Common Issues

**Dataset not loading**:
- Check HuggingFace Hub access
- Verify dataset name spelling
- Ensure dataset has required columns

**Images not displaying**:
- Check image format compatibility
- Verify annotation format (COCO-style expected)
- Check browser console for errors

**Session not resuming**:
- Ensure same username spelling
- Check `eval_sessions/` directory permissions
- Clear browser cache if needed

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export GRAID_DEBUG=1
python gradio_eval_app.py
```

## Contributing

To extend the evaluation tool:

1. **Add new evaluation metrics**: Modify response collection in `gradio_eval_app.py`
2. **Custom visualizations**: Extend `visualization_utils.py`
3. **New export formats**: Add methods to `data_export.py`
4. **Analysis tools**: Use `DataExporter.generate_summary_report()` for statistics

## License

This tool is part of the GRAID project. See main project license for details.
