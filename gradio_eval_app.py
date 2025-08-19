"""
GRAID Human Evaluation Gradio Application

This is the main Gradio interface for human evaluation of GRAID-generated datasets.
Evaluators can assess question validity, answer correctness, and difficulty ratings
for sampled questions across all question types.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

import gradio as gr

from dataset_sampler import DatasetSampler
from visualization_utils import VisualizationUtils
from eval_state_manager import EvalStateManager
from data_export import DataExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraidEvaluationApp:
    """
    Main Gradio application for GRAID dataset human evaluation.
    
    This class orchestrates the evaluation workflow, managing user sessions,
    question presentation, response collection, and data export.
    """
    
    def __init__(self, dataset_name: str = "kd7/graid-bdd100k-ground-truth"):
        """Initialize the evaluation application."""
        self.dataset_name = dataset_name
        self.dataset_sampler = DatasetSampler(dataset_name)
        self.visualization_utils = VisualizationUtils()
        self.data_exporter = DataExporter()
        
        # Session state
        self.current_session: Optional[EvalStateManager] = None
        self.current_question_start_time: Optional[float] = None
        
        logger.info(f"Initialized GRAID Evaluation App for dataset: {dataset_name}")
    
    def setup_evaluation(self, username: str, n_per_type: int, progress_callback=None) -> Tuple[str, bool, Dict]:
        """
        Setup a new evaluation session or resume existing one.
        
        Args:
            username: Evaluator name
            n_per_type: Number of questions per question type
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (status_message, show_eval_interface, initial_state)
        """
        try:
            if not username.strip():
                return "Please enter your name to continue.", False, {}
            
            username = username.strip()
            logger.info(f"Setting up evaluation for user: {username}")
            
            # Initialize session manager
            self.current_session = EvalStateManager(username)
            
            # Check if user can resume existing session
            if self.current_session.can_resume():
                progress = self.current_session.get_progress()
                resume_msg = (f"Welcome back, {username}! You have an existing session with "
                            f"{progress['completed_responses']}/{progress['total_questions']} questions completed. "
                            f"Continuing from where you left off...")
                
                # Get current question for display
                current_q = self.current_session.get_current_question()
                if current_q:
                    question_idx, question_data = current_q
                    return resume_msg, True, self._prepare_question_display(question_idx, question_data)
                else:
                    return "Session completed! Use the download button below.", False, {}
            
            # Update progress: Loading dataset
            if progress_callback:
                progress_callback("🔄 Loading dataset from HuggingFace Hub...\n\n"
                                "⏳ **First-time setup**: Downloading the full dataset... "
                                "📥 This may take several minutes for the initial download...")
            
            # Load full dataset for definitive question type discovery
            self.dataset_sampler.load_dataset()
            
            # Update progress: Sampling questions
            if progress_callback:
                progress_callback("✅ Dataset loaded successfully!\n\n"
                                "🎯 Now sampling questions for your evaluation...")
            
            sampled_questions = self.dataset_sampler.sample_questions(username, n_per_type)
            
            if not sampled_questions:
                return "No questions could be sampled from the dataset.", False, {}
            
            # Initialize new session
            dataset_info = self.dataset_sampler.get_dataset_info()
            self.current_session.initialize_evaluation(dataset_info, sampled_questions)
            
            # Get first question
            current_q = self.current_session.get_current_question()
            if current_q:
                question_idx, question_data = current_q
                
                total_questions = len(sampled_questions)
                question_types = self.dataset_sampler.get_question_types()
                
                setup_msg = (f"🎉 Welcome, {username}! Evaluation setup complete.\n\n"
                           f"📊 **Evaluation Details:**\n"
                           f"- Total questions: {total_questions}\n"
                           f"- Question types: {len(question_types)}\n"
                           f"- Questions per type: ~{n_per_type}\n"
                           f"- Dataset: {dataset_info.get('dataset_name', 'Unknown')}\n\n"
                           f"🚀 Let's begin with the first question!")
                
                return setup_msg, True, self._prepare_question_display(question_idx, question_data)
            
            return "Failed to get first question.", False, {}
            
        except Exception as e:
            logger.error(f"Failed to setup evaluation: {e}")
            return f"❌ Error setting up evaluation: {str(e)}", False, {}
    
    def _get_complete_question_data(self, question_idx: int, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get complete question data including image from dataset.
        
        The session state stores question data without images to avoid JSON serialization issues.
        This method retrieves the complete sample including the image from the dataset.
        """
        try:
            # Get the record ID for this question
            sampled_questions = self.current_session.session_data['sampled_questions']
            record_id = sampled_questions[question_idx]['dataset_index']
            
            # Get the complete sample from dataset (including image) using the unique ID
            if 'train' in self.dataset_sampler.dataset:
                split_data = self.dataset_sampler.dataset['train']
            else:
                split_name = list(self.dataset_sampler.dataset.keys())[0]
                split_data = self.dataset_sampler.dataset[split_name]
            
            # If record_id is numeric (old approach), use direct indexing
            if isinstance(record_id, int) and 'id' not in split_data.column_names:
                complete_sample = split_data[record_id]
            else:
                # Use the unique 'id' field to find the record
                # Since id corresponds to row number, we can use direct indexing if it's numeric
                if isinstance(record_id, int):
                    complete_sample = split_data[record_id]
                else:
                    # Filter by id field if it's not a simple row number
                    filtered_samples = split_data.filter(
                        lambda sample: sample.get('id') == record_id
                    )
                    if len(filtered_samples) > 0:
                        complete_sample = filtered_samples[0]
                    else:
                        logger.warning(f"Could not find sample with id: {record_id}")
                        return question_data
            
            return complete_sample
            
        except Exception as e:
            logger.error(f"Failed to retrieve complete question data: {e}")
            # Fallback to stored data (without image)
            return question_data
    
    def _prepare_question_display(self, question_idx: int, question_data: Dict[str, Any]) -> Dict:
        """Prepare question data for display in the interface."""
        try:
            # Record start time for this question
            self.current_question_start_time = time.time()
            
            # Get the complete sample data with image from dataset
            complete_question_data = self._get_complete_question_data(question_idx, question_data)
            
            # Create visualizations
            original_img, annotated_img = self.visualization_utils.create_side_by_side_display(complete_question_data)
            
            # Get progress info
            progress = self.current_session.get_progress()
            progress_text = (f"**Question {progress['current_index'] + 1} of {progress['total_questions']}** "
                           f"({progress['progress_percent']:.1f}% complete)")
            
            # Get annotation summary
            annotation_summary = self.visualization_utils.get_annotation_summary(
                complete_question_data.get('annotations', [])
            )
            
            # Format question and answer display
            question_text = f"**Question:** {complete_question_data.get('question', 'N/A')}"
            answer_text = f"**Answer:** {complete_question_data.get('answer', 'N/A')}"
            question_type_text = f"**Question Type:** {complete_question_data.get('question_type', 'N/A')}"
            
            return {
                'question_idx': question_idx,
                'original_image': original_img,
                'annotated_image': annotated_img,
                'progress_text': progress_text,
                'question_text': question_text,
                'answer_text': answer_text,
                'question_type_text': question_type_text,
                'annotation_summary': f"**Detected Objects:** {annotation_summary}",
                'show_eval_interface': True
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare question display: {e}")
            return {'show_eval_interface': False}
    
    def submit_response(self, question_idx: int, is_question_valid: str, 
                       is_answer_correct: str, difficulty_rating: int, 
                       comments: str) -> Tuple[str, bool, Dict]:
        """
        Submit evaluation response and move to next question.
        
        Args:
            question_idx: Current question index
            is_question_valid: Question validity response
            is_answer_correct: Answer correctness response
            difficulty_rating: Difficulty rating (1-5)
            comments: Optional comments
            
        Returns:
            Tuple of (status_message, show_eval_interface, next_question_data)
        """
        try:
            if not self.current_session:
                return "No active session. Please restart the evaluation.", False, {}
            
            # Validate inputs
            if not is_question_valid or not is_answer_correct:
                return "Please answer both evaluation questions before submitting.", True, {}
            
            if difficulty_rating < 1 or difficulty_rating > 5:
                return "Please select a difficulty rating between 1 and 5.", True, {}
            
            # Calculate time spent
            time_spent = 0
            if self.current_question_start_time:
                time_spent = time.time() - self.current_question_start_time
            
            # Prepare response data
            response_data = {
                'is_question_valid': is_question_valid,
                'is_answer_correct': is_answer_correct,
                'difficulty_rating': difficulty_rating,
                'time_spent_seconds': round(time_spent, 2),
                'comments': comments.strip()
            }
            
            # Save response
            self.current_session.save_response(question_idx, response_data)
            
            logger.info(f"Saved response for question {question_idx}: "
                       f"valid={is_question_valid}, correct={is_answer_correct}, "
                       f"difficulty={difficulty_rating}, time={time_spent:.1f}s")
            
            # Check if evaluation is complete
            if self.current_session.is_completed():
                progress = self.current_session.get_progress()
                completion_msg = (f"🎉 **Evaluation Complete!** 🎉\n\n"
                                f"Thank you for evaluating {progress['total_questions']} questions!\n\n"
                                f"Your responses have been saved. Use the download button below to "
                                f"get your evaluation results file.")
                
                return completion_msg, False, {'show_download': True}
            
            # Get next question
            next_q = self.current_session.get_current_question()
            if next_q:
                next_question_idx, next_question_data = next_q
                next_display = self._prepare_question_display(next_question_idx, next_question_data)
                
                return "Response saved! Moving to next question...", True, next_display
            else:
                return "No more questions available.", False, {}
                
        except Exception as e:
            logger.error(f"Failed to submit response: {e}")
            return f"Error submitting response: {str(e)}", True, {}
    
    def skip_question(self, question_idx: int) -> Tuple[str, bool, Dict]:
        """Skip current question and move to next."""
        try:
            if not self.current_session:
                return "No active session. Please restart the evaluation.", False, {}
            
            logger.info(f"Skipping question {question_idx}")
            
            # Save empty response to maintain progress
            response_data = {
                'is_question_valid': 'Skipped',
                'is_answer_correct': 'Skipped',
                'difficulty_rating': 0,
                'time_spent_seconds': 0,
                'comments': 'Question skipped by evaluator'
            }
            
            self.current_session.save_response(question_idx, response_data)
            
            # Get next question
            next_q = self.current_session.get_current_question()
            if next_q:
                next_question_idx, next_question_data = next_q
                next_display = self._prepare_question_display(next_question_idx, next_question_data)
                return "Question skipped. Moving to next question...", True, next_display
            else:
                return "No more questions available.", False, {}
                
        except Exception as e:
            logger.error(f"Failed to skip question: {e}")
            return f"Error skipping question: {str(e)}", True, {}
    
    def download_results(self, format: str = 'jsonl') -> str:
        """Generate download file for evaluation results."""
        try:
            if not self.current_session:
                return "No active session to download results from."
            
            # Export results
            export_file = self.current_session.export_to_file(format)
            logger.info(f"Generated download file: {export_file}")
            
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to generate download file: {e}")
            return ""
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        with gr.Blocks(title="GRAID Dataset Human Evaluation", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🧠 GRAID Dataset Human Evaluation
            
            Welcome to the GRAID dataset evaluation interface! This tool helps assess the quality 
            of automatically generated visual question-answering pairs.
            
            **Instructions:**
            1. Enter your name and specify how many questions per type you'd like to evaluate
            2. For each question, you'll see the original image and the same image with detected objects highlighted
            3. Evaluate whether the question makes sense and if the answer is correct given the highlighted objects
            4. Rate the difficulty and optionally add comments
            5. Download your results when complete
            """)
            
            # Setup Section
            with gr.Row():
                with gr.Column(scale=2):
                    username_input = gr.Textbox(
                        label="Your Name",
                        placeholder="Enter your name for session tracking",
                        value=""
                    )
                with gr.Column(scale=1):
                    n_questions_input = gr.Number(
                        label="Questions per Type",
                        value=10,
                        minimum=1,
                        maximum=50,
                        step=1
                    )
                with gr.Column(scale=1):
                    start_button = gr.Button("Start Evaluation", variant="primary", size="lg")
            
            setup_status = gr.Markdown("", visible=True)
            
            # Evaluation Interface (initially hidden)
            with gr.Column(visible=False) as eval_interface:
                progress_display = gr.Markdown("", visible=True)
                
                # Images side by side
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Original Image")
                        original_image = gr.Image(label="Original", interactive=False, height=400)
                    with gr.Column():
                        gr.Markdown("### Image with Detected Objects")
                        annotated_image = gr.Image(label="With Annotations", interactive=False, height=400)
                
                # Question and Answer Display
                with gr.Row():
                    with gr.Column():
                        question_display = gr.Markdown("", visible=True)
                        answer_display = gr.Markdown("", visible=True)
                        question_type_display = gr.Markdown("", visible=True)
                        annotation_summary_display = gr.Markdown("", visible=True)
                
                # Evaluation Questions
                gr.Markdown("## 📝 Evaluation Questions")
                
                with gr.Row():
                    with gr.Column():
                        question_valid_radio = gr.Radio(
                            choices=["Yes", "No", "Unclear"],
                            label="1. Is the question valid?",
                            value=None
                        )
                    with gr.Column():
                        answer_correct_radio = gr.Radio(
                            choices=["Yes", "No", "Unclear"],
                            label="2. Given the highlighted objects, is the answer correct?",
                            value=None
                        )
                
                difficulty_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    label="3. How difficult did you find this question? (1=Very Easy, 5=Very Hard)",
                    value=3
                )
                
                comments_textbox = gr.Textbox(
                    label="Comments (Optional)",
                    placeholder="Any additional observations or feedback...",
                    lines=3
                )
                
                # Action Buttons
                with gr.Row():
                    submit_button = gr.Button("Submit & Next Question", variant="primary", size="lg")
                    skip_button = gr.Button("Skip Question", variant="secondary")
            
            # Completion Section (initially hidden)
            with gr.Column(visible=False) as completion_interface:
                gr.Markdown("🎉 **Evaluation Complete!**\n\nThank you for your evaluation! Use the download button below to get your results.", visible=True)
                
                with gr.Row():
                    download_format = gr.Radio(
                        choices=["jsonl", "csv", "json"],
                        label="Download Format",
                        value="jsonl"
                    )
                    download_button = gr.Button("Download Results", variant="primary", size="lg")
                
                download_file = gr.File(label="Download your evaluation results", visible=False)
            
            # Hidden state variables
            current_question_idx = gr.State(0)
            session_active = gr.State(False)
            
            # Event handlers
            def handle_start_evaluation(username, n_per_type):
                """Handle evaluation setup with loading progress."""
                
                # Show loading message immediately
                loading_msg = ("🔄 **Setting up your evaluation...**\n\n"
                             "Please wait while we:\n"
                             "1. Load the dataset from HuggingFace Hub\n"
                             "2. Discover question types\n"
                             "3. Sample questions for your evaluation\n\n"
                             "⏳ This may take a few minutes for first-time setup and unforunately we don't have a progress bar...")
                
                # Return loading state first
                yield (
                    loading_msg,  # setup_status
                    gr.update(visible=False),  # eval setup row (hide during loading)
                    gr.update(visible=False),  # eval_interface
                    gr.update(visible=False),  # completion_interface
                    "", None, None, "", "", "", "", 0, False, None, None, 3, ""
                )
                
                # Now actually setup the evaluation
                status_msg, show_eval, question_data = self.setup_evaluation(username, n_per_type)
                
                if show_eval and question_data:
                    return (
                        status_msg,  # setup_status
                        gr.update(visible=False),  # eval setup row
                        gr.update(visible=True),   # eval_interface
                        gr.update(visible=False),  # completion_interface
                        question_data.get('progress_text', ''),  # progress_display
                        question_data.get('original_image'),     # original_image
                        question_data.get('annotated_image'),    # annotated_image
                        question_data.get('question_text', ''),  # question_display
                        question_data.get('answer_text', ''),    # answer_display
                        question_data.get('question_type_text', ''),  # question_type_display
                        question_data.get('annotation_summary', ''), # annotation_summary_display
                        question_data.get('question_idx', 0),    # current_question_idx
                        True,  # session_active
                        None,  # question_valid_radio (reset)
                        None,  # answer_correct_radio (reset)
                        3,     # difficulty_slider (reset)
                        ""     # comments_textbox (reset)
                    )
                else:
                    return (
                        status_msg,  # setup_status
                        gr.update(visible=True),   # eval setup row
                        gr.update(visible=False),  # eval_interface
                        gr.update(visible=False),  # completion_interface
                        "", None, None, "", "", "", "", 0, False, None, None, 3, ""
                    )
            
            def handle_submit_response(question_idx, is_valid, is_correct, difficulty, comments):
                """Handle response submission."""
                status_msg, show_eval, next_data = self.submit_response(
                    question_idx, is_valid, is_correct, difficulty, comments
                )
                
                if show_eval and next_data:
                    # Continue with next question
                    return (
                        status_msg,  # setup_status
                        gr.update(visible=True),   # eval_interface
                        gr.update(visible=False),  # completion_interface
                        next_data.get('progress_text', ''),     # progress_display
                        next_data.get('original_image'),        # original_image
                        next_data.get('annotated_image'),       # annotated_image
                        next_data.get('question_text', ''),     # question_display
                        next_data.get('answer_text', ''),       # answer_display
                        next_data.get('question_type_text', ''), # question_type_display
                        next_data.get('annotation_summary', ''), # annotation_summary_display
                        next_data.get('question_idx', 0),       # current_question_idx
                        None,  # question_valid_radio (reset)
                        None,  # answer_correct_radio (reset)
                        3,     # difficulty_slider (reset)
                        ""     # comments_textbox (reset)
                    )
                else:
                    # Evaluation complete
                    return (
                        status_msg,  # setup_status
                        gr.update(visible=False),  # eval_interface
                        gr.update(visible=True),   # completion_interface
                        "", None, None, "", "", "", "", 0, None, None, 3, ""
                    )
            
            def handle_skip_question(question_idx):
                """Handle question skipping."""
                status_msg, show_eval, next_data = self.skip_question(question_idx)
                
                if show_eval and next_data:
                    return (
                        status_msg,  # setup_status
                        gr.update(visible=True),   # eval_interface
                        next_data.get('progress_text', ''),     # progress_display
                        next_data.get('original_image'),        # original_image
                        next_data.get('annotated_image'),       # annotated_image
                        next_data.get('question_text', ''),     # question_display
                        next_data.get('answer_text', ''),       # answer_display
                        next_data.get('question_type_text', ''), # question_type_display
                        next_data.get('annotation_summary', ''), # annotation_summary_display
                        next_data.get('question_idx', 0),       # current_question_idx
                        None,  # question_valid_radio (reset)
                        None,  # answer_correct_radio (reset)
                        3,     # difficulty_slider (reset)
                        ""     # comments_textbox (reset)
                    )
                else:
                    return (
                        status_msg,  # setup_status
                        gr.update(visible=False),  # eval_interface
                        "", None, None, "", "", "", "", 0, None, None, 3, ""
                    )
            
            def handle_download(format):
                """Handle results download."""
                file_path = self.download_results(format)
                if file_path:
                    return gr.update(value=file_path, visible=True)
                else:
                    return gr.update(visible=False)
            
            # Wire up event handlers
            start_button.click(
                fn=handle_start_evaluation,
                inputs=[username_input, n_questions_input],
                outputs=[
                    setup_status, gr.Row(visible=True), eval_interface, completion_interface,
                    progress_display, original_image, annotated_image,
                    question_display, answer_display, question_type_display, annotation_summary_display,
                    current_question_idx, session_active,
                    question_valid_radio, answer_correct_radio, difficulty_slider, comments_textbox
                ],
                show_progress=True
            )
            
            submit_button.click(
                fn=handle_submit_response,
                inputs=[current_question_idx, question_valid_radio, answer_correct_radio, 
                       difficulty_slider, comments_textbox],
                outputs=[
                    setup_status, eval_interface, completion_interface,
                    progress_display, original_image, annotated_image,
                    question_display, answer_display, question_type_display, annotation_summary_display,
                    current_question_idx,
                    question_valid_radio, answer_correct_radio, difficulty_slider, comments_textbox
                ]
            )
            
            skip_button.click(
                fn=handle_skip_question,
                inputs=[current_question_idx],
                outputs=[
                    setup_status, eval_interface,
                    progress_display, original_image, annotated_image,
                    question_display, answer_display, question_type_display, annotation_summary_display,
                    current_question_idx,
                    question_valid_radio, answer_correct_radio, difficulty_slider, comments_textbox
                ]
            )
            
            download_button.click(
                fn=handle_download,
                inputs=[download_format],
                outputs=[download_file]
            )
        
        return app


def main():
    """Main entry point for the evaluation application."""
    # You can customize the dataset name here
    dataset_name = "kd7/graid-bdd100k-ground-truth"
    
    app_instance = GraidEvaluationApp(dataset_name)
    interface = app_instance.create_interface()
    
    # Launch the app
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True              # Enable debug mode
    )


if __name__ == "__main__":
    main()
