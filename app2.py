import streamlit as st
import pandas as pd
from typing import Dict, List
import time
from student_evaluator_v3 import StudentAnswerEvaluator


# Page configuration
st.set_page_config(
    page_title="Student Answer Evaluation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .excellent { border-left: 5px solid #28a745; }
    .good { border-left: 5px solid #ffc107; }
    .acceptable { border-left: 5px solid #fd7e14; }
    .poor { border-left: 5px solid #dc3545; }

    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'evaluator' not in st.session_state:
        with st.spinner('Initializing evaluation system...'):
            st.session_state.evaluator = StudentAnswerEvaluator()

    if 'evaluation_history' not in st.session_state:
        st.session_state.evaluation_history = []


def get_grade_info(score: float) -> tuple:
    """Get grade and color based on score"""
    if score >= 0.8:
        return "Excellent", "#28a745", "excellent"
    elif score >= 0.6:
        return "Good", "#ffc107", "good"
    elif score >= 0.4:
        return "Acceptable", "#fd7e14", "acceptable"
    else:
        return "Poor", "#dc3545", "poor"


def create_score_visualization(scores: Dict[str, float], question_type: str):
    """Create visualization for scores"""
    if question_type == "short_answer":
        if st.session_state.evaluator.bert_available:
            score_names = ['BERT Similarity', 'Confidence', 'Length Ratio']
            score_values = [
                scores.get('bert_similarity', 0),
                scores.get('confidence', 0),
                scores.get('length_ratio', 0)
            ]
        else:
            score_names = ['Word Overlap', 'TF-IDF Similarity', 'Sequence Similarity']
            score_values = [
                scores.get('word_overlap', 0),
                scores.get('tfidf_similarity', 0),
                scores.get('sequence_similarity', 0)
            ]
    elif question_type == "essay":
        score_names = ['Overall Similarity', 'Average Sentence Match', 'Coverage Score', 'Content Depth']
        score_values = [
            scores.get('overall_similarity', 0),
            scores.get('average_sentence_match', 0),
            scores.get('coverage_score', 0),
            scores.get('content_depth', 0)
        ]
    else:  # MCQ
        score_names = ['Exact Match', 'Similarity']
        score_values = [scores.get('exact_match', 0), scores.get('similarity', 0)]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=score_values,
        theta=score_names,
        fill='toself',
        name='Scores',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Score Breakdown",
        height=400
    )

    return fig


def evaluate_single_answer():
    """Single answer evaluation interface"""
    st.subheader("Single Answer Evaluation")

    # Question type selection
    question_type = st.selectbox(
        "Select Question Type:",
        ["Short Answer", "Essay", "Multiple Choice"],
        help="Choose the type of question you want to evaluate"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model/Correct Answer:**")
        model_answer = st.text_area(
            "Enter the model/correct answer:",
            height=150,
            placeholder="Type the correct answer here...",
            key="model_answer"
        )

    with col2:
        st.write("**Student Answer:**")
        student_answer = st.text_area(
            "Enter the student's answer:",
            height=150,
            placeholder="Type the student's answer here...",
            key="student_answer"
        )

    if st.button("Evaluate Answer", type="primary"):
        if model_answer and student_answer:
            with st.spinner('Evaluating answer...'):
                # Evaluate based on question type
                if question_type == "Short Answer":
                    scores = st.session_state.evaluator.evaluate_short_answer(model_answer, student_answer)
                    final_score = scores.get('final_score', 0)
                    eval_type = "short_answer"
                elif question_type == "Essay":
                    scores = st.session_state.evaluator.evaluate_essay(model_answer, student_answer)
                    final_score = scores.get('final_essay_score', 0)
                    eval_type = "essay"
                else:  # Multiple Choice
                    scores = st.session_state.evaluator.evaluate_mcq(model_answer, student_answer)
                    final_score = scores.get('score', 0)
                    eval_type = "mcq"

                # Display results
                st.success("Evaluation completed!")

                # Main score display
                grade, color, css_class = get_grade_info(final_score)

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="score-card {css_class}">
                        <h2 style="color: {color}; text-align: center; margin: 0;">
                            {final_score:.2f}/1.00
                        </h2>
                        <h3 style="color: black; text-align: center; margin: 0.5rem 0;">
                            Grade: {grade}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

                # Add to history
                st.session_state.evaluation_history.append({
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'question_type': question_type,
                    'model_answer': model_answer[:50] + "..." if len(model_answer) > 50 else model_answer,
                    'student_answer': student_answer[:50] + "..." if len(student_answer) > 50 else student_answer,
                    'score': final_score,
                    'grade': grade
                })

        else:
            st.error("Please enter both model answer and student answer!")


def show_evaluation_history():
    """Display evaluation history"""
    st.subheader("Evaluation History")

    if st.session_state.evaluation_history:
        history_df = pd.DataFrame(st.session_state.evaluation_history)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Evaluations", len(history_df))
        with col2:
            st.metric("Average Score", f"{history_df['score'].mean():.3f}")
        with col3:
            excellent_count = len(history_df[history_df['grade'] == 'Excellent'])
            st.metric("Excellent Answers", excellent_count)

        # History table
        st.dataframe(history_df, use_container_width=True)

        # Clear history button
        if st.button("Clear History"):
            st.session_state.evaluation_history = []
            st.rerun()

        # Score trend
        if len(history_df) > 1:
            fig = px.line(history_df, x='timestamp', y='score', title='Score Trend Over Time')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation history yet. Start evaluating answers to see history here!")


def show_example_data():
    """Show example data format"""
    st.subheader("Example Data Format")

    example_data = pd.DataFrame({
        'question': [
            'What is the capital of France?',
            'Explain photosynthesis',
            'What is 2+2?'
        ],
        'model_answer': [
            'Paris is the capital of France.',
            'Photosynthesis is the process by which plants convert sunlight into energy.',
            'A) 4'
        ],
        'student_answer': [
            'Paris',
            'Plants use sunlight to make food',
            'A) 4'
        ],
        'question_type': [
            'short_answer',
            'short_answer',
            'mcq'
        ]
    })

    st.dataframe(example_data)

    # Download example CSV
    csv = example_data.to_csv(index=False)
    st.download_button(
        label="Download Example CSV",
        data=csv,
        file_name="example_evaluation_data.csv",
        mime="text/csv"
    )


def main():
    """Main app function"""
    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">📚 Student Answer Evaluation System</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Single Answer Evaluation", "Evaluation History", "Example Data"]
    )

    # System info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Information")

    if hasattr(st.session_state.evaluator, 'sentence_model') and st.session_state.evaluator.sentence_model:
        st.sidebar.success("✅ BERT Model: Loaded")
    else:
        st.sidebar.warning("⚠️ BERT Model: Not Available")

    if st.session_state.evaluator.nltk_available:
        st.sidebar.success("✅ NLTK: Available")
    else:
        st.sidebar.warning("⚠️ NLTK: Using Fallback")

    # Main content based on page selection
    if page == "Single Answer Evaluation":
        evaluate_single_answer()
    elif page == "Evaluation History":
        show_evaluation_history()
    elif page == "Example Data":
        show_example_data()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Student Answer Evaluation System | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":

    main()
