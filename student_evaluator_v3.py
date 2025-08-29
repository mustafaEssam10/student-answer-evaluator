import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from difflib import SequenceMatcher
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')


def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    try:
        try:
            nltk.data.find('tokenizers/punkt')
        except:
            print("Downloading punkt tokenizer...")
            nltk.download('punkt', quiet=False)

        try:
            nltk.data.find('corpora/stopwords')
        except:
            print("Downloading stopwords...")
            nltk.download('stopwords', quiet=False)

    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False
    return True


class StudentAnswerEvaluator:
    def __init__(self, use_bert_only=True):
        """
        Initialize the evaluator with BERT as primary method

        Args:
            use_bert_only (bool): If True, primarily use BERT for evaluation
        """
        self.use_bert_only = use_bert_only
        self.nltk_available = download_nltk_data()

        # Initialize traditional methods as backup
        if not use_bert_only:
            self.stemmer = PorterStemmer()
            if self.nltk_available:
                try:
                    self.stop_words = set(stopwords.words('english'))
                except:
                    self.stop_words = self._get_fallback_stopwords()
                    self.nltk_available = False
            else:
                self.stop_words = self._get_fallback_stopwords()

            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

        # Load BERT model - using the more accurate all-mpnet-base-v2
        try:
            print("Loading BERT model (all-mpnet-base-v2)...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ BERT model loaded successfully!")
            self.bert_available = True
        except Exception as e:
            print(f"❌ Could not load BERT model: {e}")
            if use_bert_only:
                print("⚠️  Switching to traditional methods as fallback")
                self.use_bert_only = False
                self._init_traditional_methods()
            self.sentence_model = None
            self.bert_available = False

    def _init_traditional_methods(self):
        """Initialize traditional methods if BERT fails"""
        self.stemmer = PorterStemmer()
        if self.nltk_available:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = self._get_fallback_stopwords()
        else:
            self.stop_words = self._get_fallback_stopwords()

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    def _get_fallback_stopwords(self):
        """Fallback stopwords if NLTK is not available"""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }

    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization without NLTK"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [token for token in text.split() if token.strip()]
        return tokens

    def simple_sent_tokenize(self, text: str) -> List[str]:
        """Simple sentence tokenization without NLTK"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        return sentences

    def preprocess_text(self, text: str) -> str:
        """Clean and prepare text"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def calculate_bert_similarity(self, model_answer: str, student_answer: str) -> float:
        """Calculate similarity using BERT embeddings (primary method)"""
        if not self.bert_available or self.sentence_model is None:
            return 0.0

        try:
            # Get embeddings for both texts
            embeddings = self.sentence_model.encode([model_answer, student_answer])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"BERT similarity calculation failed: {e}")
            return 0.0

    def calculate_bert_similarity_detailed(self, model_answer: str, student_answer: str) -> Dict[str, float]:
        """Calculate detailed BERT similarity with confidence scoring"""
        if not self.bert_available:
            return {'similarity': 0.0, 'confidence': 0.0}

        try:
            # Calculate basic similarity
            similarity = self.calculate_bert_similarity(model_answer, student_answer)

            # Calculate confidence based on text lengths and similarity score
            min_length = min(len(model_answer.split()), len(student_answer.split()))
            max_length = max(len(model_answer.split()), len(student_answer.split()))

            # Length ratio affects confidence
            length_ratio = min_length / max_length if max_length > 0 else 0

            # Higher similarity with good length ratio = higher confidence
            confidence = similarity * (0.7 + 0.3 * length_ratio)

            return {
                'similarity': similarity,
                'confidence': confidence,
                'length_ratio': length_ratio
            }
        except:
            return {'similarity': 0.0, 'confidence': 0.0, 'length_ratio': 0.0}

    # Backup traditional methods (only used if BERT fails)
    def calculate_word_overlap(self, model_answer: str, student_answer: str) -> float:
        """Calculate word overlap - backup method"""
        if not hasattr(self, 'stemmer'):
            return 0.0

        try:
            model_tokens = set(self.simple_tokenize(model_answer))
            student_tokens = set(self.simple_tokenize(student_answer))

            if not model_tokens:
                return 0.0

            intersection = len(model_tokens.intersection(student_tokens))
            return intersection / len(model_tokens)
        except:
            return 0.0

    def calculate_tfidf_similarity(self, model_answer: str, student_answer: str) -> float:
        """Calculate TF-IDF similarity - backup method"""
        if not hasattr(self, 'tfidf_vectorizer'):
            return 0.0

        try:
            documents = [model_answer, student_answer]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0

    def calculate_sequence_similarity(self, model_answer: str, student_answer: str) -> float:
        """Calculate sequence similarity - backup method"""
        try:
            return SequenceMatcher(None,
                                   self.preprocess_text(model_answer),
                                   self.preprocess_text(student_answer)).ratio()
        except:
            return 0.0

    def evaluate_short_answer(self, model_answer: str, student_answer: str) -> Dict[str, float]:
        """Evaluate short answers - BERT-focused approach"""
        scores = {}

        if self.bert_available:
            # Primary BERT-based evaluation
            bert_details = self.calculate_bert_similarity_detailed(model_answer, student_answer)
            scores['bert_similarity'] = bert_details['similarity']
            scores['confidence'] = bert_details['confidence']
            scores['length_ratio'] = bert_details['length_ratio']

            # Final score is primarily based on BERT
            scores['final_score'] = bert_details['similarity']

            # Add traditional methods as supplementary info (if not BERT-only mode)
            if not self.use_bert_only:
                scores['word_overlap'] = self.calculate_word_overlap(model_answer, student_answer)
                scores['tfidf_similarity'] = self.calculate_tfidf_similarity(model_answer, student_answer)
                scores['sequence_similarity'] = self.calculate_sequence_similarity(model_answer, student_answer)

                # Slightly adjust BERT score with traditional methods for very low scores
                if scores['bert_similarity'] < 0.3:
                    traditional_avg = np.mean([
                        scores['word_overlap'],
                        scores['tfidf_similarity'],
                        scores['sequence_similarity']
                    ])
                    # Give 10% weight to traditional methods for low BERT scores
                    scores['final_score'] = 0.9 * scores['bert_similarity'] + 0.1 * traditional_avg

        else:
            # Fallback to traditional methods if BERT not available
            scores['word_overlap'] = self.calculate_word_overlap(model_answer, student_answer)
            scores['tfidf_similarity'] = self.calculate_tfidf_similarity(model_answer, student_answer)
            scores['sequence_similarity'] = self.calculate_sequence_similarity(model_answer, student_answer)

            # Weighted average of traditional methods
            weights = {'word_overlap': 0.4, 'tfidf_similarity': 0.4, 'sequence_similarity': 0.2}
            scores['final_score'] = sum(scores[key] * weights[key] for key in weights if key in scores)
            scores['bert_similarity'] = 0.0
            scores['confidence'] = 0.5  # Lower confidence without BERT

        return scores

    def evaluate_essay(self, model_answer: str, student_answer: str) -> Dict[str, float]:
        """Evaluate essays - BERT-focused approach"""
        scores = {}

        if self.bert_available:
            # Overall semantic similarity using BERT
            overall_similarity = self.calculate_bert_similarity(model_answer, student_answer)
            scores['overall_similarity'] = overall_similarity

            # Sentence-by-sentence analysis
            if self.nltk_available:
                try:
                    model_sentences = sent_tokenize(model_answer)
                    student_sentences = sent_tokenize(student_answer)
                except:
                    model_sentences = self.simple_sent_tokenize(model_answer)
                    student_sentences = self.simple_sent_tokenize(student_answer)
            else:
                model_sentences = self.simple_sent_tokenize(model_answer)
                student_sentences = self.simple_sent_tokenize(student_answer)

            # For each model sentence, find best matching student sentence
            sentence_scores = []
            for model_sent in model_sentences:
                best_score = 0
                for student_sent in student_sentences:
                    sent_similarity = self.calculate_bert_similarity(model_sent, student_sent)
                    if sent_similarity > best_score:
                        best_score = sent_similarity
                sentence_scores.append(best_score)

            scores['average_sentence_match'] = np.mean(sentence_scores) if sentence_scores else 0

            # Coverage: how many model sentences have good matches (>0.4 threshold for BERT)
            coverage_threshold = 0.4
            coverage_score = sum(1 for score in sentence_scores if score > coverage_threshold) / len(
                model_sentences) if model_sentences else 0
            scores['coverage_score'] = coverage_score

            # Content depth: reward comprehensive answers
            length_factor = min(len(student_sentences) / len(model_sentences), 1.5) if model_sentences else 1
            scores['content_depth'] = length_factor

            # Final essay score (BERT-focused)
            final_score = (
                    overall_similarity * 0.5 +  # Overall semantic match
                    scores['average_sentence_match'] * 0.3 +  # Sentence-level matches
                    coverage_score * 0.15 +  # Coverage of key points
                    (length_factor - 1) * 0.05  # Bonus for comprehensive answers
            )
            scores['final_essay_score'] = min(final_score, 1.0)  # Cap at 1.0

        else:
            # Fallback to traditional methods
            scores['overall_similarity'] = self.calculate_tfidf_similarity(model_answer, student_answer)
            scores['average_sentence_match'] = scores['overall_similarity']
            scores['coverage_score'] = self.calculate_word_overlap(model_answer, student_answer)
            scores['content_depth'] = 1.0
            scores['final_essay_score'] = np.mean([
                scores['overall_similarity'],
                scores['coverage_score']
            ])

        return scores

    def evaluate_mcq(self, model_answer: str, student_answer: str) -> Dict[str, float]:
        """Evaluate multiple choice questions"""
        # Clean answers
        model_clean = self.preprocess_text(model_answer).strip()
        student_clean = self.preprocess_text(student_answer).strip()

        # Exact match
        exact_match = 1.0 if model_clean == student_clean else 0.0

        # If not exact match, use BERT for semantic similarity (handles cases like "A) 4" vs "4")
        if exact_match == 0.0 and self.bert_available:
            bert_similarity = self.calculate_bert_similarity(model_answer, student_answer)
            # For MCQ, high similarity (>0.85) should count as correct
            is_correct = bert_similarity > 0.85
        else:
            # Fallback to sequence similarity
            similarity = self.calculate_sequence_similarity(model_answer, student_answer)
            is_correct = exact_match == 1.0 or similarity > 0.9
            bert_similarity = similarity

        return {
            'exact_match': exact_match,
            'similarity': bert_similarity,
            'is_correct': is_correct,
            'score': 1.0 if is_correct else 0.0
        }

    def get_evaluation_summary(self, scores: Dict, question_type: str) -> Dict[str, str]:
        """Get human-readable summary of evaluation"""
        if question_type == "short_answer":
            final_score = scores.get('final_score', 0)
        elif question_type == "essay":
            final_score = scores.get('final_essay_score', 0)
        else:  # MCQ
            final_score = scores.get('score', 0)

        # Grade classification
        if final_score >= 0.85:
            grade = "Excellent"
            description = "Outstanding answer with strong semantic similarity to the model answer."
        elif final_score >= 0.70:
            grade = "Good"
            description = "Good answer that captures most key concepts and ideas."
        elif final_score >= 0.50:
            grade = "Acceptable"
            description = "Acceptable answer with some relevant content, but missing key elements."
        elif final_score >= 0.30:
            grade = "Poor"
            description = "Poor answer with limited relevance to the expected response."
        else:
            grade = "Very Poor"
            description = "Answer shows little to no understanding of the question."

        return {
            'grade': grade,
            'score': f"{final_score:.2f}",
            'percentage': f"{final_score * 100:.1f}%",
            'description': description,
            'method': "BERT-based semantic analysis" if self.bert_available else "Traditional methods"
        }


# Simple test function
def test_evaluator():
    """Test the BERT-focused evaluator"""
    print("Testing BERT-focused Student Answer Evaluator")
    print("=" * 50)

    evaluator = StudentAnswerEvaluator(use_bert_only=True)

    # Test cases
    test_cases = [
        {
            'question': 'What is photosynthesis?',
            'model_answer': 'Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.',
            'student_answer': 'Plants use sunlight to make food from CO2 and water, producing sugar and oxygen.',
            'type': 'short_answer'
        },
        {
            'question': 'What is the capital of France?',
            'model_answer': 'Paris',
            'student_answer': 'The capital city of France is Paris',
            'type': 'short_answer'
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Question: {test['question']}")
        print(f"Model Answer: {test['model_answer']}")
        print(f"Student Answer: {test['student_answer']}")

        scores = evaluator.evaluate_short_answer(test['model_answer'], test['student_answer'])
        summary = evaluator.get_evaluation_summary(scores, test['type'])

        print(f"Score: {summary['score']} ({summary['percentage']})")
        print(f"Grade: {summary['grade']}")
        print(f"Method: {summary['method']}")
        if evaluator.bert_available:
            print(f"BERT Similarity: {scores.get('bert_similarity', 0):.3f}")
            print(f"Confidence: {scores.get('confidence', 0):.3f}")


if __name__ == "__main__":
    test_evaluator()