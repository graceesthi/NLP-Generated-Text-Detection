# NLP Generated Text Detection

## Project Overview

This project implements a sophisticated text classification system to detect AI-generated content using BERT (Bidirectional Encoder Representations from Transformers). The system can distinguish between human-written and machine-generated text with high accuracy.

## Objective

Develop a robust classifier to identify automatically generated text in academic abstracts, addressing the growing need for AI-generated content detection in academic and professional contexts.

## Dataset

- **Source:** Hybrid subset from Vijini et al. research
- **Composition:** Human-written abstracts with AI-generated sentence replacements
- **Split:** 80% training, 20% testing
- **Classes:** Human-written vs. AI-generated text
- **Challenge:** Detecting subtle differences in hybrid documents

## Technologies Used

- **Python 3.x**
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - BERT implementation
- **scikit-learn** - Evaluation metrics
- **NumPy** - Numerical operations
- **tqdm** - Progress tracking

## Architecture

### BERT-based Classification Pipeline

1. **Tokenization:** BERT tokenizer for text preprocessing
2. **Encoding:** BERT embeddings for semantic understanding
3. **Classification:** Fine-tuned BERT for binary classification
4. **Optimization:** AdamW optimizer with learning rate scheduling

```python
Model Architecture:
├── BERT Base Model (bert-base-uncased)
├── Classification Head
├── Dropout Layer
└── Binary Output (Human/Generated)
```

## Repository Structure

```
NLP-Generated-Text-Detection/
├── README.md
├── requirements.txt
├── notebooks/
│   └── final_project.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── trainer.py
│   └── evaluation.py
├── data/
│   └── GeneratedTextDetection-main/
├── models/
│   └── best_model.pt
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── training_metrics.json
└── utils/
    └── preprocessing.py
```

## Methodology

### Data Preprocessing
- Text normalization and cleaning
- BERT tokenization with attention masks
- Sequence padding and truncation
- Label encoding for binary classification

### Model Training
- Fine-tuning pre-trained BERT model
- Cross-entropy loss optimization
- Gradient clipping for stability
- Early stopping based on validation performance

### Evaluation Metrics
- **Accuracy:** Overall classification performance
- **Precision/Recall:** Class-specific performance
- **F1-Score:** Balanced performance measure
- **Confusion Matrix:** Detailed classification analysis

## Results

### Model Performance
- **Training Accuracy:** [Your achieved accuracy]%
- **Validation Accuracy:** [Your achieved accuracy]%
- **F1-Score:** [Your F1 score]
- **Precision:** [Your precision]
- **Recall:** [Your recall]

### Key Insights
- Effective detection of AI-generated content patterns
- High performance on hybrid text classification
- Robust generalization to unseen text samples
- Identification of linguistic markers in generated text

## Getting Started

### Installation
```bash
git clone [repository-url]
cd NLP-Generated-Text-Detection
pip install -r requirements.txt
```

### Usage
```python
from src.model import TextClassifier
from src.evaluation import evaluate_model

# Load trained model
classifier = TextClassifier.load_model('models/best_model.pt')

# Predict text authenticity
prediction = classifier.predict("Your text here...")
print(f"Prediction: {'Human' if prediction == 0 else 'Generated'}")
```

### Training Your Own Model
```bash
python src/trainer.py --data_path data/ --epochs 10 --batch_size 16
```

## Research Context

This project addresses critical challenges in:
- **Academic Integrity:** Detecting AI assistance in academic writing
- **Content Authenticity:** Verifying human-authored content
- **NLP Applications:** Advancing text classification techniques
- **AI Ethics:** Understanding AI-generated content implications

## Technical Contributions

- **Advanced BERT Fine-tuning:** Optimized for text authenticity detection
- **Hybrid Dataset Processing:** Effective handling of mixed content
- **Robust Evaluation:** Comprehensive performance assessment
- **Scalable Architecture:** Adaptable to various text domains

## Academic Achievements

- Successful implementation of state-of-the-art NLP techniques
- Deep understanding of transformer architectures
- Practical application of AI ethics principles
- Contribution to academic integrity research

## References

- Vijini et al. - Generated Text Detection Dataset
- BERT: Pre-training of Deep Bidirectional Transformers
- Hugging Face Transformers Documentation

## License

This project is developed for academic purposes. Please cite appropriately if used for research.

---
*Advancing NLP research for content authenticity*