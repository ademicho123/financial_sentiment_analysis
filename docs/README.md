# Documentation Index

Welcome to the Financial Sentiment Analysis project documentation! This directory contains comprehensive architecture diagrams and technical documentation.

## 📚 Documentation Files

### 1. [Architecture Overview](architecture.md)
**Comprehensive system architecture documentation**

Contains detailed diagrams and explanations for:
- **System Architecture**: High-level component structure
- **Data Flow Architecture**: How data moves through the system
- **Component Interaction**: Sequence diagrams showing interactions
- **ML Pipeline Architecture**: Complete machine learning workflow
- **Deployment Architecture**: Production deployment setup
- **Technology Stack**: All frameworks and libraries used
- **Performance Considerations**: Optimization strategies
- **Security Considerations**: Data privacy and model security

**Best for**: Understanding the overall system design and how components work together

---

### 2. [Class Diagrams](class_diagrams.md)
**Object-oriented design and structure**

Contains detailed diagrams for:
- **Core Classes and Modules**: Class relationships and dependencies
- **Data Models**: Data structures and their relationships
- **Workflow State Machine**: Application state transitions
- **Component Dependencies**: Module dependency graph
- **Model Architecture**: TinyBERT internal structure
- **Testing Architecture**: Test organization and coverage
- **Design Patterns**: Patterns used in the codebase

**Best for**: Understanding the code structure and object-oriented design

---

### 3. [API Reference](api_reference.md)
**Complete API documentation**

Contains detailed documentation for:
- **dashboard.py**: Streamlit interface functions
- **sentiment_predictor.py**: Prediction and analysis functions
- **financial_sentiment_analysis.py**: Core ML pipeline functions
- **attention_visualization.py**: Attention heatmap generation
- **shap_analysis.py**: SHAP explainability functions
- **Data Structures**: Return types and formats
- **Usage Examples**: Code snippets and workflows
- **Error Handling**: Exception types and handling strategies

**Best for**: Developers integrating with or extending the codebase

---

## 🎯 Quick Navigation

### For New Users
1. Start with [Architecture Overview](architecture.md) to understand the system
2. Review the **Data Flow Architecture** section to see how analysis works
3. Check **Deployment Architecture** to understand how to run the application

### For Developers
1. Read [Class Diagrams](class_diagrams.md) to understand code structure
2. Use [API Reference](api_reference.md) for function signatures and parameters
3. Review **Design Patterns** to understand architectural decisions

### For Data Scientists
1. Check **ML Pipeline Architecture** in [Architecture Overview](architecture.md)
2. Review **Model Architecture** in [Class Diagrams](class_diagrams.md)
3. See **Training Configuration** in [API Reference](api_reference.md)

### For DevOps/Deployment
1. Review **Deployment Architecture** in [Architecture Overview](architecture.md)
2. Check **Technology Stack** for dependencies
3. Review **Performance Considerations** for optimization

---

## 🔍 Diagram Types

This documentation uses several types of diagrams:

### Mermaid Diagrams
All diagrams are created using Mermaid syntax and can be viewed in:
- GitHub (native support)
- VS Code (with Mermaid extension)
- Any Markdown viewer with Mermaid support

### Diagram Categories

#### 1. **Component Diagrams**
Show the structure and relationships between system components
- Found in: [Architecture Overview](architecture.md)

#### 2. **Sequence Diagrams**
Show interactions between components over time
- Found in: [Architecture Overview](architecture.md), [Class Diagrams](class_diagrams.md)

#### 3. **Flowcharts**
Show data flow and process steps
- Found in: [Architecture Overview](architecture.md)

#### 4. **Class Diagrams**
Show object-oriented structure and relationships
- Found in: [Class Diagrams](class_diagrams.md)

#### 5. **State Diagrams**
Show application state transitions
- Found in: [Class Diagrams](class_diagrams.md)

---

## 📖 Documentation Standards

### Code Examples
All code examples are tested and follow Python best practices:
- Type hints where applicable
- Clear variable names
- Comprehensive error handling
- Inline comments for complex logic

### Diagram Conventions

#### Color Coding
- 🟢 **Green**: User-facing components (UI, input/output)
- 🔵 **Blue**: Application logic and business rules
- 🟠 **Orange**: Core processing and data handling
- 🟣 **Purple**: ML models and algorithms
- 🔴 **Red**: Results and final outputs

#### Naming Conventions
- **PascalCase**: Classes and interfaces
- **snake_case**: Functions and variables
- **UPPER_CASE**: Constants and configuration

---

## 🛠️ Project Structure

```
financial_sentiment_analysis/
├── docs/                           # 📁 This directory
│   ├── README.md                   # This file
│   ├── architecture.md             # System architecture
│   ├── class_diagrams.md           # Class structure
│   └── api_reference.md            # API documentation
├── data/                           # 📊 Training data
│   ├── lloyds/
│   ├── iag/
│   └── ...
├── sample_data/                    # 🧪 Test data
├── dashboard.py                    # 🖥️ Streamlit UI
├── sentiment_predictor.py          # 🎯 Prediction logic
├── financial_sentiment_analysis.py # 🧠 Core ML pipeline
├── attention_visualization.py      # 📈 Attention plots
├── shap_analysis.py               # 🔍 Explainability
├── unit_test.py                   # ✅ Unit tests
├── requirements.txt               # 📦 Dependencies
└── README.md                      # 📝 Project README
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/ademicho123/financial_sentiment_analysis.git
cd financial_sentiment_analysis

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start the Streamlit dashboard
streamlit run dashboard.py
```

### Running Tests
```bash
# Run unit tests
pytest unit_test.py -v
```

---

## 📊 Key Features

### 1. Multi-Model Support
- TinyBERT (primary)
- AdaBoost
- SVM
- Random Forest
- Naive Bayes
- Logistic Regression
- Neural Networks

### 2. Advanced Analysis
- **Sentiment Classification**: Bullish, Neutral, Bearish
- **SHAP Explanations**: Understand model decisions
- **Attention Visualization**: See what the model focuses on
- **VADER Scoring**: Traditional sentiment analysis

### 3. Data Processing
- PDF text extraction
- CSV data integration
- Text preprocessing pipeline
- Data augmentation
- Class balancing (SMOTE)

### 4. Interactive Dashboard
- Upload PDF files
- Paste text directly
- Real-time analysis
- Interactive visualizations
- Detailed metrics

---

## 🔧 Configuration

### Model Configuration
```python
MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
```

### Preprocessing Configuration
```python
SENTIMENT_THRESHOLD_POSITIVE = 0.05
SENTIMENT_THRESHOLD_NEGATIVE = -0.05
```

---

## 📈 Performance Metrics

### Model Performance (TinyBERT)
- **Accuracy**: ~85-90% (varies by dataset)
- **Inference Speed**: ~50-100ms per text
- **Model Size**: ~60MB
- **Memory Usage**: ~500MB (with model loaded)

### Supported Input
- **Max Text Length**: 512 tokens
- **Supported Formats**: PDF, plain text
- **Languages**: English (primary)

---

## 🤝 Contributing

### Documentation Updates
When updating documentation:
1. Keep diagrams consistent with code
2. Update all affected documentation files
3. Test all code examples
4. Follow existing formatting conventions

### Adding New Features
When adding features:
1. Update relevant architecture diagrams
2. Add API documentation
3. Include usage examples
4. Update this README if needed

---

## 📝 Version History

### Current Version: 1.0.0
- Initial documentation release
- Complete architecture diagrams
- Comprehensive API reference
- Class structure documentation

---

## 📧 Contact & Support

For questions or issues:
- **GitHub Issues**: [Project Issues](https://github.com/ademicho123/financial_sentiment_analysis/issues)
- **Documentation Issues**: Report in GitHub Issues with "docs" label

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

### Technologies Used
- **Hugging Face Transformers**: Pre-trained models
- **PyTorch**: Deep learning framework
- **Streamlit**: Web interface
- **SHAP**: Model interpretability
- **NLTK**: Natural language processing

### Model Credits
- **TinyBERT**: Huawei Noah's Ark Lab
- **VADER**: C.J. Hutto and Eric Gilbert

---

## 📚 Additional Resources

### External Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [SHAP Documentation](https://shap.readthedocs.io)

### Research Papers
- TinyBERT: Distilling BERT for Natural Language Understanding
- VADER: A Parsimonious Rule-based Model for Sentiment Analysis

---

**Last Updated**: January 4, 2026

**Documentation Version**: 1.0.0
