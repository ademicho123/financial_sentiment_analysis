# Class Diagram - Financial Sentiment Analysis

This document provides detailed class diagrams showing the object-oriented structure of the project.

## Core Classes and Modules

```mermaid
classDiagram
    class StreamlitDashboard {
        +str page_title
        +str layout
        +read_pdf(file) str
        +explain_shap_values(shap_values, class_names) str
        +main()
    }
    
    class SentimentPredictor {
        -AutoTokenizer tokenizer
        -AutoModelForSequenceClassification model
        +model_predict(texts) ndarray
        +predict_sentiment(input_text) dict
        +analyze_text(input_text) dict
    }
    
    class FinancialSentimentAnalysis {
        +str MODEL_NAME
        +SentimentIntensityAnalyzer sid
        +read_pdf_sentences(file_path) list
        +extract_and_merge(pdf_path, csv_path) DataFrame
        +assign_sentiment_scores(text) float
        +assign_scores(data) DataFrame
        +assign_directions(data) DataFrame
        +preprocess_text(text) str
        +preprocess_data(df) DataFrame
        +sentiment_counts(data) dict
        +prepare_dataset(data, tokenizer, sample_frac, random_state, chunk_size) tuple
        +train_and_evaluate(model_name, train_dataset, test_dataset) dict
        +create_comprehensive_report(company_name, metrics, report_df, all_labels, all_preds) DataFrame
        +main(company_name, pdf_path, csv_path) dict
    }
    
    class TinyBERTForSequenceClassification {
        -BertModel bert
        -Dropout dropout
        -Linear classifier
        +forward(input_ids, attention_mask, token_type_ids, labels) tuple
    }
    
    class AttentionVisualizer {
        +visualize_attention(tokens, attention_weights, layer, head) Figure
    }
    
    class SHAPAnalyzer {
        -Explainer explainer
        +generate_shap_values(text) Explanation
    }
    
    class DataPreprocessor {
        -WordNetLemmatizer lemmatizer
        -set stopwords
        +clean_text(text) str
        +tokenize(text) list
        +lemmatize(tokens) list
        +remove_stopwords(tokens) list
    }
    
    class ModelTrainer {
        -AdamW optimizer
        -CosineAnnealingWarmRestarts scheduler
        -Accelerator accelerator
        +train_epoch(model, dataloader) float
        +evaluate(model, dataloader) dict
        +save_model(model, path)
        +load_model(path) Model
    }
    
    StreamlitDashboard --> SentimentPredictor : uses
    SentimentPredictor --> FinancialSentimentAnalysis : imports
    SentimentPredictor --> AttentionVisualizer : uses
    SentimentPredictor --> SHAPAnalyzer : uses
    FinancialSentimentAnalysis --> TinyBERTForSequenceClassification : uses
    FinancialSentimentAnalysis --> DataPreprocessor : uses
    FinancialSentimentAnalysis --> ModelTrainer : uses
    TinyBERTForSequenceClassification --|> AutoModelForSequenceClassification : extends
```

## Data Models

```mermaid
classDiagram
    class FinancialDocument {
        +str content
        +str source
        +datetime date
        +str company
        +extract_text() str
    }
    
    class PDFDocument {
        +str file_path
        +list~str~ sentences
        +extract_sentences() list
    }
    
    class NewsArticle {
        +str title
        +str content
        +datetime published_date
        +str source_url
    }
    
    class SentimentResult {
        +str sentiment
        +float score
        +str direction
        +str preprocessed_text
        +Explanation shap_values
        +Figure attention_plot
        +dict to_dict()
    }
    
    class ModelMetrics {
        +float accuracy
        +float precision
        +float recall
        +float f1_score
        +ndarray confusion_matrix
        +dict classification_report
        +list cross_val_scores
    }
    
    class Dataset {
        +DataFrame data
        +list features
        +list labels
        +int num_samples
        +split(test_size) tuple
        +augment() Dataset
        +balance() Dataset
    }
    
    FinancialDocument <|-- PDFDocument : inherits
    FinancialDocument <|-- NewsArticle : inherits
    SentimentPredictor ..> SentimentResult : creates
    ModelTrainer ..> ModelMetrics : produces
    FinancialSentimentAnalysis ..> Dataset : processes
```

## Workflow State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle
    
    Idle --> InputReceived : User uploads PDF or enters text
    
    InputReceived --> Preprocessing : Start analysis
    
    Preprocessing --> Tokenization : Text cleaned
    Tokenization --> ModelInference : Tokens ready
    
    ModelInference --> PostProcessing : Predictions generated
    
    PostProcessing --> SentimentScoring : Calculate VADER score
    PostProcessing --> DirectionAssignment : Assign direction
    PostProcessing --> SHAPAnalysis : Generate explanations
    PostProcessing --> AttentionVisualization : Create heatmap
    
    SentimentScoring --> ResultAggregation
    DirectionAssignment --> ResultAggregation
    SHAPAnalysis --> ResultAggregation
    AttentionVisualization --> ResultAggregation
    
    ResultAggregation --> DisplayResults : All components ready
    
    DisplayResults --> Idle : Analysis complete
    
    Preprocessing --> Error : Invalid input
    ModelInference --> Error : Model failure
    PostProcessing --> Error : Processing error
    
    Error --> Idle : Error handled
```

## Component Dependencies

```mermaid
graph LR
    subgraph "External Dependencies"
        A[torch]
        B[transformers]
        C[nltk]
        D[shap]
        E[streamlit]
        F[pandas]
        G[numpy]
        H[matplotlib]
        I[seaborn]
        J[PyPDF2]
        K[scikit-learn]
        L[imbalanced-learn]
        M[nlpaug]
    end
    
    subgraph "Project Modules"
        N[dashboard.py]
        O[sentiment_predictor.py]
        P[financial_sentiment_analysis.py]
        Q[attention_visualization.py]
        R[shap_analysis.py]
        S[unit_test.py]
    end
    
    N --> E
    N --> F
    N --> G
    N --> O
    N --> J
    N --> D
    N --> H
    
    O --> F
    O --> A
    O --> D
    O --> P
    O --> Q
    
    P --> A
    P --> B
    P --> C
    P --> F
    P --> G
    P --> H
    P --> I
    P --> J
    P --> K
    P --> L
    P --> M
    
    Q --> H
    Q --> I
    Q --> G
    
    R --> D
    
    S --> P
    S --> B
    S --> F
    
    style N fill:#4CAF50,stroke:#2E7D32,color:#fff
    style O fill:#2196F3,stroke:#1565C0,color:#fff
    style P fill:#FF9800,stroke:#E65100,color:#fff
```

## Model Architecture

```mermaid
graph TB
    subgraph "TinyBERT Architecture"
        A[Input: Token IDs<br/>Shape: batch_size × seq_length]
        
        B[Embedding Layer<br/>Token + Position + Segment]
        
        C1[Transformer Layer 1<br/>Self-Attention + FFN]
        C2[Transformer Layer 2<br/>Self-Attention + FFN]
        C3[Transformer Layer 3<br/>Self-Attention + FFN]
        C4[Transformer Layer 4<br/>Self-Attention + FFN]
        
        D[Pooler<br/>CLS Token Extraction]
        
        E[Dropout<br/>p=0.1]
        
        F[Linear Classifier<br/>312 → 3]
        
        G[Output: Logits<br/>Shape: batch_size × 3]
        
        A --> B
        B --> C1
        C1 --> C2
        C2 --> C3
        C3 --> C4
        C4 --> D
        D --> E
        E --> F
        F --> G
        
        C1 -.->|Attention Weights| H[Attention Layer 1]
        C2 -.->|Attention Weights| I[Attention Layer 2]
        C3 -.->|Attention Weights| J[Attention Layer 3]
        C4 -.->|Attention Weights| K[Attention Layer 4]
        
        H --> L[Attention Visualization]
        I --> L
        J --> L
        K --> L
    end
    
    style A fill:#E3F2FD,stroke:#1976D2
    style G fill:#C8E6C9,stroke:#388E3C
    style L fill:#FFF9C4,stroke:#F57F17
```

## Testing Architecture

```mermaid
graph TB
    subgraph "Unit Tests"
        A[test_read_pdf_sentences_valid]
        B[test_read_pdf_sentences_invalid_path]
        C[test_extract_and_merge_valid]
        D[test_extract_and_merge_missing_files]
        E[test_assign_sentiment_scores_positive]
        F[test_assign_sentiment_scores_negative]
        G[test_assign_sentiment_scores_neutral]
        H[test_assign_directions_bearish]
        I[test_assign_directions_bullish]
        J[test_assign_directions_neutral]
        K[test_preprocess_text_lowercase]
        L[test_preprocess_text_remove_numbers]
        M[test_preprocess_data_valid]
        N[test_prepare_dataset_valid]
    end
    
    subgraph "Test Targets"
        O[read_pdf_sentences]
        P[extract_and_merge]
        Q[assign_sentiment_scores]
        R[assign_directions]
        S[preprocess_text]
        T[preprocess_data]
        U[prepare_dataset]
    end
    
    subgraph "Test Data"
        V[sample_data/pdf_2.pdf]
        W[sample_data/pdf_3.pdf]
        X[sample_data/iag_sample_csv.csv]
    end
    
    A --> O
    B --> O
    C --> P
    D --> P
    E --> Q
    F --> Q
    G --> Q
    H --> R
    I --> R
    J --> R
    K --> S
    L --> S
    M --> T
    N --> U
    
    O --> V
    O --> W
    P --> V
    P --> W
    P --> X
    
    style A fill:#C8E6C9,stroke:#388E3C
    style B fill:#C8E6C9,stroke:#388E3C
    style C fill:#C8E6C9,stroke:#388E3C
    style D fill:#C8E6C9,stroke:#388E3C
    style E fill:#C8E6C9,stroke:#388E3C
    style F fill:#C8E6C9,stroke:#388E3C
    style G fill:#C8E6C9,stroke:#388E3C
    style H fill:#C8E6C9,stroke:#388E3C
    style I fill:#C8E6C9,stroke:#388E3C
    style J fill:#C8E6C9,stroke:#388E3C
    style K fill:#C8E6C9,stroke:#388E3C
    style L fill:#C8E6C9,stroke:#388E3C
    style M fill:#C8E6C9,stroke:#388E3C
    style N fill:#C8E6C9,stroke:#388E3C
```

---

## Key Design Patterns

### 1. **Facade Pattern**
- `SentimentPredictor` acts as a facade, providing a simplified interface to the complex subsystem of preprocessing, model inference, and analysis.

### 2. **Strategy Pattern**
- Different preprocessing strategies can be applied (lemmatization, stemming, etc.)
- Multiple model architectures can be swapped (TinyBERT, AdaBoost, SVM, etc.)

### 3. **Pipeline Pattern**
- Data flows through a series of transformations: Input → Preprocessing → Tokenization → Model → Post-processing → Output

### 4. **Singleton Pattern**
- Model and tokenizer are loaded once and reused across predictions to save memory and time

### 5. **Observer Pattern**
- Streamlit's reactive framework observes user inputs and triggers re-computation when needed

---

## Module Responsibilities

### `dashboard.py`
- **Responsibility**: User interface and interaction
- **Key Functions**: File upload, text input, result display, visualization rendering
- **Dependencies**: Streamlit, SentimentPredictor

### `sentiment_predictor.py`
- **Responsibility**: Orchestrate prediction workflow
- **Key Functions**: Text analysis, model prediction, SHAP generation, attention visualization
- **Dependencies**: FinancialSentimentAnalysis, AttentionVisualizer, TinyBERT

### `financial_sentiment_analysis.py`
- **Responsibility**: Core ML pipeline and data processing
- **Key Functions**: Data extraction, preprocessing, model training, evaluation
- **Dependencies**: PyTorch, Transformers, NLTK, scikit-learn

### `attention_visualization.py`
- **Responsibility**: Visualize model attention weights
- **Key Functions**: Generate heatmaps showing token importance
- **Dependencies**: Matplotlib, Seaborn

### `shap_analysis.py`
- **Responsibility**: Model interpretability
- **Key Functions**: Generate SHAP values for predictions
- **Dependencies**: SHAP library

### `unit_test.py`
- **Responsibility**: Automated testing
- **Key Functions**: Validate core functionality
- **Dependencies**: pytest, FinancialSentimentAnalysis

---

## Data Flow Between Classes

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant Predictor
    participant Preprocessor
    participant Model
    participant SHAP
    participant Visualizer
    
    User->>Dashboard: Input text/PDF
    Dashboard->>Predictor: analyze_text()
    
    Predictor->>Preprocessor: preprocess_text()
    Preprocessor->>Preprocessor: clean, tokenize, lemmatize
    Preprocessor-->>Predictor: cleaned_text
    
    Predictor->>Model: tokenize + forward()
    Model->>Model: Embedding → Transformers → Classifier
    Model-->>Predictor: logits, attention_weights
    
    Predictor->>Predictor: softmax, argmax
    
    par Parallel Analysis
        Predictor->>SHAP: explainer()
        SHAP-->>Predictor: shap_values
    and
        Predictor->>Visualizer: visualize_attention()
        Visualizer-->>Predictor: attention_plot
    end
    
    Predictor-->>Dashboard: SentimentResult
    Dashboard-->>User: Display results
```

---

## Conclusion

This class diagram documentation provides a comprehensive view of the object-oriented structure, dependencies, and design patterns used in the Financial Sentiment Analysis project. The modular design ensures maintainability, testability, and extensibility.
