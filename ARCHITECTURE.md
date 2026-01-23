# 🏗️ Architecture & Data Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Beautiful Web Interface                   │ │
│  │  • Glassmorphism Design                               │ │
│  │  • Animated Gradients                                 │ │
│  │  • Interactive Forms                                  │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP Request/Response
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK WEB SERVER                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Routes:                                              │ │
│  │  • GET  /       → Home Page                          │ │
│  │  • POST /predict → Make Prediction                   │ │
│  │  • GET  /info   → Model Information                  │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREDICTION ENGINE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     KNN      │  │ Decision Tree│  │     MLP      │      │
│  │   Model      │  │    Model     │  │   Model      │      │
│  │  (k=9)       │  │ (depth=3)    │  │ (100→50)     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                  ┌─────────────────┐                        │
│                  │ Ensemble Voting │                        │
│                  │ (Majority 2/3)  │                        │
│                  └─────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│ User Input   │
│ (8 Features) │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Input Validation & Preprocessing     │
│ • Check ranges                       │
│ • Convert to DataFrame               │
│ • Normalize for MLP                  │
└──────┬───────────────────────────────┘
       │
       ├─────────────┬─────────────┬──────────────┐
       ▼             ▼             ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   KNN    │  │   Tree   │  │   MLP    │  │  Store   │
│ Predict  │  │ Predict  │  │ Predict  │  │  Input   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘
     │             │             │
     │  Result     │  Result     │  Result
     │  + Conf%    │  + Conf%    │  + Conf%
     │             │             │
     └─────────────┼─────────────┘
                   ▼
         ┌──────────────────┐
         │ Ensemble Voting  │
         │ • Collect votes  │
         │ • Majority rule  │
         │ • Calculate conf │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Format Response  │
         │ • All predictions│
         │ • Confidence %   │
         │ • Input echo     │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  JSON Response   │
         │  to Browser      │
         └──────────────────┘
```

## Model Training Flow

```
┌──────────────┐
│ First Run    │
│ or Retrain   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────┐
│ Load Dataset                │
│ • Read diabetes.csv         │
│ • 768 samples, 8 features   │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ Train-Test Split            │
│ • 80% Training (614)        │
│ • 20% Testing (154)         │
│ • Random state = 42         │
└──────┬──────────────────────┘
       │
       ├──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Train    │ │ Train    │ │ Train    │ │ Create   │
│ KNN      │ │ Tree     │ │ MLP      │ │ Scaler   │
│ Model    │ │ Model    │ │ Model    │ │          │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Evaluate │ │ Evaluate │ │ Evaluate │ │ Fit on   │
│ on Test  │ │ on Test  │ │ on Test  │ │ Train    │
│ ~79%     │ │ ~75%     │ │ ~83%     │ │ Data     │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Save Models    │
         │ • knn.pkl      │
         │ • tree.pkl     │
         │ • mlp.pkl      │
         │ • scaler.pkl   │
         └────────────────┘
```

## File Organization

```
DIBETECS-PREDECTION/
│
├── 🎯 Core Application
│   ├── app.py                 # Flask web server
│   ├── main.py                # CLI training script
│   └── download_dataset.py    # Dataset downloader
│
├── 🤖 Machine Learning Models
│   └── models/
│       ├── __init__.py
│       ├── knn.py            # K-Nearest Neighbors
│       ├── decision_tree.py  # Decision Tree
│       └── mlp.py            # Neural Network
│
├── 📊 Data
│   └── data/
│       └── diabetes.csv      # Pima Indians dataset
│
├── 🎨 Frontend
│   ├── templates/
│   │   ├── index.html        # Prediction page
│   │   └── info.html         # Information page
│   └── static/
│       ├── style.css         # Styling & animations
│       └── script.js         # Interactivity
│
├── 💾 Saved Models (auto-generated)
│   └── saved_models/
│       ├── knn_model.pkl
│       ├── tree_model.pkl
│       ├── mlp_model.pkl
│       └── scaler.pkl
│
└── 📚 Documentation
    ├── README.md             # Main documentation
    ├── project_info.yaml     # Comprehensive specs
    ├── QUICKSTART.md         # Quick start guide
    ├── PROJECT_SUMMARY.md    # Project summary
    ├── ARCHITECTURE.md       # This file
    ├── requirements.txt      # Dependencies
    └── LICENSE               # MIT License
```

## Component Interactions

```
┌─────────────────────────────────────────────────────────┐
│                    HTML Templates                        │
│  ┌──────────────┐              ┌──────────────┐        │
│  │  index.html  │◄────────────►│  info.html   │        │
│  │  (Predict)   │   Navigation │  (Learn)     │        │
│  └──────┬───────┘              └──────────────┘        │
│         │                                               │
│         │ Uses                                          │
│         ▼                                               │
│  ┌──────────────────────────────────────┐              │
│  │         Static Assets                │              │
│  │  ┌──────────┐      ┌──────────┐     │              │
│  │  │style.css │◄────►│script.js │     │              │
│  │  │(Design)  │      │(Logic)   │     │              │
│  │  └──────────┘      └──────────┘     │              │
│  └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
                         │
                         │ Rendered by
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Flask App (app.py)                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Routes & Controllers                           │   │
│  │  • Render templates                             │   │
│  │  • Handle form submissions                      │   │
│  │  • Coordinate predictions                       │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         │ Uses                          │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Model Manager                                  │   │
│  │  • Load/Save models                             │   │
│  │  • Train if needed                              │   │
│  │  • Make predictions                             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                         │
                         │ Imports
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Model Implementations                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  knn.py  │  │ tree.py  │  │  mlp.py  │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

## Technology Stack Layers

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│  • HTML5 (Structure)                    │
│  • CSS3 (Styling, Animations)           │
│  • JavaScript (Interactivity)           │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Application Layer               │
│  • Flask 3.0.0 (Web Framework)          │
│  • Jinja2 (Template Engine)             │
│  • Werkzeug (WSGI Utility)              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Business Logic Layer            │
│  • Prediction Engine                    │
│  • Ensemble Voting                      │
│  • Input Validation                     │
│  • Model Management                     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Machine Learning Layer          │
│  • scikit-learn 1.3.0                   │
│  • KNN Classifier                       │
│  • Decision Tree Classifier             │
│  • MLP Classifier                       │
│  • StandardScaler                       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Data Layer                      │
│  • pandas 2.2.1 (DataFrames)            │
│  • numpy 1.26.0 (Arrays)                │
│  • joblib 1.3.2 (Persistence)           │
│  • CSV Files (Storage)                  │
└─────────────────────────────────────────┘
```

## Request-Response Cycle

```
1. User fills form
   ↓
2. JavaScript captures submit
   ↓
3. AJAX POST to /predict
   ↓
4. Flask receives request
   ↓
5. Extract form data
   ↓
6. Create DataFrame
   ↓
7. Predict with KNN → Result A
   ↓
8. Predict with Tree → Result B
   ↓
9. Predict with MLP → Result C
   ↓
10. Ensemble voting → Final Result
    ↓
11. Format JSON response
    ↓
12. Send to browser
    ↓
13. JavaScript receives response
    ↓
14. Update UI with results
    ↓
15. Show animations
```

---

**This architecture ensures:**
- ✅ Separation of concerns
- ✅ Scalability
- ✅ Maintainability
- ✅ User-friendly interface
- ✅ Robust predictions
