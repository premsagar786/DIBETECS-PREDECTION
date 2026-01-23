# 🏥 Diabetes Prediction System

An advanced machine learning web application that predicts diabetes risk using multiple ML algorithms with a beautiful, modern interface.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🤖 **Three ML Models**: K-Nearest Neighbors, Decision Tree, and Neural Network (MLP)
- 🎯 **Ensemble Prediction**: Combines all models using majority voting for improved accuracy
- 💎 **Beautiful UI**: Modern, responsive design with glassmorphism effects
- ⚡ **Real-time Predictions**: Instant results with confidence scores
- 📊 **Model Comparison**: See predictions from all models side-by-side
- 💾 **Model Persistence**: Trained models are saved for fast predictions
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- 📚 **Educational**: Comprehensive information about each ML algorithm

## 🎯 Demo

The application provides:
- Interactive form for entering patient diagnostic data
- Real-time predictions from three different ML models
- Ensemble prediction with confidence scores
- Detailed information about each algorithm
- Beautiful, animated user interface

## 📊 Dataset

This project uses the **Pima Indians Diabetes Database** from the National Institute of Diabetes and Digestive and Kidney Diseases.

- **Total Samples**: 768
- **Features**: 8 diagnostic measurements
- **Target**: Binary classification (Diabetic/Non-Diabetic)
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Features:
1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration (mg/dL)
3. **Blood Pressure**: Diastolic blood pressure (mm Hg)
4. **Skin Thickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (μU/mL)
6. **BMI**: Body mass index (kg/m²)
7. **Diabetes Pedigree Function**: Genetic influence score
8. **Age**: Age in years

## 🧰 Models Implemented

### 1. K-Nearest Neighbors (KNN)
- **Accuracy**: ~79%
- **Parameters**: k=9 neighbors
- **Strengths**: Simple, effective for non-linear patterns

### 2. Decision Tree
- **Accuracy**: ~75%
- **Parameters**: max_depth=3
- **Strengths**: Highly interpretable, no normalization needed

### 3. Multi-Layer Perceptron (MLP)
- **Accuracy**: ~83%
- **Architecture**: 100 → 50 neurons
- **Strengths**: Captures complex patterns, highest accuracy

### 4. Ensemble Method
- **Accuracy**: ~80-85%
- **Method**: Majority voting (2 out of 3)
- **Benefit**: Most reliable predictions

## 📁 Project Structure

```
DIBETECS-PREDECTION/
├── models/                    # ML model implementations
│   ├── __init__.py
│   ├── knn.py                # K-Nearest Neighbors
│   ├── decision_tree.py      # Decision Tree
│   └── mlp.py                # Neural Network
├── data/                      # Dataset
│   └── diabetes.csv
├── templates/                 # HTML templates
│   ├── index.html            # Main prediction page
│   └── info.html             # Model information page
├── static/                    # Static assets
│   ├── style.css             # Styling with animations
│   └── script.js             # Interactive functionality
├── saved_models/              # Trained models (auto-generated)
│   ├── knn_model.pkl
│   ├── tree_model.pkl
│   ├── mlp_model.pkl
│   └── scaler.pkl
├── app.py                     # Flask web application
├── main.py                    # CLI model training
├── download_dataset.py        # Dataset downloader
├── requirements.txt           # Dependencies
├── project_info.yaml          # Comprehensive documentation
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/premsagar786/Dibetes-Prediction.git
cd Dibetes-Prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- Flask 3.0.0
- scikit-learn 1.3.0
- pandas 2.2.1
- numpy 1.26.0
- joblib 1.3.2

### Step 3: Download Dataset (Optional)
```bash
python download_dataset.py
```
*Note: Dataset is already included in the repository*

### Step 4: Run the Application
```bash
python app.py
```

### Step 5: Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

## 💻 Usage

### Web Interface
1. Navigate to `http://localhost:5000`
2. Enter patient diagnostic measurements in the form
3. Click **"Analyze & Predict"**
4. View predictions from all models plus ensemble result
5. Check confidence scores for each prediction

### Command Line
Train models and view accuracy:
```bash
python main.py
```

### API Endpoints

#### GET `/`
Home page with prediction form

#### POST `/predict`
Make predictions
- **Parameters**: All 8 diagnostic features
- **Response**: JSON with predictions from all models

#### GET `/info`
Information page about models and dataset

## 📈 Performance

| Model | Typical Accuracy |
|-------|-----------------|
| KNN | ~79% |
| Decision Tree | ~75% |
| MLP (Neural Network) | ~83% |
| **Ensemble** | **~80-85%** |

*Evaluated using 80-20 train-test split*

## 🎨 Design Features

- **Modern UI**: Glassmorphism effects with gradient backgrounds
- **Dark Theme**: Easy on the eyes with vibrant accent colors
- **Smooth Animations**: Floating gradient orbs and transitions
- **Responsive**: Mobile-first design approach
- **Accessible**: Semantic HTML and keyboard shortcuts
- **Interactive**: Hover effects and micro-animations

## 🔮 Future Enhancements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for robust evaluation
- [ ] Additional models (Random Forest, SVM, XGBoost)
- [ ] SHAP values for model interpretability
- [ ] User authentication and prediction history
- [ ] Export predictions to PDF
- [ ] Batch prediction from CSV upload
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Heroku/Google Cloud)
- [ ] RESTful API with authentication

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for **EDUCATIONAL and DEMONSTRATION purposes only**. 

- ❌ Do NOT use for actual medical diagnosis
- ❌ Do NOT use for treatment decisions
- ✅ Always consult qualified healthcare professionals
- ✅ Predictions are based on statistical models
- ✅ May not be accurate for individual cases

## 📚 Documentation

For comprehensive project information, see:
- **project_info.yaml**: Detailed technical documentation
- **README.md**: This file
- **In-app Info Page**: `/info` endpoint

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Prem Sagar Malhotra**

- 📧 Email: [sagarprem2006@gmail.com](mailto:sagarprem2006@gmail.com)
- 💼 LinkedIn: [Prem Sagar Malhotra](https://www.linkedin.com/in/prem-sagar-malhotra-99428a348)
- 🐙 GitHub: [@premsagar786](https://github.com/premsagar786)

## 🙏 Acknowledgments

- **Dataset**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Libraries**: scikit-learn, Flask, pandas, numpy teams
- **Inspiration**: Healthcare AI applications and ML education

## 📞 Support

For questions or issues:
- 📧 Email: sagarprem2006@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/premsagar786/Dibetes-Prediction/issues)
- 💬 Response time: Usually within 24-48 hours

---

<div align="center">
  <p>Made with ❤️ and 🤖 by Prem Sagar Malhotra</p>
  <p>© 2026 Diabetes Prediction System</p>
</div>
