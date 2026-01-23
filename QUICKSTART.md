# 🚀 Quick Start Guide - Diabetes Prediction System

## Prerequisites
- Python 3.7+ installed on your system
- pip package manager

## Installation Steps

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- scikit-learn (machine learning)
- pandas (data processing)
- numpy (numerical computing)
- joblib (model persistence)

### 2. Download Dataset (if needed)
```bash
py download_dataset.py
```

The dataset is already included, but you can re-download if needed.

### 3. Run the Application
```bash
py app.py
```

### 4. Access the Web Interface
Open your browser and go to:
```
http://localhost:5000
```

## Alternative: Command Line Mode

To just train models and see accuracy scores:
```bash
py main.py
```

## Troubleshooting

### Python not found
- Make sure Python is installed
- Try `python` instead of `py`
- Check if Python is in your PATH

### Module not found errors
- Run: `py -m pip install -r requirements.txt`
- Make sure all dependencies are installed

### Port 5000 already in use
- The app will show an error
- Close any other applications using port 5000
- Or modify `app.py` to use a different port

## Features to Try

1. **Make Predictions**
   - Enter patient data in the form
   - Click "Analyze & Predict"
   - See results from 3 different ML models
   - View ensemble prediction

2. **Learn About Models**
   - Click "About Models" in navigation
   - Read about each algorithm
   - Understand the dataset

3. **Keyboard Shortcuts**
   - `Ctrl + Enter`: Submit form
   - `Escape`: Reset form

## Sample Data

### Example 1: Likely Diabetic
- Pregnancies: 6
- Glucose: 148
- Blood Pressure: 72
- Skin Thickness: 35
- Insulin: 125
- BMI: 33.6
- Diabetes Pedigree: 0.627
- Age: 50

### Example 2: Likely Non-Diabetic
- Pregnancies: 1
- Glucose: 85
- Blood Pressure: 66
- Skin Thickness: 29
- Insulin: 85
- BMI: 26.6
- Diabetes Pedigree: 0.351
- Age: 31

## Next Steps

- Explore the model information page
- Try different input values
- Check the `project_info.yaml` for detailed documentation
- Read the `README.md` for comprehensive information

## Support

For issues or questions:
- Email: sagarprem2006@gmail.com
- GitHub: https://github.com/premsagar786

---

**Happy Predicting! 🏥🤖**
