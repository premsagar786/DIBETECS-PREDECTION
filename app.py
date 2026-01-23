from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.knn import train_knn
from models.decision_tree import train_decision_tree
from models.mlp import train_mlp

app = Flask(__name__)

# Global variables to store models and scaler
models = {}
scaler = None
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

def load_or_train_models():
    """Load existing models or train new ones"""
    global models, scaler
    
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if models exist
    if (os.path.exists(f'{model_dir}/knn_model.pkl') and 
        os.path.exists(f'{model_dir}/tree_model.pkl') and 
        os.path.exists(f'{model_dir}/mlp_model.pkl') and
        os.path.exists(f'{model_dir}/scaler.pkl')):
        
        print("Loading existing models...")
        models['knn'] = joblib.load(f'{model_dir}/knn_model.pkl')
        models['tree'] = joblib.load(f'{model_dir}/tree_model.pkl')
        models['mlp'] = joblib.load(f'{model_dir}/mlp_model.pkl')
        scaler = joblib.load(f'{model_dir}/scaler.pkl')
        print("✓ Models loaded successfully!")
    else:
        print("Training new models...")
        # Load data
        data = pd.read_csv('data/diabetes.csv')
        
        # Split features and target
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Train models
        print("Training KNN...")
        models['knn'] = train_knn(X_train, y_train, n_neighbors=9)
        
        print("Training Decision Tree...")
        models['tree'] = train_decision_tree(X_train, y_train, max_depth=3)
        
        print("Training MLP...")
        X_train_scaled = scaler.transform(X_train)
        from sklearn.neural_network import MLPClassifier
        models['mlp'] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        models['mlp'].fit(X_train_scaled, y_train)
        
        # Evaluate models
        X_test_scaled = scaler.transform(X_test)
        print(f"\n✓ KNN Accuracy: {models['knn'].score(X_test, y_test):.2%}")
        print(f"✓ Decision Tree Accuracy: {models['tree'].score(X_test, y_test):.2%}")
        print(f"✓ MLP Accuracy: {models['mlp'].score(X_test_scaled, y_test):.2%}")
        
        # Save models
        joblib.dump(models['knn'], f'{model_dir}/knn_model.pkl')
        joblib.dump(models['tree'], f'{model_dir}/tree_model.pkl')
        joblib.dump(models['mlp'], f'{model_dir}/mlp_model.pkl')
        joblib.dump(scaler, f'{model_dir}/scaler.pkl')
        print("\n✓ Models saved successfully!")

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using all three models"""
    try:
        # Get input data from form
        input_data = []
        for feature in feature_names:
            value = float(request.form.get(feature, 0))
            input_data.append(value)
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Make predictions with all models
        predictions = {}
        
        # KNN prediction
        knn_pred = models['knn'].predict(input_df)[0]
        knn_proba = models['knn'].predict_proba(input_df)[0]
        predictions['knn'] = {
            'result': 'Diabetic' if knn_pred == 1 else 'Non-Diabetic',
            'confidence': float(max(knn_proba) * 100)
        }
        
        # Decision Tree prediction
        tree_pred = models['tree'].predict(input_df)[0]
        tree_proba = models['tree'].predict_proba(input_df)[0]
        predictions['tree'] = {
            'result': 'Diabetic' if tree_pred == 1 else 'Non-Diabetic',
            'confidence': float(max(tree_proba) * 100)
        }
        
        # MLP prediction
        input_scaled = scaler.transform(input_df)
        mlp_pred = models['mlp'].predict(input_scaled)[0]
        mlp_proba = models['mlp'].predict_proba(input_scaled)[0]
        predictions['mlp'] = {
            'result': 'Diabetic' if mlp_pred == 1 else 'Non-Diabetic',
            'confidence': float(max(mlp_proba) * 100)
        }
        
        # Calculate ensemble prediction (majority voting)
        votes = [knn_pred, tree_pred, mlp_pred]
        ensemble_pred = 1 if sum(votes) >= 2 else 0
        ensemble_confidence = (sum(votes) / len(votes)) * 100
        
        predictions['ensemble'] = {
            'result': 'Diabetic' if ensemble_pred == 1 else 'Non-Diabetic',
            'confidence': float(ensemble_confidence)
        }
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_data': dict(zip(feature_names, input_data))
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/info')
def info():
    """Display information about the models"""
    return render_template('info.html')

@app.route('/guide')
def guide():
    """Display diabetes health guide"""
    return render_template('guide.html')

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 Diabetes Prediction System")
    print("=" * 60)
    
    # Load or train models
    load_or_train_models()
    
    print("\n" + "=" * 60)
    print("🚀 Starting Flask server...")
    print("=" * 60)
    print("\n📍 Open your browser and go to: http://localhost:5000")
    print("\n⏹  Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
