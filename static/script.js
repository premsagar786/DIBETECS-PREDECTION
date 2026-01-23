// Form handling and predictions
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');
    const resetBtn = document.getElementById('resetBtn');
    const resultsCard = document.getElementById('resultsCard');
    const predictBtn = document.getElementById('predictBtn');

    // Form submission
    if (form) {
        form.addEventListener('submit', async function (e) {
            e.preventDefault();

            // Show loading state
            predictBtn.classList.add('loading');
            predictBtn.disabled = true;
            const originalText = predictBtn.innerHTML;
            predictBtn.innerHTML = '<span>Analyzing...</span>';

            // Get form data
            const formData = new FormData(form);

            try {
                // Send prediction request
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    displayResults(data.predictions, data.input_data);

                    // Scroll to results
                    setTimeout(() => {
                        resultsCard.scrollIntoView({
                            behavior: 'smooth',
                            block: 'nearest'
                        });
                    }, 100);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while making the prediction. Please try again.');
            } finally {
                // Remove loading state
                predictBtn.classList.remove('loading');
                predictBtn.disabled = false;
                predictBtn.innerHTML = originalText;
            }
        });
    }

    // Reset button
    if (resetBtn) {
        resetBtn.addEventListener('click', function () {
            form.reset();
            if (resultsCard) {
                resultsCard.style.display = 'none';
            }
        });
    }

    // Display results function
    function displayResults(predictions, inputData) {
        const ensembleResult = document.getElementById('ensembleResult');
        const modelsResults = document.getElementById('modelsResults');

        // Show results card
        resultsCard.style.display = 'block';

        // Display ensemble result
        const isDiabetic = predictions.ensemble.result === 'Diabetic';
        ensembleResult.innerHTML = `
            <div class="ensemble-title">🎯 Ensemble Prediction</div>
            <div class="ensemble-prediction ${isDiabetic ? 'result-diabetic' : 'result-non-diabetic'}">
                ${predictions.ensemble.result}
            </div>
            <div class="ensemble-confidence">
                Confidence: ${predictions.ensemble.confidence.toFixed(1)}%
            </div>
        `;

        // Display individual model results
        const modelNames = {
            'knn': 'K-Nearest Neighbors',
            'tree': 'Decision Tree',
            'mlp': 'Neural Network (MLP)'
        };

        const modelIcons = {
            'knn': '🎯',
            'tree': '🌳',
            'mlp': '🧠'
        };

        let modelsHTML = '';
        for (const [key, value] of Object.entries(predictions)) {
            if (key !== 'ensemble') {
                const isDiabetic = value.result === 'Diabetic';
                modelsHTML += `
                    <div class="model-result">
                        <div class="model-name">
                            ${modelIcons[key]} ${modelNames[key]}
                        </div>
                        <div class="model-prediction ${isDiabetic ? 'result-diabetic' : 'result-non-diabetic'}">
                            ${value.result}
                        </div>
                        <div class="model-confidence">
                            Confidence: ${value.confidence.toFixed(1)}%
                        </div>
                    </div>
                `;
            }
        }

        modelsResults.innerHTML = modelsHTML;

        // Add animation
        resultsCard.style.opacity = '0';
        resultsCard.style.transform = 'translateY(20px)';

        setTimeout(() => {
            resultsCard.style.transition = 'all 0.5s ease';
            resultsCard.style.opacity = '1';
            resultsCard.style.transform = 'translateY(0)';
        }, 50);
    }

    // Add input validation and formatting
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        // Add focus animation
        input.addEventListener('focus', function () {
            this.parentElement.style.transform = 'scale(1.02)';
            this.parentElement.style.transition = 'transform 0.2s ease';
        });

        input.addEventListener('blur', function () {
            this.parentElement.style.transform = 'scale(1)';
        });

        // Validate on input
        input.addEventListener('input', function () {
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            const value = parseFloat(this.value);

            if (value < min) {
                this.value = min;
            } else if (value > max) {
                this.value = max;
            }
        });
    });

    // Add keyboard shortcuts
    document.addEventListener('keydown', function (e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (form && !predictBtn.disabled) {
                form.dispatchEvent(new Event('submit'));
            }
        }

        // Escape to reset form
        if (e.key === 'Escape') {
            if (resetBtn) {
                resetBtn.click();
            }
        }
    });

    // Add tooltips on hover for better UX
    const labels = document.querySelectorAll('.label-text');
    labels.forEach(label => {
        label.addEventListener('mouseenter', function () {
            this.style.color = '#667eea';
        });

        label.addEventListener('mouseleave', function () {
            this.style.color = '';
        });
    });

    // Parallax effect for background orbs
    document.addEventListener('mousemove', function (e) {
        const orbs = document.querySelectorAll('.gradient-orb');
        const mouseX = e.clientX / window.innerWidth;
        const mouseY = e.clientY / window.innerHeight;

        orbs.forEach((orb, index) => {
            const speed = (index + 1) * 10;
            const x = (mouseX - 0.5) * speed;
            const y = (mouseY - 0.5) * speed;

            orb.style.transform = `translate(${x}px, ${y}px)`;
        });
    });

    // Add sample data button functionality
    function loadSampleData(type) {
        const sampleData = {
            diabetic: {
                Pregnancies: 6,
                Glucose: 148,
                BloodPressure: 72,
                SkinThickness: 35,
                Insulin: 125,
                BMI: 33.6,
                DiabetesPedigreeFunction: 0.627,
                Age: 50
            },
            nonDiabetic: {
                Pregnancies: 1,
                Glucose: 85,
                BloodPressure: 66,
                SkinThickness: 29,
                Insulin: 85,
                BMI: 26.6,
                DiabetesPedigreeFunction: 0.351,
                Age: 31
            }
        };

        const data = sampleData[type];
        for (const [key, value] of Object.entries(data)) {
            const input = document.getElementById(key);
            if (input) {
                input.value = value;

                // Add animation
                input.style.background = 'rgba(102, 126, 234, 0.2)';
                setTimeout(() => {
                    input.style.background = '';
                }, 500);
            }
        }
    }

    // Expose function globally if needed
    window.loadSampleData = loadSampleData;

    // Add smooth scroll for all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Console easter egg
    console.log('%c🏥 Diabetes Prediction System', 'color: #667eea; font-size: 24px; font-weight: bold;');
    console.log('%cDeveloped by Prem Sagar Malhotra', 'color: #764ba2; font-size: 14px;');
    console.log('%cPowered by Machine Learning 🤖', 'color: #4facfe; font-size: 12px;');
});
