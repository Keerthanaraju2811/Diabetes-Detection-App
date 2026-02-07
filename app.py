from flask import Flask, render_template, request
import csv
import pickle
import numpy as np
import random
import datetime
import os

# -------------------------------------------------
# Flask App Configuration (IMPORTANT FIX)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# -------------------------------------------------
# Load ML Model and Scaler
# -------------------------------------------------
try:
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(os.path.join(BASE_DIR, 'lr.pkl'), 'rb') as model_file:
        lr_model = pickle.load(model_file)

    print("✅ ML Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: lr.pkl or scaler.pkl not found.")
    scaler = None
    lr_model = None
except Exception as e:
    print(f"❌ ERROR: {e}")
    scaler = None
    lr_model = None

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def calculate_blood_sugar_range(glucose):
    if glucose < 100:
        return "Normal (Under 100 mg/dL)", "bg-green-100 text-green-800"
    elif 100 <= glucose <= 125:
        return "Prediabetes Range (100–125 mg/dL)", "bg-yellow-100 text-yellow-800"
    else:
        return "Diabetes Range (Over 125 mg/dL)", "bg-red-100 text-red-800"


def get_diet_plan(prediction_int, glucose, bmi):
    diet_data = {}

    if prediction_int == 1:
        diet_data['severity'] = 'Positive'

        if glucose > 200:
            diet_data['recommendation'] = "Severe: Immediate consultation required."
            diet_data['food'] = ["Lean meats", "Legumes", "Whole grains (small portions)"]
            diet_data['fruits'] = ["Berries", "Apples (limited)"]
            diet_data['vegetables'] = ["Leafy greens", "Broccoli", "Peppers"]

        elif glucose > 140:
            diet_data['recommendation'] = "Moderate Risk: Follow a low GI diet."
            diet_data['food'] = ["Quinoa", "Barley", "Beans", "Nuts"]
            diet_data['fruits'] = ["Apples", "Pears", "Citrus"]
            diet_data['vegetables'] = ["Spinach", "Kale", "Green beans"]

        else:
            diet_data['recommendation'] = "High Risk: Control sugar & maintain BMI."
            diet_data['food'] = ["Oats", "Whole wheat", "Seeds"]
            diet_data['fruits'] = ["Berries", "Pears"]
            diet_data['vegetables'] = ["Tomatoes", "Cucumbers"]

    else:
        diet_data['severity'] = 'Negative'
        diet_data['recommendation'] = "Great job! Maintain a healthy lifestyle."
        diet_data['food'] = ["Balanced diet", "Lean protein", "Healthy fats"]
        diet_data['fruits'] = ["Seasonal fruits"]
        diet_data['vegetables'] = ["All vegetables"]

    diet_data['5210_rule'] = [
        "5 servings of fruits & vegetables",
        "1 hour physical activity",
        "2 hours max screen time",
        "0 sugary drinks"
    ]

    return diet_data


def get_positive_quote():
    quotes = [
        "Your health is an investment, not an expense.",
        "Small steps today lead to big changes tomorrow.",
        "Healthy habits create a healthy future.",
        "Knowledge is the first step toward wellness.",
        "Consistency beats intensity."
    ]
    return random.choice(quotes)

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/form')
def form_page():
    return render_template('index1.html')


@app.route('/result', methods=['POST'])
def result():
    if not lr_model or not scaler:
        return render_template('index1.html', error="ML model not loaded.")

    try:
        def get_value(key):
            return float(request.form.get(key, 0))

        user_name = request.form.get('name', 'Anonymous')

        raw_features = np.array([[
            get_value('Glucose'),
            get_value('BloodPressure'),
            get_value('SkinThickness'),
            get_value('Insulin'),
            get_value('BMI'),
            get_value('DiabetesPedigreeFunction'),
            get_value('Age')
        ]])

        scaled = scaler.transform(raw_features)
        prediction = int(lr_model.predict(scaled)[0])

        glucose = raw_features[0][0]
        bmi = raw_features[0][4]

        blood_range, range_class = calculate_blood_sugar_range(glucose)
        data = get_diet_plan(prediction, glucose, bmi)

        data.update({
            "blood_sugar_range": blood_range,
            "range_class": range_class,
            "quote": get_positive_quote() if prediction == 1 else None,
            "prediction_label": "POSITIVE for Diabetes" if prediction else "NEGATIVE for Diabetes",
            "prediction_class": "text-red-600" if prediction else "text-green-600"
        })

        # Save history
        with open(os.path.join(BASE_DIR, 'user_history.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                user_name,
                raw_features[0][6],
                glucose,
                bmi,
                data["prediction_label"],
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

        return render_template('result.html', data=data)

    except Exception as e:
        print("Prediction Error:", e)
        return render_template('index1.html', error="Invalid input values.")


@app.route('/history')
def history():
    history_data = []

    try:
        with open(os.path.join(BASE_DIR, 'user_history.csv'), newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                history_data.append({
                    "name": row[0],
                    "age": row[1],
                    "glucose": row[2],
                    "bmi": row[3],
                    "prediction": row[4],
                    "timestamp": row[5]
                })
    except FileNotFoundError:
        pass

    return render_template('history.html', history=history_data)

# -------------------------------------------------
# Run App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
