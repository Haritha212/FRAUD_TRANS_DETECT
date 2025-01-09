from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the pre-trained models and encoders
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('final_model.pkl', 'rb'))

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the result page
@app.route('/templates/result.html', methods=['POST'])
def result():
    try:
        # Retrieve form data
        amount = float(request.form['amount'])
        type_ = request.form['type']
        old_bal_org = float(request.form['old_bal_org'])
        new_bal_org = float(request.form['new_bal_org'])
        old_bal_dest = float(request.form['old_bal_dest'])
        new_bal_dest = float(request.form['new_bal_dest'])

        # Prepare input data with the correct number of features
        # Assuming missing features are filled with zeros or a similar placeholder
        type_encoded = encoder.transform([[type_]])  # Encode the type column
        input_data = np.zeros((1, 13))  # Create an array with 13 features (expected by scaler)
        input_data[0, 0] = type_encoded[0][0]  # Encoded transaction type
        input_data[0, 1] = amount  # Transaction amount
        input_data[0, 2] = old_bal_org  # Old balance (origin)
        input_data[0, 3] = new_bal_org  # New balance (origin)
        input_data[0, 4] = old_bal_dest  # Old balance (destination)
        input_data[0, 5] = new_bal_dest  # New balance (destination)

        # Scale the data
        input_data = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_data)

        # Interpret the result
        result_text = "Fraudulent" if prediction[0] == 1 else "Legitimate"

        return render_template('result.html', result=result_text)

    except Exception as e:
        return f"Error: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)