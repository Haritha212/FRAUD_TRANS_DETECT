# FRAUD TRANSCATION DETECTION

## Objective
Our idea aims to develop a fraud detection system for financial transactions that can accurately identify fraudulent activities and prevent potential losses. By leveraging machine learning algorithms and advanced data analytics techniques, we aim to create a robust and effective solution that enhances security and trust in financial transactions.

## Features

- **EDA on transaction data**: Detailed visualizations on banking transcation trends.
- **Web deployment for induviduals**: Check if your transcations are fraudulant or not in webpage.
- **Analyze bulk transactions**: Upload a transactions data CSV in required format to analyze bulk data (suitable for banks).

## Technologies Used

- **Backend**: Python (Flask), Streamlit
- **Frontend**: HTML, CSS
- **Data Visualization**: Matplotlib, Seaborn
- **Database**: CSV
- **Other Libraries**: Pandas, NumPy, Sklearn, Imblearn

## Setup Instructions

Follow the steps below to set up and run the project locally.

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/traffic-accident-analysis.git
    cd traffic-accident-analysis
    ```

2. **Set up a Python Virtual Environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # For Linux/macOS
    env\Scripts\activate     # For Windows
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask App**:
    ```bash
    python app.py
    ```

5. **Open in Browser**:
    Navigate in your web browser.

## Usage

1. Navigate the sidebar to access different pages:
    - Home
    - Severity Prediction
    - Peak Time Prediction
    - About
2. Use the filter dropdowns to refine your analysis based on specific criteria (district, police station, etc.).
3. Explore the interactive map to view accident trends.

## Acknowledgments

- Flask for backend framework.
