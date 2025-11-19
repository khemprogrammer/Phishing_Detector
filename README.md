# Phishing_Detector

This is a Django-based web application for detecting phishing URLs. It uses machine learning models to analyze URLs and predict whether they are legitimate or phishing.

## Features

- URL input and prediction
- Displays prediction probability
- Integrates with machine learning models for phishing detection

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/khemprogrammer/Phishing_Detector.git
   cd Phishing_Detector
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the machine learning models:**

   ```bash
   python -m ml.train
   ```

5. **Run the Django development server:**

   ```bash
   python manage.py runserver
   ```

   The application will be accessible at `http://127.0.0.1:8000/`.

## Usage

Enter a URL in the provided input field on the homepage and click "Analyze URL" to get a prediction.

## Project Structure

- `detector/`: Django application for the web interface.
- `ml/`: Contains machine learning models, feature extraction logic, and training scripts.
- `ml/data/`: Stores datasets used for training.
- `ml/artifacts/`: Stores trained models and metadata.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## License

This project is licensed under the MIT License.