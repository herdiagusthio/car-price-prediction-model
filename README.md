# Car Price Prediction Model

A machine learning project for training, validating, and preparing car price prediction models for deployment using an automated pipeline.

## 🚗 Project Overview

This project implements a complete machine learning pipeline for developing car price prediction models. The primary goal is to train, evaluate, and export models that can predict car prices based on various vehicle features.

### Key Features

- **Automated ML Pipeline**: End-to-end pipeline for training, validation, and deployment readiness checks.
- **Model Training**: Random Forest Regressor with comprehensive training pipeline.
- **Data Processing**: Advanced preprocessing with feature engineering and handling of unknown categories.
- **Model Export**: ONNX format for cross-platform compatibility.
- **CI/CD Integration**: GitHub Actions workflow for continuous integration and testing.
- **Containerization**: Docker support for consistent training environments.
- **Demo API**: Simple FastAPI interface for testing trained models.

## 📊 Dataset

The model is trained on the [Car Price Prediction dataset](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) from Kaggle. It contains 205 car records with 26 features including:

### Input Features
- **Categorical**: fuel type, aspiration, door number, car body, drive wheel, engine location, engine type, cylinder number, fuel system, brand
- **Numerical**: symboling, wheelbase, car length/width/height, curb weight, engine size, bore ratio, stroke, compression ratio, horsepower, peak RPM, city/highway MPG

### Target Variable
- **Price**: Car price in USD

## 🏗️ Project Structure

```
car-price-prediction-model/
├── .github/
│   └── workflows/
│       └── ci-cd.yml      # CI/CD pipeline configuration
├── app/
│   ├── config.py          # Configuration settings
│   ├── pipeline.py        # Main ML pipeline
│   ├── preprocessing.py   # Data preprocessing functions
│   ├── train.py           # Model training script
│   ├── predict.py         # Prediction script
│   └── ...
├── data/                    # Raw data (download from Kaggle)
├── logs/                    # Pipeline logs and results
├── models/                  # Trained models and preprocessors
│   ├── best_model.onnx
│   ├── preprocessor.joblib
│   └── ...
├── tests/                   # Unit and integration tests
├── main.py                  # FastAPI application
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
├── .gitignore               # Files to ignore in git
└── README.md                # Project documentation
```

## 📚 Documentation

- **[API Documentation](docs/API_DOCUMENTATION.md)** - Complete API reference with curl examples
- **[ML Engineering Best Practices](docs/ML_ENGINEERING_BEST_PRACTICES.md)** - Technical implementation details
- **Interactive API Docs** - Available at `http://localhost:8000/docs` when server is running

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- Docker (optional)

### Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd car-price-prediction-model
    ```

2.  **Download the dataset**

    This project uses the [Car Price Prediction dataset](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) from Kaggle.

    a. **Set up your Kaggle API credentials**

    To use the Kaggle API, you need to provide your credentials.

    1.  Go to your Kaggle account page and click "Create New API Token". This will download a `kaggle.json` file.
    2.  Create a directory named `.kaggle` in your home directory.
        ```bash
        mkdir -p ~/.kaggle
        ```
    3.  Move the downloaded `kaggle.json` file to that directory (ensure the path is correct if you downloaded it elsewhere).
        ```bash
        mv ~/Downloads/kaggle.json ~/.kaggle/
        ```
    4.  Set the correct permissions for the file.
        ```bash
        chmod 600 ~/.kaggle/kaggle.json
        ```

    b. **Download and extract the dataset**
    ```bash
    python3 -m app.download_data
    ```
    > **Important**: If you encounter an `OSError: Could not find kaggle.json`, please ensure you have completed the credential setup steps above. The `kaggle.json` file must be in the `~/.kaggle` directory.

    This will download `CarPrice_Assignment.csv` into the `data/` directory.

3.  **Install dependencies**
    ```bash
    pip3 install -r requirements.txt
    ```

### Running the Pipeline

The main entry point for this project is the `app/pipeline.py` script.

1.  **Run the full pipeline (training, validation, deployment checks)**
    ```bash
    python3 -m app.pipeline --validate --deploy
    ```
    This command runs the complete pipeline, including model training, validation against the production model, and deployment readiness checks.

2.  **Run a quick pipeline test**
    ```bash
    python3 -m app.pipeline --quick
    ```
    This command runs the pipeline with default settings and is ideal for quickly verifying the environment.

3.  **Run demo API (optional)**
    ```bash
    uvicorn main:app --reload
    ```
    - API Documentation: http://127.0.0.1:8000/docs

### 🔧 API Testing

To test the API:

```bash
# Start the API server
python3 main.py

# Test basic connectivity
curl http://localhost:8000/

# Check model status
curl http://localhost:8000/model/info

# Make a prediction
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"symboling": 0, "CarName": "toyota corolla", "fueltype": "gas", "aspiration": "std", "doornumber": "four", "carbody": "sedan", "drivewheel": "fwd", "enginelocation": "front", "wheelbase": 100.4, "carlength": 176.6, "carwidth": 66.2, "carheight": 54.3, "curbweight": 2337, "enginetype": "ohc", "cylindernumber": "four", "enginesize": 109, "fuelsystem": "mpfi", "boreratio": 3.19, "stroke": 3.4, "compressionratio": 10.0, "horsepower": 102, "peakrpm": 5500, "citympg": 24, "highwaympg": 30}'

# View interactive documentation
open http://localhost:8000/docs
```

**📖 For complete API documentation with more examples, see [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)**

### Docker Environment

1.  **Build the Docker image**
    ```bash
    docker build -t car-price-model-trainer .
    ```

2.  **Run the pipeline in a container**
    ```bash
    docker run -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs car-price-model-trainer python3 -m app.pipeline --validate --deploy
    ```

3.  **Run demo API in container (optional)**
    ```bash
    docker run -p 8000:8000 car-price-model-trainer
    ```

## 🔄 CI/CD Pipeline

This project uses GitHub Actions for Continuous Integration. The workflow is defined in `.github/workflows/ci-cd.yml`.

The pipeline automatically triggers on pushes and pull requests to the `main` branch and performs the following steps:
1.  Sets up the Python environment.
2.  Installs dependencies.
3.  Runs the quick pipeline test to ensure that the model training and prediction processes are working correctly.

## 🧪 Model Testing

### Command Line Testing

You can test the prediction script directly with a sample payload:

```bash
python3 -m app.predict
```

This will run a sample prediction using a predefined test data and display the results.

### Demo API (Optional)

A simple FastAPI interface is provided for interactive testing. See the "Quick Start" section for instructions on how to run it.

## 🔧 Model Training

The model training is orchestrated by the main pipeline (`app/pipeline.py`).

### Training Process

When you run the full pipeline, it executes the following steps:
1.  **Data Drift Analysis**: Checks for drift between new and existing data.
2.  **Model Training**:
    - Loads and validates the dataset.
    - Performs comprehensive data preprocessing.
    - Trains a Random Forest Regressor model.
    - Saves the `preprocessor` object.
    - Converts the trained model to ONNX format.
    - Saves model artifacts (`best_model.onnx`, `preprocessor.joblib`, metadata).
3.  **Model Validation**: Evaluates the model against the production version.
4.  **Deployment Preparation**: Runs a series of checks to ensure the model is ready for deployment.

### Training Output

-   `models/`: Contains the trained model (`best_model.onnx`), the preprocessor (`preprocessor.joblib`), and metadata.
-   `logs/`: Contains pipeline results and validation reports.
-   Console output showing pipeline progress and results.

## 🛠️ Model Development

### Data Preprocessing Pipeline

The preprocessing module (`app/preprocessing.py`) handles:

1. **Brand Extraction**: Automatically extracts car brand from `CarName` field
2. **Data Cleaning**: Corrects common brand name typos (maxda→mazda, porcshce→porsche, etc.)
3. **Feature Engineering**: Creates new features from existing data
4. **Data Validation**: Removes irrelevant columns (`car_ID`, original `CarName`)
5. **Categorical Feature Handling**: Uses `OneHotEncoder` to handle known and unknown categories gracefully.

### Model Architecture

The model is a `RandomForestRegressor` from scikit-learn, chosen for its robustness and performance on this type of tabular data.

- **Algorithm**: Random Forest Regressor (scikit-learn)
- **Input Features**: 74 features after one-hot encoding
- **Output**: Continuous price prediction in USD
- **Export Format**: ONNX for cross-platform deployment
- **Training Data**: 205 car records with 26 original features

### Feature Engineering

The model processes:
- **14 Numerical Features**: wheelbase, dimensions, weight, engine specs, performance metrics
- **10 Categorical Features**: fuel type, body style, drive type, engine configuration, brand
- **Final Feature Set**: 74 features after one-hot encoding categorical variables

### Supported Car Brands

The model recognizes the following car brands:
- Alfa Romeo, Audi, BMW, Buick, Chevrolet
- Dodge, Honda, Isuzu, Jaguar, Mazda
- Mercury, Mitsubishi, Nissan, Peugeot, Plymouth
- Porsche, Renault, Saab, Subaru, Toyota
- Volkswagen, Volvo

## 📋 Requirements

### Python Dependencies

```
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
skl2onnx
onnx
onnxruntime
```

### System Requirements

- **Memory**: Minimum 512MB RAM
- **Storage**: ~50MB for model and dependencies
- **CPU**: Any modern processor

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `best_model.onnx` and `kolom_model.txt` are in the `models/` directory
   - Run training script if files are missing: `python3 -m app.train`

2. **Import errors**
   - Check that all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version compatibility (3.9+)

3. **Prediction errors**
   - Ensure input data matches the expected schema
   - Check that all required fields are provided
   - Verify data types match the API specification

## 📈 Model Performance

### Training Metrics
- **Model Size**: ~2MB (ONNX format)
- **Training Time**: ~1-2 minutes on standard hardware
- **Feature Count**: 74 features after preprocessing
- **Training Data**: 205 samples

### Inference Performance
- **Prediction Time**: ~10-50ms per sample
- **Memory Usage**: ~50MB for model loading
- **Format**: ONNX for optimized cross-platform inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔮 Future Enhancements

### Model Development
- [ ] Hyperparameter tuning and optimization
- [ ] Cross-validation and model evaluation metrics
- [ ] Additional ML algorithms comparison (XGBoost, Neural Networks)
- [ ] Feature importance analysis and selection
- [ ] Model explainability and SHAP values

### Data and Training
- [ ] Automated data validation and quality checks
- [ ] Support for larger datasets
- [ ] Online learning capabilities
- [ ] Data augmentation techniques
- [ ] Integration with external car databases

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the troubleshooting section

---

**Built with ❤️ using FastAPI, scikit-learn, and ONNX**