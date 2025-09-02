# 🏥 ChatDoctor - AI-Powered Health Assistant

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15.10-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

*An intelligent medical chatbot that combines machine learning and AI to provide personalized health assistance*

**🏗️ Developed in Hong Kong, April 2024**

</div>

## 📅 Project Timeline & Important Notes

This project was developed in **Hong Kong during April 2024**, when OpenAI's API was still directly accessible from Hong Kong. 

> **⚠️ Important**: As of late 2024, OpenAI API access from Hong Kong has been restricted. If you're running this project from Hong Kong or other restricted regions, you may need to:
> - Use a VPN service to access OpenAI API
> - Consider alternative API providers
> - Use only the local Tree Model mode (which works offline)

The local Tree Model provides full functionality without requiring external API access.

## 🌟 Features

- **🤖 Dual AI System**: Choose between local ML models (Random Forest/SVM) and OpenAI GPT-3.5-turbo
- **🎯 Disease Prediction**: Symptom-based disease prediction using trained machine learning models
- **💊 Medicine Information**: Comprehensive medicine database with usage information
- **🖥️ Modern GUI**: Beautiful PyQt5 interface with web-based chat display
- **🔍 Smart Search**: Fuzzy matching for symptom recognition and spell correction
- **📊 Model Visualization**: Jupyter notebook with model accuracy analysis
- **🔧 RAG Support**: Retrieval-Augmented Generation for enhanced responses (optional)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda package manager
- OpenAI API key (for API mode)

### Installation

1. **Clone the repository**
   ```bash
   # Replace 'your-username' with your GitHub username if you have forked the repository
   git clone https://github.com/your-username/medical_chatbot.git
   cd medical_chatbot
   ```

2. **Create and activate conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate medical_chatbot
   ```

3. **Set up OpenAI API key** (Required for API mode)
   
   Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```
   
   > **🌏 Regional Note**: If you're in Hong Kong or other restricted regions, consider using the Tree Model mode instead, which works completely offline.

4. **Run the application**
   ```bash
   python main.py
   ```

> **Note**: First loading may take around 1 minute, but subsequent loads will be faster (< 10 seconds)

## 🎮 Usage

### Getting Started

1. Launch the application using `python main.py`
2. Choose your preferred model:
   - **Tree Model**: Local ML-based prediction (faster, offline)
   - **API Model**: OpenAI GPT-3.5-turbo (more conversational, requires internet)
3. Start chatting about your health concerns!

### Model Modes

#### 🌳 Tree Model (Local ML)
- Uses Random Forest and SVM classifiers
- Trained on medical datasets
- Provides disease predictions based on symptoms
- Works offline
- Includes medicine information lookup

#### 🤖 API Model (OpenAI)
- Powered by GPT-3.5-turbo
- More conversational and context-aware
- Requires OpenAI API key
- Online connectivity required

## 📁 Project Structure

```
├── main.py                              # Application entry point
├── gui.py                               # PyQt5 GUI implementation
├── middleware.py                        # Model routing and logic
├── tree_model_medicine.py              # Local ML disease prediction
├── api_model.py                         # OpenAI API integration
├── disease_prediction_model_generator.py # Model training script
├── inference_model_training.ipynb       # Model analysis and visualization
├── environment.yml                      # Conda environment configuration
├── .env                                 # Environment variables (API keys)
├── data/                                # Training and reference data
│   ├── training.csv                     # ML training dataset
│   ├── symptom_Description.csv          # Disease descriptions
│   ├── symptom_precaution.csv          # Medical precautions
│   ├── medicine_use.csv                 # Medicine database
│   ├── fuzzy_dictionary_unique.txt      # Fuzzy matching dictionary
│   └── rag_data/                        # RAG-specific data
├── model/                               # Trained ML models
│   ├── rfc.model                        # Random Forest model
│   └── svc.model                        # SVM model
└── RAG_code/                            # RAG implementation (optional)
    ├── api_model_rag.py                 # RAG-enhanced API model
    └── middleware_rag.py                # RAG middleware
```

## 🔧 Technical Details

### Machine Learning Models

- **Random Forest Classifier**: Primary prediction model
- **Support Vector Machine**: Secondary prediction model
- **Training Data**: Kaggle medical dataset with symptom-disease mappings
- **Features**: 132 symptom features for disease classification

### Key Components

| Component | Description |
|-----------|-------------|
| `main.py` | Application launcher with PyQt5 setup |
| `gui.py` | Complete GUI implementation with chat interface |
| `middleware.py` | Smart routing between Tree and API models |
| `tree_model_medicine.py` | ML-based disease prediction engine |
| `api_model.py` | OpenAI GPT integration with conversation history |

### Data Sources

- **Training Dataset**: Comprehensive symptom-disease mapping
- **Medicine Database**: 17MB database with medicine information
- **Symptom Descriptions**: Detailed disease information
- **Precautions**: Medical advice and precautionary measures

## 🖥️ Platform Compatibility

- **macOS**: Optimal experience (recommended)
- **Windows**: Fully functional with minor GUI sizing differences
- **Linux**: Compatible (requires Qt5 dependencies)

## 🔬 Model Performance

The project includes detailed model analysis in `inference_model_training.ipynb` showing:
- Accuracy comparisons between different algorithms
- Feature importance analysis
- Cross-validation results
- Performance metrics visualization

## 🚧 Optional Features

### RAG (Retrieval-Augmented Generation)

The `RAG_code/` directory contains enhanced versions with retrieval-augmented generation:
- `api_model_rag.py`: RAG-enhanced OpenAI integration
- `middleware_rag.py`: RAG-specific middleware

> **Note**: RAG features are separate to maintain cost efficiency and performance in the main application.

## 🛠️ Development

### Training New Models

To retrain the machine learning models:

```bash
python disease_prediction_model_generator.py
```

This will generate new `rfc.model` and `svc.model` files in the `model/` directory.

### Model Analysis

Open `inference_model_training.ipynb` in Jupyter to:
- Analyze model performance
- Visualize accuracy metrics
- Compare different algorithms
- Explore feature importance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

## 🙏 Acknowledgments

- Medical datasets from Kaggle
- OpenAI for GPT-3.5-turbo API
- PyQt5 for the GUI framework
- scikit-learn for machine learning models

---

<div align="center">
Made with ❤️ for better healthcare accessibility
</div>

