# MLStarterKit 🚀

A machine learning project template with configuration management via Hydra, scalable pipelines, and production-ready setup including AWS Elastic Beanstalk deployment.


## Features ✨

- **AWS Elastic Beanstalk Ready**: Pre-configured deployment files (Procfile, nginx, EB extensions)
- **Hydra Configuration**: Structured YAML configs for easy experiment management
- **Modular Pipelines**: Separate training & prediction pipelines
- **Template Structure**: Organized components for rapid development

## Installation 💻

```bash
# Clone repo
git clone [your-repo-url]
cd MLStarterKit

# Create virtual environment (Python 3.8+ recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

## Project Structure 🗂️
├── conf/               # Hydra configuration files
│   ├── config.yaml     # Main configuration
│   ├── model/          # Model hyperparameters
│   └── data/           # Data paths & preprocessing
├── components/         # Reusable pipeline components
├── pipeline/           # Core workflows
│   ├── train_pipeline.py
│   └── predict_pipeline.py
├── dto/                # Data transfer objects
├── src/                # Source utilities
│   ├── logger.py       # Custom logging
│   ├── exception.py    # Error handling
│   └── utils.py        # Helper functions
├── artifacts/          # Saved models & preprocessors
├── notebook/           # Exploration notebooks
├── app.py              # Flask application
├── requirements.txt    # Dependencies
└── Procfile            # Deployment configuration
