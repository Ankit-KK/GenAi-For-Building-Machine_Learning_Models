```markdown
# AnalysisAI

## Overview
AnalysisAI is an interactive Streamlit app that leverages OpenAI's API
to generate machine learning model code from a dataset.
This tool is designed for data scientists and analysts looking
to quickly prototype models with interpretability and transparency.

## Features
- **Streamlit Interface**: Easy-to-use interface for uploading datasets and specifying target columns.
- **OpenAI Integration**: Automatically generates Python code for data preparation, model training, and evaluation.
- **Custom Prompts**: Incorporates best practices and detailed explanations in the generated code.
- **Code Download**: Download the generated Python code for further customization and use.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ankit-kk/AnalysisAI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd AnalysisAI
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run AnalysisAI.py
   ```
2. Upload your dataset and specify the target column.
3. Review and download the generated machine learning model code.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [OpenAI](https://www.openai.com/)
- [Streamlit](https://www.streamlit.io/)
