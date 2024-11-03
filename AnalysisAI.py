import streamlit as st
import pandas as pd
from openai import OpenAI
import traceback
from io import StringIO
import re
import base64
import os

# Initialize OpenAI client with a custom API key
@st.cache_resource
def get_openai_client(api_key):
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

def dataset_to_string(df):
    """Convert a dataset to a string format suitable for the model."""
    data_sample = df.head().to_string()
    data_info = df.describe(include='all').to_string()
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

def create_eda_prompt(data_str):
    """Create a custom Model Training prompt for the language model."""
    return f"""
   
    **Role**: You are an expert Data Scientist with a focus on interpretability and transparency.

    **Dataset Overview**:
    - **Data Sample**:
      ```
      {data_str.split('Data Description:')[0].strip()}
      ```

    - **Data Description**:
      ```
      {data_str.split('Data Description:')[1].strip()}
      ```

    I have provided you with a dataset containing various features. Your task is to perform comprehensive model training and evaluation. The dataset includes a mix of numerical and categorical features. Please create a Python function that performs the following tasks with detailed explanations:

    1. **Prompt for Target Column:**
       - Explain why it is important to specify a target column and how it affects model training.

    2. **Data Preparation:**
       - Separates features and the target variable from the dataset.
       - Drops irrelevant columns that do not contribute to the model's prediction.
       - Explain the criteria used for deciding which columns to drop and the impact of keeping or removing these columns.
       - Performs data cleaning, including handling missing values.
       - Explain the strategies used for handling missing values and their implications on the model.

    3. **Feature Engineering and Preprocessing:**
       - Identifies numerical and categorical columns.
       - Applies preprocessing steps:
         - For numerical columns:
           - Imputes missing values with the mean.
           - Standardizes features.
           - Explain why mean imputation is used and the benefits of standardization.
         - For categorical columns:
           - Imputes missing values with the most frequent value.
           - Encodes categorical features using one-hot encoding or other appropriate encoding techniques.
           - Explain the choice of encoding technique and the rationale for imputing missing values with the most frequent value.
       - Converts any remaining categorical or object columns to numeric values suitable for modeling.
       - Describe the method used for conversion and the importance of having numeric values for modeling.

    4. **Model Definition and Evaluation:**
       - Defines and evaluates the following models:
         - Logistic Regression
         - Decision Tree Classifier
         - Random Forest Classifier
         - Gradient Boosting Classifier
         - Support Vector Classifier (SVC)
         - K-Nearest Neighbors (KNN)
         - Naive Bayes (GaussianNB)
       - Uses cross-validation to evaluate each model's performance.
       - Fits each model with the preprocessed data (including any preprocessing steps such as scaling and encoding applied earlier).
       - Prints the mean cross-validation accuracy score for each model.
       - Explain the choice of models and the rationale behind using cross-validation for performance evaluation.

    5. **Selection of the Best Model:**
       - Identifies and prints the best-performing model based on cross-validation accuracy.
       - Explain how the best model is selected and why it is considered the best.

    6. **Final Evaluation:**
       - Trains the best-performing model on the entire training set.
       - Evaluates the model on the test set.
       - Prints the confusion matrix, classification report, and accuracy score for the test set.
       - Prints the Best Model Name
       - Explain the significance of each evaluation metric and how the final model's performance is assessed.

    7. **Code Execution Verification:**
       - Ensure that the generated code is executable without errors by running it in the background. 
       - Verify that the code completes successfully and that no errors are encountered during execution.

    Ensure that each step includes comments and explanations for the decisions made, particularly regarding data preprocessing, feature engineering, model selection, and evaluation. The goal is to provide transparency and clarity in the machine learning process, so the user can understand the rationale behind each decision.
    """

def preprocess_generated_code(code):
    # Remove any markdown code block indicators
    code = re.sub(r'```python|```', '', code)
    
    # Remove any explanatory text before the actual code
    code = re.sub(r'^.*?import', 'import', code, flags=re.DOTALL)
    
    # Replace triple quotes with double quotes
    code = code.replace("'''", '"""')
    
    # Ensure necessary imports are present
    if "import matplotlib.pyplot as plt" not in code:
        code = "import matplotlib.pyplot as plt\n" + code
    if "import seaborn as sns" not in code:
        code = "import seaborn as sns\n" + code
    
    return code.strip()

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    st.title("MLAutoGen")

    # Prompt the user to input their API key at the start of the app
    st.info("🎊 Its Free Again. Use This API Key : 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 🎊")
    api_key = st.text_input("Enter your NVIDIA API Key:", type="password")

    if not api_key:
        st.error("API Key is required to proceed.")
        return

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Prompt for target column
        target_column = st.text_input("Enter the target column name:")

        if st.button("Generate ML Model"):
            if not target_column or target_column not in df.columns:
                st.error("Please provide a valid target column name.")
                return

            data_str = dataset_to_string(df)
            eda_prompt = create_eda_prompt(data_str)

            # Update the prompt to include the target column
            eda_prompt_with_target = eda_prompt.replace(
                "Your task is to perform comprehensive model training and evaluation.",
                f"Your task is to perform comprehensive model training and evaluation with the target column '{target_column}'."
            )

            client = get_openai_client(api_key)

            try:
                with st.spinner("Generating ML model code..."):
                    completion = client.chat.completions.create(
                        model="meta/llama-3.1-8b-instruct",
                        messages=[{"role": "user", "content": eda_prompt_with_target}],
                        temperature=0.2,
                        top_p=0.7,
                        max_tokens=2048,
                        stream=True
                    )

                    generated_code = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            generated_code += chunk.choices[0].delta.content

                # Preprocess the generated code
                processed_code = preprocess_generated_code(generated_code)

                st.subheader("Generated Code:")
                st.code(processed_code)

                # Save to Python file
                file_path = "ML_model_generated.py"
                with open(file_path, "w") as f:
                    f.write(processed_code)
                st.success(f"Generated code saved to '{file_path}'")

                # Add download button
                st.markdown(get_binary_file_downloader_html(file_path, 'Generated Python File'), unsafe_allow_html=True)

                # Warning message about potential code adjustments
                st.warning("The generated code might contain minor errors or require slight adjustments.")

            except Exception as e:
                st.error("The API Key is invalid or credits are over. Please use a valid API Key.")
                st.info("You can get an free API key from here: [NVIDIA Meta LLaMA API Key](https://build.nvidia.com/meta/llama-3_1-405b-instruct)")

if __name__ == "__main__":
    main()
