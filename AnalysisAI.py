import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import traceback
from io import StringIO
import re
import base64
import os

# Initialize OpenAI client with your NVIDIA API base URL and API key
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["api_key"]  # Store your API key in Streamlit secrets
    )

def dataset_to_string(df):
    """Convert a dataset to a string format suitable for the model."""
    data_sample = df.head().to_string()
    data_info = df.describe(include='all').to_string()
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

def create_eda_prompt(data_str):
    """Create a custom Model Training prompt for the language model."""
    return f"""
    **Role**: You are an expert data Scientist.

    **Dataset Overview**:
    - **Data Sample**:
      ```
      {df.head().to_string(}
      ```

    - **Data Description**:
      ```
      {df.describe(include='all').to_string()}
      ```

    I have provided you with a dataset containing various features. Your task is to perform comprehensive model training and evaluation. The dataset includes a mix of numerical and categorical features. Please create a Python function that:

    1. **Prompts for Target Column:**
       - Asks the user for the target column name.

    2. **Data Preparation:**
       - Separates features and the target variable from the dataset.
       - Drops irrelevant columns that do not contribute to the model's prediction.
       - Performs data cleaning, including handling missing values.

    3. **Feature Engineering and Preprocessing:**
       - Identifies numerical and categorical columns.
       - Applies preprocessing steps:
         - For numerical columns:
           - Imputes missing values with the mean.
           - Standardizes features.
         - For categorical columns:
           - Imputes missing values with the most frequent value.
           - Encodes categorical features using one-hot encoding or other appropriate encoding techniques.
       - Converts any remaining categorical or object columns to numeric values suitable for modeling.

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

    5. **Selection of the Best Model:**
       - Identifies and prints the best-performing model based on cross-validation accuracy.

    6. **Final Evaluation:**
       - Trains the best-performing model on the entire training set.
       - Evaluates the model on the test set.
       - Prints the confusion matrix, classification report, and accuracy score for the test set.
       - Prints the Best Model Name

    7. **Code Execution Verification:**
       - Ensure that the generated code is executable without errors by running it in the background. 
       - Verify that the code completes successfully and that no errors are encountered during execution.

    Ensure the function handles preprocessing, including dropping irrelevant columns and converting categorical columns to numeric, effectively. Evaluate model performance thoroughly to provide the best model and validate the code by executing it to ensure it runs without errors.
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
    st.title("Machine Learning Model Generator")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        if st.button("Generate ML Model"):
            data_str = dataset_to_string(df)
            eda_prompt = create_eda_prompt(data_str)

            client = get_openai_client()

            with st.spinner("Generating ML model code..."):
                completion = client.chat.completions.create(
                    model="meta/llama-3.1-8b-instruct",
                    messages=[{"role": "user", "content": eda_prompt}],
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

            st.subheader("Code Execution:")
            output_buffer = StringIO()
            error_occurred = False
            try:
                exec(processed_code, {
                    'pd': pd, 
                    'np': np, 
                    'plt': plt, 
                    'sns': sns, 
                    'print': lambda *args, **kwargs: print(*args, file=output_buffer, **kwargs),
                    'df': df  # Pass the dataframe to the executed code
                })
                st.success("Code executed successfully!")
                st.text(output_buffer.getvalue())
            except Exception as e:
                error_occurred = True
                st.error("An error occurred during code execution:")
                st.code(traceback.format_exc())
                st.warning("The generated code might need manual adjustments. Please review and modify as necessary.")

            # Save to Python file
            file_path = "ML_model_generated.py"
            with open(file_path, "w") as f:
                f.write(processed_code)
            st.success(f"Generated code saved to '{file_path}'")

            # Add download button
            st.markdown(get_binary_file_downloader_html(file_path, 'Generated Python File'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
