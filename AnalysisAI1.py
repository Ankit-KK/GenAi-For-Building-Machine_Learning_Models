import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from openai import OpenAI
import traceback  # For detailed error reporting

# Initialize OpenAI client with your NVIDIA API base URL and API key
api_key = st.secrets["api_key"]  # Store your API key in Streamlit secrets
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=api_key  # Replace with your actual API key
)

@st.cache(show_spinner=False)
def dataset_to_string(df):
    """Convert a dataset to a string format suitable for the model."""
    data_sample = df.head().to_string()
    data_info = df.describe(include='all').to_string()
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

@st.cache(show_spinner=False)
def create_eda_prompt(data_str):
    """Create a custom Model Training prompt for the language model."""
    return f"""





**Role**: You are an expert data Scientist.

**Dataset Overview**:
- **Data Sample**:
  ```
  {df.head().to_string()}
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

**4. Model Definition and Evaluation:**

- Defines and evaluates the following models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes (GaussianNB)
- Uses cross-validation to evaluate each model’s performance.
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

def execute_generated_code(generated_code):
    output_buffer = StringIO()
    error_occurred = False
    try:
        exec(generated_code, {'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'print': lambda *args, **kwargs: print(*args, file=output_buffer, **kwargs)})
    except Exception as e:
        error_occurred = True
        error_traceback = traceback.format_exc()
    
    analysis_results = output_buffer.getvalue()
    return analysis_results, error_occurred, error_traceback

def main():
    st.title('Machine Learning Model Generator')
    
    # Upload dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display dataset information
        st.write('Dataset Information:')
        data_str = dataset_to_string(df)
        st.text_area(label='Data Sample and Description', value=data_str, height=300)
        
        # Generate EDA prompt
        eda_prompt = create_eda_prompt(data_str)
        st.write('Generated EDA Prompt:')
        st.text_area(label='EDA Prompt', value=eda_prompt, height=300)
        
        # Generate and execute code
        if st.button('Generate and Execute ML Model Code'):
            completion = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": eda_prompt}],
                temperature=0.5,
                top_p=0.7,
                max_tokens=2048,
                stream=True
            )
            
            generated_code = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    generated_code += chunk.choices[0].delta.content
            
            # Preprocess generated code
            generated_code = generated_code.replace("'''", "\"\"\"")
            generated_code = generated_code.replace("''", "\"")
            if not "import matplotlib.pyplot as plt" in generated_code:
                generated_code = "import matplotlib.pyplot as plt\n" + generated_code
            if not "import seaborn as sns" in generated_code:
                generated_code = "import seaborn as sns\n" + generated_code
            
            # Execute the generated code
            analysis_results, error_occurred, error_traceback = execute_generated_code(generated_code)
            
            if error_occurred:
                st.error(f"Error occurred during execution:\n{error_traceback}")
            else:
                st.write('Generated Code Output:')
                st.text_area(label='Output', value=analysis_results, height=300)
                st.success("Code executed successfully!")

if __name__ == "__main__":
    main()