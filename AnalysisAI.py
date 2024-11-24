import streamlit as st
import pandas as pd
import re
import base64
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import traceback

api_key = st.secrets["API_KEY"]

# Initialize NVIDIA LangChain client
@st.cache_resource
def get_nvidia_client():
     return ChatNVIDIA(
  model="meta/llama-3.2-3b-instruct",
  api_key=api_key, 
  temperature=0.2,
  top_p=0.7,
  max_tokens=4096,
)

def dataset_to_string(df):
    """Convert a dataset to a string format suitable for the model."""
    data_sample = df.head().to_string()
    data_info = df.describe(include='all').to_string()
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

def create_ml_training_prompt(data_str, target_column):
    """Create a detailed prompt for generating a machine learning pipeline."""
    return f"""
     **Role**: You are an expert Data Scientist and Machine Learning Engineer with a focus on interpretability, transparency, and robust model development.

    I have provided you with a dataset containing various features, including the target column '{target_column}'. Your task is to create a Python script for a comprehensive machine learning pipeline. The dataset includes a mix of numerical and categorical features. Please ensure all steps are well-documented and explain the rationale behind each decision.

    ### Dataset Overview
    - **Data Sample**:
      ```
      {data_str.split('Data Description:')[0].strip()}
      ```
    - **Data Description**:
      ```
      {data_str.split('Data Description:')[1].strip()}
      ```

    ### Tasks and Expectations

    **1. Prompt for Target Column:**
       - Explain why specifying a target column is critical and how it influences the machine learning workflow.

    **2. Data Preparation:**
       - Separate features and the target variable from the dataset.
       - Drop irrelevant or redundant columns based on statistical thresholds or domain knowledge.
       - Handle missing values with appropriate strategies:
         - For numerical features: Use mean or median imputation.
         - For categorical features: Use mode imputation or a placeholder value.
       - Standardize numerical features (e.g., `StandardScaler`) and encode categorical features (e.g., one-hot encoding or target encoding).
       - Clearly describe the impact of each preprocessing step on the dataset and model performance.

    **3. Feature Engineering and Selection:**
       - Identify numerical and categorical columns and apply preprocessing as needed.
       - Remove multicollinear features using correlation analysis or Variance Inflation Factor (VIF).
       - Use techniques like Recursive Feature Elimination (RFE) or Tree-based Feature Selection to reduce dimensionality.
       - Explain how feature engineering contributes to improved model performance.

    **4. Model Training and Evaluation:**
       - Train and evaluate the following models:
         - Logistic Regression
         - Decision Tree Classifier
         - Random Forest
         - Gradient Boosting (e.g., XGBoost, LightGBM, CatBoost)
         - Support Vector Machine (SVM)
         - K-Nearest Neighbors (KNN)
         - Multi-Layer Perceptron (MLP)
         - Naive Bayes
       - Use stratified K-fold cross-validation (e.g., 5-fold or 10-fold) for consistent performance evaluation.
       - Include hyperparameter tuning for advanced models using Grid Search or Random Search.
       - Explain why specific models are included and the role of cross-validation in mitigating overfitting.

    **5. Performance Metrics and Visualization:**
       - Report and visualize key metrics for each model:
         - Classification: Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix.
         - Regression (if applicable): Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared.
       - Plot feature importance for interpretable models.
       - Provide a detailed explanation of each metric and its significance in evaluating model performance.

    **6. Model Comparison and Selection:**
       - Compare all trained models and identify the best-performing one based on evaluation metrics.
       - Justify the selection of the final model with supporting metrics and visualizations.

    **7. Final Model Training and Deployment:**
       - Train the selected model on the entire training dataset.
       - Evaluate the model on a separate test set, reporting metrics and insights.
       - Save the trained model using `pickle` or `joblib` for deployment.
       - Include instructions for loading the saved model and making predictions.

    **8. Code Verification and Usability:**
       - Ensure the generated code is modular, executable, and free of errors.
       - Add detailed comments for each step to make the script user-friendly and explainable, even for non-technical users.

    ### Additional Guidelines
    - Use Python libraries like `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn` to implement the solution.
    - Avoid unnecessary dependencies, and prioritize computational efficiency.
    - Ensure all functions and logic are modular and reusable.

    Provide the complete Python script for this pipeline, ready for execution. Each step should include sufficient comments, detailed explanations, and visualizations to enhance interpretability and usability.
    """

def preprocess_generated_code(code):
    """Preprocess the code generated by the language model."""
    code = re.sub(r'```python|```', '', code)
    code = code.replace("'''", '"""')
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
    st.title("MLAutoGen: Advanced Machine Learning Model Trainer")

    # Feedback Section using Google Form
    st.sidebar.subheader("I appreciate your feedback.")
    st.sidebar.markdown("""
    <a href="https://forms.gle/rTrFC4rwqfJ9B6mE9" target="_blank">
        <button style="
            background-color: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 14px; 
            margin: 4px 2px; 
            cursor: pointer;
            border: none;
            border-radius: 8px;
        ">
            Submit Feedback
        </button>
    </a>
    """, unsafe_allow_html=True)

    # File Upload Section
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        target_column = st.text_input("Enter the target column name:")
        if st.button("Generate ML Models"):
            if not target_column or target_column not in df.columns:
                st.error("Please provide a valid target column name.")
                return

            data_str = dataset_to_string(df)
            ml_prompt = create_ml_training_prompt(data_str, target_column)

            client = get_nvidia_client()

            try:
                with st.spinner("Generating ML model code..."):
                    generated_code = ""
                    for chunk in client.stream([{"role": "user", "content": ml_prompt}]):
                        if chunk.content:
                            generated_code += chunk.content

                processed_code = preprocess_generated_code(generated_code)

                st.subheader("Generated Code:")
                st.code(processed_code)

                # Save the generated code for download
                file_path = "ML_model_generated.py"
                with open(file_path, "w") as f:
                    f.write(processed_code)
                st.success(f"Generated code saved to '{file_path}'")

                # Provide download link
                st.markdown(get_binary_file_downloader_html(file_path, 'Generated Python File'), unsafe_allow_html=True)

                st.warning("The generated code might require minor adjustments before execution.")

            except Exception as e:
                st.error("An error occurred during code generation.")
                st.exception(traceback.format_exc())

if __name__ == "__main__":
    main()
