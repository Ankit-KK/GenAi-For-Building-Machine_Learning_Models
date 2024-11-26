import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import traceback

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["api_key"]
    )

# Convert dataset to a formatted string
def dataset_to_string(df):
    try:
        data_sample = df.head().to_string()
        data_info = df.describe(include='all').to_string()
    except Exception as e:
        st.error(f"Error processing dataset: {e}")
        return ""
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

# Generate an ML training pipeline prompt
def create_ml_training_prompt(data_str, target_column):
    return f"""
    **Role**: You are an expert Data Scientist and Machine Learning Engineer with a focus on interpretability, transparency, and robust model development.

    I have provided you with a dataset containing various features, including the target column '{target_column}'. Your task is to create a Python script for a comprehensive machine learning pipeline. The dataset includes a mix of numerical and categorical features. Please ensure all steps are well-documented and explain the rationale behind each decision.

    ### Dataset Overview:
    - **Data Sample:**
      ```
      {data_str.split('Data Description:')[0].strip()}
      ```

    - **Data Description:**
      ```
      {data_str.split('Data Description:')[1].strip()}
      ```

    ### Tasks and Expectations:

    **1. Data Preparation:**
       - Handle missing values, encode categorical features, and scale numerical features.

    **2. Feature Engineering:**
       - Identify and remove multicollinear features.
       - Select features using advanced techniques like RFE.

    **3. Model Training and Evaluation:**
       - Train multiple models, perform hyperparameter tuning, and evaluate metrics like accuracy, precision, recall, and F1-score.

    **4. Model Deployment:**
       - Save the best model for deployment using `pickle` or `joblib`.

    **Output Requirements:**
    - Python code for each step with detailed comments.
    - Use libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.
    """

# Preprocess the generated code
def preprocess_generated_code(code):
    code = re.sub(r'```python|```', '', code)
    return code.strip()

# Main Streamlit app function
def main():
    st.title("MLAutoGen: Machine Learning Pipeline Generator")

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

    client = get_openai_client()

    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        target_column = st.text_input("Enter the target column name:")
        if st.button("Generate ML Pipeline Code"):
            if not target_column or target_column not in df.columns:
                st.error("Please provide a valid target column name.")
                return

            data_str = dataset_to_string(df)
            if not data_str:
                return

            ml_prompt = create_ml_training_prompt(data_str, target_column)

            try:
                with st.spinner("Generating ML pipeline code..."):
                    completion = client.chat.completions.create(
                        model="meta/llama-3.2-3b-instruct",
                        messages=[{"role": "user", "content": ml_prompt}],
                        temperature=0.5,
                        top_p=0.9,
                        max_tokens=2048,
                        stream=True
                    )

                    generated_code = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            generated_code += chunk.choices[0].delta.content

                processed_code = preprocess_generated_code(generated_code)
                st.subheader("Generated ML Pipeline Code:")
                st.code(processed_code)

                # Provide download option
                file_path = "ml_pipeline_generated.py"
                with open(file_path, "w") as f:
                    f.write(processed_code)

                with open(file_path, "r") as f:
                    st.download_button("Download Generated Code", f, file_name="ml_pipeline_generated.py", mime="text/plain")

            except Exception as e:
                st.error(f"Error generating ML pipeline code: {e}")
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
