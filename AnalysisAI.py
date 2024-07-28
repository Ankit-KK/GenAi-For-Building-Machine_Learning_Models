import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import traceback
from io import StringIO

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
      {data_str.split('Data Description:')[0]}
      ```

    - **Data Description**:
      ```
      {data_str.split('Data Description:')[1]}
      ```

    I have provided you with a dataset containing various features. Your task is to perform comprehensive model training and evaluation. The dataset includes a mix of numerical and categorical features. Please create a Python function that:

    [Rest of your prompt here...]
    """

def main():
    st.title("Machine Learning Model Generator")

    # File uploader
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

            st.subheader("Generated Code:")
            st.code(generated_code)

            st.subheader("Code Execution:")
            output_buffer = StringIO()
            error_occurred = False
            try:
                exec(generated_code, {
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
                st.error(f"An error occurred during code execution:\n{traceback.format_exc()}")

            # Save to Python file
            with open("ML_model_generated.py", "w") as f:
                f.write(generated_code)
            st.success("Generated code saved to 'ML_model_generated.py'")

if __name__ == "__main__":
    main()
