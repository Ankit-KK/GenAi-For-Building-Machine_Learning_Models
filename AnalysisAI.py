import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import traceback  # For detailed error reporting
import openai
import fpdf



# Initialize OpenAI client with your NVIDIA API base URL and API key
# Initialize OpenAI client with your NVIDIA API base URL and API key
api_key = st.secrets["api_key"]  # Store your API key in Streamlit secrets
client = openai.OpenAI(
   base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


def dataset_to_string(df):
    """Convert a dataset to a string format suitable for the model."""
    data_sample = df.head().to_string()
    data_info = df.describe(include='all').to_string()
    return f"Data Sample:\n{data_sample}\n\nData Description:\n{data_info}"

def create_eda_prompt(df):
    """Create a custom EDA prompt for the language model."""
    data_str = dataset_to_string(df)
    return f"""
**Role**: You are an expert data analyst.

**Context**: I have provided you with a dataset containing various features. Your task is to perform a comprehensive exploratory data analysis (EDA) to uncover insights, patterns, and potential issues in the data. The dataset includes a mix of numerical and categorical features, and it is crucial to explore these thoroughly to inform further analysis or decision-making.

**Dataset Overview**:
- **Data Sample**:
  ```
  {df.head().to_string()}
  ```

- **Data Description**:
  ```
  {df.describe(include='all').to_string()}
  ```

**Tasks**:

1. **Data Overview**:
   - Print "Executing Task 1: Data Overview"
   - Inspect the first few rows of the dataset to understand its structure.
   - Determine the data types of each column (numerical, categorical, etc.).
   - Check for missing values and describe the proportion of missing data in each column.

2. **Descriptive Statistics**:
   - Print "Executing Task 2: Descriptive Statistics"
   - Calculate summary statistics (mean, median, mode, standard deviation, variance, minimum, maximum) for each numerical column.
   - Provide insights on the distribution of these numerical features (e.g., skewness, kurtosis).

3. **Data Visualization**:
   - Print "Executing Task 3: Data Visualization"
   - Plot histograms and density plots for each numerical column to visualize distributions.
   - Create scatter plots to examine relationships between key numerical variables (e.g., feature vs. target variable).
   - Use box plots to identify outliers and understand the spread of the data.

4. **Categorical Data Analysis**:
   - Print "Executing Task 4: Categorical Data Analysis"
   - Summarize the frequency of each category within categorical columns.
   - Use bar plots or count plots to visualize the distribution of categorical variables.
   - Analyze the relationship between categorical variables and the target variable (if applicable), using grouped bar charts or other appropriate visualizations.

5. **Correlation Analysis**:
   - Print "Executing Task 5: Correlation Analysis"
   - Compute the correlation matrix for numerical features.
   - Visualize the correlation matrix using a heatmap and identify pairs of highly correlated features.
   - Discuss potential implications of multicollinearity and suggest strategies for dealing with it.

6. **Advanced Analysis**:
   - Print "Executing Task 6: Advanced Analysis"
   - **Handle Missing Values:**
       - Check for missing values in the dataset.
       - If missing values are present:
           - Choose an appropriate strategy (e.g., imputation, dropping rows/columns) based on the type and extent of missingness.
           - Explain the rationale behind the chosen strategy and its potential impact on the analysis.
           - Implement the chosen strategy to handle missing values.
   - Perform clustering (e.g., K-means) or dimensionality reduction (e.g., PCA) on the preprocessed data to uncover patterns or groupings in the data.
   - Identify any anomalies or unusual patterns that might warrant further investigation.

7. **Insights and Recommendations**:
   - Print "Executing Task 7: Insights and Recommendations"
   - Summarize the key findings from the EDA, highlighting significant patterns, trends, or anomalies.
   - Provide actionable insights based on the analysis, such as data cleaning steps, feature engineering ideas, or further analyses that could be conducted.
   - Suggest potential next steps, including any additional data that may be required or further analyses that could enhance understanding.

**Instructions for Model**:
- Provide Python code snippets for each task, ensuring that the code is efficient, well-commented, and easy to understand.
- Include print statements before each task to indicate which task is being executed.
- Execute the code snippets where necessary to validate the findings and ensure there are no errors.
- If any assumptions are made during the analysis, clearly state them and explain their rationale.

**Output**:
- The analysis should be comprehensive and thorough, providing clear and actionable insights based on the data.
- Include any visualizations as part of the output to support the findings and provide a clear understanding of the data.
"""

def main():
    st.title("Exploratory Data Analysis with Streamlit")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display dataset
        st.write("### Dataset")
        st.write(df.head())
        
        # Create EDA prompt
        eda_prompt = create_eda_prompt(df)
        
        # Generate code using the language model
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": eda_prompt}],
            temperature=0.5,
            top_p=0.7,
            max_tokens=2048
        )
        
        generated_code = ""
        for chunk in completion.choices:
            if chunk.delta.content is not None:
                generated_code += chunk.delta.content
        
        # Preprocess generated code (handle potential errors)
        generated_code = generated_code.replace("'''", "\"\"\"")
        generated_code = generated_code.replace("''", "\"")
        if not "import matplotlib.pyplot as plt" in generated_code:
            generated_code = "import matplotlib.pyplot as plt\n" + generated_code
        if not "import seaborn as sns" in generated_code:
            generated_code = "import seaborn as sns\n" + generated_code
        
        # Display generated code
        st.write("### Generated Code")
        st.code(generated_code, language='python')
        
        # Execute the generated code and capture output
        output_buffer = StringIO()
        error_occurred = False
        try:
            exec(generated_code, {'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'print': lambda *args, **kwargs: print(*args, file=output_buffer, **kwargs)})
        except Exception as e:
            error_occurred = True
            error_traceback = traceback.format_exc()
            st.error(f"Error occurred: {error_traceback}")
        
        if not error_occurred:
            # Save results to a file
            analysis_results = output_buffer.getvalue()
            with open("analysis_results.txt", "w") as file:
                file.write(analysis_results)
            
            # Provide a download link
            with open("analysis_results.txt", "rb") as file:
                st.download_button(
                    label="Download Analysis Results",
                    data=file,
                    file_name="analysis_results.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
