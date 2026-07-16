import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Project Creator", page_icon="🤖")

st.title("Machine Learning Project Creator")
st.write("Follow the steps below to configure and start your ML project.")

if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False

uploaded_file = None

if not st.session_state.setup_complete:
    st.subheader("Step 1: Setup Configuration")
    st.info("Please fill out your personal and project details before uploading a dataset.")
    
    with st.form("setup_form"):
        st.write("### 👤 Personal Details")
        col1, col2 = st.columns(2)
        with col1:
            author_name = st.text_input("Full Name *", "")
            organization = st.text_input("Organization / Company", "")
        with col2:
            email = st.text_input("Email Address", "")
            
        st.write("### 📁 Project Details")
        project_name = st.text_input("Project Name *", "My ML Project")
        project_desc = st.text_area("Project Description", "Briefly describe your project goals...")
        task_type = st.selectbox("Problem Type", ["Classification", "Regression", "Clustering", "Other"])
        data_source = st.text_input("Data Source URL (Optional)", "")
        
        st.write("* denotes mandatory fields")
        submitted = st.form_submit_button("Save & Continue", type="primary")
        
        if submitted:
            if not author_name.strip() or not project_name.strip():
                st.error("Please fill in all mandatory fields like Full Name and Project Name.")
            else:
                st.session_state.setup_complete = True
                st.session_state.author_name = author_name
                st.session_state.project_name = project_name
                st.session_state.task_type = task_type
                st.rerun()
else:
    task_type = st.session_state.task_type
    
    st.sidebar.header("Setup Summary")
    st.sidebar.markdown(f"**Author:** {st.session_state.author_name}")
    st.sidebar.markdown(f"**Project:** {st.session_state.project_name}")
    st.sidebar.markdown(f"**Task Type:** {task_type}")
    
    if st.sidebar.button("Edit Details"):
        st.session_state.setup_complete = False
        st.rerun()

    st.subheader("Step 2: Dataset Upload")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        st.success("File uploaded successfully!")
        
        # Basic Details Section
        st.subheader("Basic Details")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Duplicates", df.duplicated().sum())
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / 1024
            st.metric("Memory", f"{memory_usage:.1f} KB")
            
        num_cols = df.select_dtypes(include=['number']).shape[1]
        cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
        st.write(f"**Dataset Type:** Tabular dataset containing **{num_cols} numerical** and **{cat_cols} categorical** features.")

        # Display dataset overview
        st.subheader("Data Overview")
        st.write("First 5 rows of the dataset:")
        st.dataframe(df.head())
        
        st.subheader("Missing Values")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Feature', 'Missing Values']
        st.dataframe(missing_df)
            
        # Display columns and data types
        st.subheader("Column Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=['Data Type']).astype(str))
        
        # Display statistical summary
        st.subheader("Basic Statistical Description")
        st.dataframe(df.describe())
        
        # Next steps placeholder
        st.divider()
        st.subheader("Project Configuration (Basics)")
        st.write("Please fill in the basic details for your machine learning task:")
        
        col_target, col_features = st.columns(2)
        with col_target:
            target_col = st.selectbox("Select Target Variable (What to predict)", options=df.columns)
            
        with col_features:
            features = st.multiselect(
                "Select Feature Columns (Inputs)", 
                options=[c for c in df.columns if c != target_col], 
                default=[c for c in df.columns if c != target_col]
            )
            
        if target_col:
            st.success(f"Great! You are setting up a **{task_type}** model to predict **{target_col}** using **{len(features)}** features.")
            
        st.info("Next steps could include data preprocessing, model selection, and training.")
        
    except Exception as e:
        st.error(f"Error reading the file: {e}")