import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.decomposition import PCA

# Define a color map for churn prediction categories
color_map = {
    0: '#1f77b4',  # Blue for "No Churn"
    1: '#ff7f0e'   # Orange for "Churn"
}

# Load the model and scaler from files
with open('modelo_gradient_boosting.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

# Load datasets
df_train = pd.read_csv('supply_chain_train.csv')
df_test = pd.read_csv('supply_chain_test.csv')

# Define columns
categorical_columns = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
columns_to_normalize = ['Customer_Age', 'Dependent_count', 'Months_on_book', 
                        'Total_Relationship_Count', 'Months_Inactive_12_mon',
                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# Ensure 'test_idx' column exists
if 'CLIENTNUM' not in df_test.columns:
    st.write("The 'CLIENTNUM' column is not present in the test data.")
    st.stop()

# Preprocessing function
def preprocess_for_prediction(df):
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    model_features = loaded_model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]
    df[columns_to_normalize] = loaded_scaler.transform(df[columns_to_normalize])
    return df

# Initialize session state
if 'selected_customers_df' not in st.session_state:
    st.session_state.selected_customers_df = pd.DataFrame()

# Configure the page
st.set_page_config(page_title="Customer Churn Analysis - Bank", layout="wide", initial_sidebar_state="expanded")

# Sidebar with logo
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.header("Customer Churn Analysis")
st.sidebar.subheader("Navigation")
selection = st.sidebar.radio("Go to", ["Home", 
                                      "Test Data Visualization", "Churn Prediction", 
                                      "User Activity Comparison", "Interactive Analysis", 
                                      "Additional Visualizations", "Customer Simulation"])

# Application title
st.title("Customer Churn Analysis - Bank")

image_path = "score.png" 

# Home
if selection == "Home":
    st.header("Welcome to the Customer Churn Analysis Application")
    st.markdown(f"""
    This application helps you analyze customer churn risk in a bank. It provides tools for understanding and predicting which customers are likely to leave the bank based on their behaviors and characteristics.
    
    ### Utility of the Application
    The application is designed to:
    - **Predict Churn**: Estimate the likelihood that a customer will stop using the bank's services based on their attributes and past behavior.
    - **Visualize Data**: Offer visual insights into the distribution of customer characteristics and churn predictions.
    - **Compare User Activity**: Compare the activity levels of selected customers to identify patterns and differences.
    - **Interactive Analysis**: Explore relationships between different features and their association with churn.
    - **Customer Simulation**: Allow users to input details for a new customer and predict their churn risk.

    ### Methods Used
    - **Predictive Modeling**:
      - **Gradient Boosting**: A powerful machine learning model that combines multiple decision trees to improve prediction accuracy. It is used to estimate the probability of churn based on customer data.
    - **Data Preprocessing**:
      - **Normalization**: Scaling numerical features to ensure they contribute equally to the model's performance.
      - **Categorical Encoding**: Converting categorical variables into binary format to be used by the model.
      - **Feature Alignment**: Ensuring that the input data includes all features required by the model, adding missing ones with zero values if necessary.
    - **Visual Analysis**:
      - **Feature Distributions**: Histograms and bar charts to show how different features are distributed in the data.
      - **Correlation Analysis**: Heatmaps to explore relationships between features and their correlation with churn.
      - **PCA**: Principal Component Analysis to reduce data dimensionality and visualize the separation between churn and non-churn customers in a 2D space.
    - **Interactive Analysis**:
      - **Scatter Plots**: Explore feature relationships and their association with churn predictions.
      - **Comparison of Selected Users**: Analyze and compare selected customer activity and attributes.
    - **Customer Simulation**:
      - **Manual Input**: Enter details for a new customer to simulate their churn risk prediction.

    This application provides a comprehensive view of customer behavior and helps make informed decisions to improve customer retention.
    """)


# Training Data Visualization
elif selection == "Training Data Visualization":
    st.header("Overview of Training Data")
    st.write("Explore the dataset used to train the model and understand the distribution of features.")
    st.dataframe(df_train)

    st.subheader("Feature Distributions")
    for feature in columns_to_normalize:
        fig = px.histogram(df_train, x=feature, title=f'Distribution of {feature}', color_discrete_sequence=['#1f77b4'])
        fig.update_layout(xaxis_title=feature, yaxis_title='Frequency', title_font_size=24)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Feature Distributions")
    for feature in categorical_columns:
        if feature in df_train.columns:
            fig = px.bar(df_train[feature].value_counts().reset_index(), x='index', y=feature,
                         title=f'Distribution of {feature}', labels={'index': feature, feature: 'Count'},
                         color_discrete_sequence=['#1f77b4'])
            fig.update_layout(xaxis_title=feature, yaxis_title='Count', title_font_size=24)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Analysis")
    # Filter out non-numeric columns
    numeric_df = df_train.select_dtypes(include=['number'])
    
    # Create heatmap using seaborn and matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=24)
    
    # Convert matplotlib figure to streamlit-compatible format
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, use_column_width=True)
    buf.close()
    plt.close(fig)

    st.subheader("Interactive Correlation Matrix")
    # Create an interactive heatmap using Plotly
    fig = ff.create_annotated_heatmap(
        z=numeric_df.corr().values,
        x=list(numeric_df.corr().columns),
        y=list(numeric_df.corr().columns),
        colorscale='coolwarm',
        showscale=True
    )
    fig.update_layout(title='Interactive Correlation Matrix', title_font_size=24)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Analysis")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(numeric_df.fillna(0))
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Churn Prediction'] = df_train['Churn Prediction']

    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Churn Prediction',
                     color_discrete_map=color_map, title='PCA of Training Data')
    fig.update_layout(xaxis_title='PC1', yaxis_title='PC2', title_font_size=24)
    st.plotly_chart(fig, use_container_width=True)

# Test Data Visualization
elif selection == "Test Data Visualization":
    st.header("Overview of Test Data")
    st.write("Explore the dataset used for making predictions.")
    st.dataframe(df_test)

    st.subheader("Feature Distributions")
    for feature in columns_to_normalize:
        fig = px.histogram(df_test, x=feature, title=f'Distribution of {feature}', color_discrete_sequence=['#1f77b4'])
        fig.update_layout(xaxis_title=feature, yaxis_title='Frequency', title_font_size=24)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Feature Distributions")
    for feature in categorical_columns:
        if feature in df_test.columns:
            fig = px.bar(df_test[feature].value_counts().reset_index(), x='index', y=feature,
                         title=f'Distribution of {feature}', labels={'index': feature, feature: 'Count'},
                         color_discrete_sequence=['#1f77b4'])
            fig.update_layout(xaxis_title=feature, yaxis_title='Count', title_font_size=24)
            st.plotly_chart(fig, use_container_width=True)

# Churn Prediction
elif selection == "Churn Prediction":
    st.header("Churn Prediction")

    input_idx = st.text_input("Enter the CLIENTNUM of the customer:", "")

    if input_idx:
        try:
            test_idx_value = int(input_idx)
            selected_row = df_test[df_test['CLIENTNUM'] == test_idx_value]

            if not selected_row.empty:
                st.write(f"Details for customer with CLIENTNUM {test_idx_value}:")
                st.dataframe(selected_row)

                selected_row_for_prediction = selected_row.drop(columns=['CLIENTNUM'])
                selected_row_for_prediction = preprocess_for_prediction(selected_row_for_prediction)

                if st.button("Predict Churn"):
                    prediction = loaded_model.predict(selected_row_for_prediction)
                    message = 'CHURN' if prediction[0] == 1 else 'NO CHURN'
                    color = 'red' if prediction[0] == 1 else 'green'
                    st.markdown(f"<h2 style='color: {color};'>{message}</h2>", unsafe_allow_html=True)
                    

                    st.subheader("Visual Analysis")
                    for feature in columns_to_normalize:
                        fig = px.histogram(df_train, x=feature, title=f'Distribution of {feature}', color_discrete_sequence=['#1f77b4'])
                        fig.update_layout(xaxis_title=feature, yaxis_title='Frequency', title_font_size=24)
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.write("Customer with the given CLIENTNUM not found.")
        except ValueError:
            st.write("Please enter a valid CLIENTNUM.")

# User Activity Comparison
elif selection == "User Activity Comparison":
    st.header("Compare User Activity")

    profile_df = df_test.copy()  # Use df_test for comparison

    select, compare = st.tabs(["Select Customers", "Compare Selected"])

    with select:
        st.header("All Customers")
        event = st.dataframe(profile_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="multi-row")

        st.header("Selected Customers")
        people = event.selection.rows
        if len(people) > 0:
            selected_customers_df = profile_df.iloc[people].copy()
            selected_customers_df = preprocess_for_prediction(selected_customers_df)
            selected_customers_df['Churn Prediction'] = loaded_model.predict(selected_customers_df)
            selected_customers_df.reset_index(drop=True, inplace=True)  # Reset index
            st.session_state.selected_customers_df = selected_customers_df
            st.dataframe(selected_customers_df.style.highlight_max(axis=0), use_container_width=True)
        else:
            st.write("No customers selected.")

    with compare:
        if not st.session_state.selected_customers_df.empty:
            selected_customers_df = st.session_state.selected_customers_df

            # Validate available columns
            available_columns = selected_customers_df.columns.tolist()
            categorical_columns = [col for col in categorical_columns if col in available_columns]
            columns_to_normalize = [col for col in columns_to_normalize if col in available_columns]

            st.subheader("Comparison of Total Transaction Amount")
            if 'test_idx' in selected_customers_df.columns:
                fig = px.bar(selected_customers_df, x='test_idx', y='Total_Trans_Amt', color='Churn Prediction',
                             labels={'Total_Trans_Amt': 'Total Transaction Amount', 'Churn Prediction': 'Churn Prediction'},
                             title='Total Transaction Amount by Churn Prediction', color_discrete_map=color_map)
                fig.update_layout(xaxis_title='Customer Number', yaxis_title='Total Transaction Amount', title_font_size=24)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Column 'test_idx' not found in the selected customers DataFrame.")

            st.subheader("Customer Age by Churn Prediction")
            fig = px.box(selected_customers_df, x='Churn Prediction', y='Customer_Age',
                         labels={'Customer_Age': 'Customer Age'},
                         title='Customer Age by Churn Prediction', color_discrete_map=color_map)
            fig.update_layout(xaxis_title='Churn Prediction', yaxis_title='Customer Age', title_font_size=24)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No customers selected for comparison.")

# Interactive Analysis
elif selection == "Interactive Analysis":
    st.header("Interactive Analysis")
    st.write("Explore relationships between different features and churn prediction.")
    st.write("Select the features you want to analyze and compare their distributions.")

    # Preprocess test data
    selected_data = preprocess_for_prediction(df_test.copy())
    selected_data['Churn Prediction'] = loaded_model.predict(selected_data)

    # Update options based on available columns after preprocessing
    x_axis = st.selectbox("Select X Axis", options=[col for col in columns_to_normalize if col in selected_data.columns])
    y_axis = st.selectbox("Select Y Axis", options=[col for col in columns_to_normalize if col in selected_data.columns])
    color_by = st.selectbox("Color by", options=[col for col in categorical_columns if col in selected_data.columns] + ["Churn Prediction"])

    # Create scatter plot
    fig = px.scatter(selected_data, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} vs {x_axis}",
                     color_discrete_map=color_map if color_by == "Churn Prediction" else None)
    fig.update_layout(title_font_size=24, xaxis_title=x_axis, yaxis_title=y_axis)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pairwise Correlation")
    feature_pair_x = st.selectbox("Select Feature for X Axis", options=[col for col in columns_to_normalize if col in selected_data.columns])
    feature_pair_y = st.selectbox("Select Feature for Y Axis", options=[col for col in columns_to_normalize if col in selected_data.columns])
    if feature_pair_x and feature_pair_y:
        fig = px.scatter(selected_data, x=feature_pair_x, y=feature_pair_y, color='Churn Prediction',
                         color_discrete_map=color_map, title=f'{feature_pair_y} vs {feature_pair_x}')
        fig.update_layout(xaxis_title=feature_pair_x, yaxis_title=feature_pair_y, title_font_size=24)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA Analysis")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(selected_data.fillna(0)[columns_to_normalize])
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Churn Prediction'] = selected_data['Churn Prediction']

    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Churn Prediction',
                     color_discrete_map=color_map, title='PCA of Test Data')
    fig.update_layout(xaxis_title='PC1', yaxis_title='PC2', title_font_size=24)
    st.plotly_chart(fig, use_container_width=True)

# Additional Visualizations
elif selection == "Additional Visualizations":
    st.header("Additional Visualizations")

    profile_df = df_test.copy()  # Use df_test directly for additional visualizations

    # Ensure 'Churn Prediction' is present in the DataFrame
    profile_df = preprocess_for_prediction(profile_df)
    profile_df['Churn Prediction'] = loaded_model.predict(profile_df)

    select, compare = st.tabs(["General Statistics", "Feature Relationships"])

    with select:
        st.subheader("General Statistics")
        
        # Statistics summary
        st.write("### Summary Statistics")
        st.write(profile_df.describe())
        
        # Distribution of numerical features
        st.write("### Distribution of Numerical Features")
        for feature in columns_to_normalize:
            if feature in profile_df.columns:
                fig = px.histogram(profile_df, x=feature, title=f'Distribution of {feature}', color_discrete_sequence=['#1f77b4'])
                fig.update_layout(xaxis_title=feature, yaxis_title='Frequency', title_font_size=24)
                st.plotly_chart(fig, use_container_width=True)

        # Pairwise correlation heatmap
        st.subheader("Pairwise Correlation Heatmap")
        numeric_df = profile_df.select_dtypes(include=['number'])
        
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=24)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, use_column_width=True)
        buf.close()
        plt.close(fig)

    with compare:
        st.subheader("Feature Relationships")
        
        # Scatter plots for feature relationships
        feature_x = st.selectbox("Select Feature for X Axis", options=[col for col in columns_to_normalize if col in profile_df.columns])
        feature_y = st.selectbox("Select Feature for Y Axis", options=[col for col in columns_to_normalize if col in profile_df.columns])

        if feature_x and feature_y:
            color_by = st.selectbox("Color by", options=['Churn Prediction'] + [col for col in categorical_columns if col in profile_df.columns])

            if color_by:
                fig = px.scatter(profile_df, x=feature_x, y=feature_y, color=color_by,
                                 color_discrete_map=color_map if color_by == 'Churn Prediction' else None,
                                 title=f'{feature_y} vs {feature_x}')
                fig.update_layout(xaxis_title=feature_x, yaxis_title=feature_y, title_font_size=24)
                st.plotly_chart(fig, use_container_width=True)

# Customer Simulation
elif selection == "Customer Simulation":
    st.header("Customer Simulation")

    st.write("Enter the details of a new customer to predict churn risk.")

    # Collecting user inputs for new customer
    customer_data = {}
    
    customer_data['Customer_Age'] = st.number_input("Customer Age", min_value=0, max_value=120, value=30)
    customer_data['Dependent_count'] = st.number_input("Dependent Count", min_value=0, max_value=10, value=0)
    customer_data['Months_on_book'] = st.number_input("Months on Book", min_value=0, max_value=120, value=12)
    customer_data['Total_Relationship_Count'] = st.number_input("Total Relationship Count", min_value=1, max_value=10, value=1)
    customer_data['Months_Inactive_12_mon'] = st.number_input("Months Inactive in the Last 12 Months", min_value=0, max_value=12, value=0)
    customer_data['Contacts_Count_12_mon'] = st.number_input("Contacts Count in the Last 12 Months", min_value=0, max_value=10, value=1)
    customer_data['Credit_Limit'] = st.number_input("Credit Limit", min_value=0, max_value=100000, value=5000)
    customer_data['Total_Revolving_Bal'] = st.number_input("Total Revolving Balance", min_value=0, max_value=100000, value=0)
    customer_data['Avg_Open_To_Buy'] = st.number_input("Average Open To Buy", min_value=0, max_value=100000, value=5000)
    customer_data['Total_Amt_Chng_Q4_Q1'] = st.number_input("Total Amount Change from Q4 to Q1", min_value=-1.0, max_value=1.0, value=0.0)
    customer_data['Total_Trans_Amt'] = st.number_input("Total Transaction Amount", min_value=0, max_value=100000, value=1000)
    customer_data['Total_Trans_Ct'] = st.number_input("Total Transaction Count", min_value=0, max_value=100, value=10)
    customer_data['Total_Ct_Chng_Q4_Q1'] = st.number_input("Total Count Change from Q4 to Q1", min_value=-1.0, max_value=1.0, value=0.0)
    customer_data['Avg_Utilization_Ratio'] = st.number_input("Average Utilization Ratio", min_value=0.0, max_value=1.0, value=0.2)

    # Collecting categorical inputs
    customer_data['Gender'] = st.selectbox("Gender", options=["Male", "Female"])
    customer_data['Education_Level'] = st.selectbox("Education Level", options=["Uneducated", "High School", "College", "Graduate", "Post-Graduate"])
    customer_data['Marital_Status'] = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
    customer_data['Income_Category'] = st.selectbox("Income Category", options=["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "More than $120K"])
    customer_data['Card_Category'] = st.selectbox("Card Category", options=["Blue", "Silver", "Gold", "Platinum"])

    if st.button("Predict Churn"):
        new_customer_df = pd.DataFrame([customer_data])
        new_customer_df = preprocess_for_prediction(new_customer_df)
        churn_prediction = loaded_model.predict(new_customer_df)
        message = 'Churn' if churn_prediction[0] == 1 else 'No Churn'
        color = 'red' if churn_prediction[0] == 1 else 'green'
        st.markdown(f"<h2 style='color: {color};'>{message}</h2>", unsafe_allow_html=True)