import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
df = pd.read_excel("Data_Train.xlsx")

def main():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(to bottom, #87CEEB, #F5F5F5);
            background-attachment: fixed;
            background-size: cover;
        }}
        .stMarkdown h1 {{
            font-weight: bold;
            font-size: 36px;  /* Adjust font size as needed */
            color: #3F7E41;  /* Optional: Set text color */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.container():
        st.markdown(
            """
            <h1 style='text-align: center;'>Flight Price Prediction</h1>
            """,
            unsafe_allow_html=True
        )
if __name__ == "__main__":
    main()
    
# Separate features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Encode categorical variables
le_Airline = LabelEncoder()
le_Source = LabelEncoder()
le_Destination = LabelEncoder()
le_Route = LabelEncoder()
le_Additional_Info = LabelEncoder()

X['Airline'] = le_Airline.fit_transform(X['Airline'])
X['Source'] = le_Source.fit_transform(X['Source'])
X['Destination'] = le_Destination.fit_transform(X['Destination'])
X['Route'] = le_Route.fit_transform(X['Route'])
X['Additional_Info'] = le_Additional_Info.fit_transform(X['Additional_Info'])

# Define input fields
airline = st.selectbox("Select Airline", le_Airline.classes_)
date_of_journey = st.date_input("Date of Journey")
source = st.selectbox("Select Source", le_Source.classes_)
destination = st.selectbox("Select Destination", le_Destination.classes_)
route = st.selectbox("Select Route", le_Route.classes_)
dep_time = st.time_input("Departure Time")
arrival_time = st.time_input("Arrival Time")
duration = st.number_input("Duration (hours)", min_value=0, max_value=24)
total_stops = st.number_input("Total Stops", min_value=0, max_value=10)
additional_info = st.selectbox("Additional Info", le_Additional_Info.classes_)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Airline': [le_Airline.transform([airline])[0]],
    'Date_of_Journey': [pd.to_datetime(date_of_journey).timestamp()],
    'Source': [le_Source.transform([source])[0]],
    'Destination': [le_Destination.transform([destination])[0]],
    'Route': [le_Route.transform([route])[0]],
    'Dep_Time': [dep_time.hour * 60 + dep_time.minute],
    'Arrival_Time': [arrival_time.hour * 60 + arrival_time.minute],
    'Duration': [duration],
    'Total_Stops': [total_stops],
    'Additional_Info': [le_Additional_Info.transform([additional_info])[0]]
})

# Load pre-trained model
filename = 'RandomForest_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Predict fare
if st.button("Predict"):
    prediction = loaded_model.predict(input_data)
    st.success(f"Predicted Fare: {prediction[0]:.2f}")

