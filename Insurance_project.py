import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
model_path = 'insurance_model.pkl'
model = pickle.load(open(model_path, 'rb'))
def encode_region(region):
    regions = {
        'northeast': 0,
        'northwest': 1,
        'southeast': 2,
        'southwest': 3,
    }
    return regions.get(region, -1)
st.title('Insurance Prediction Model')
st.write("Welcome to the Insurance Prediction Model! Fill out the form below to predict insurance charges based on your inputs.")
st.sidebar.header("Input Data")
age = st.sidebar.slider('Age', min_value=0, max_value=120, value=25, step=1)
sex = st.sidebar.selectbox('Sex', ('male', 'female'))
bmi = st.sidebar.slider('BMI', min_value=10.0, max_value=50.0, value=22.0, step=0.1)
children = st.sidebar.slider('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox('Are you a smoker?', ('No', 'Yes'))
region = st.sidebar.selectbox('Region', ('northeast', 'northwest', 'southeast', 'southwest'))

if st.sidebar.button('Predict'):
    sex_encoded = 1 if sex == 'male' else 0
    smoker_encoded = 1 if smoker == 'Yes' else 0
    region_encoded = encode_region(region)
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    prediction = model.predict(input_data)
    st.success(f'The predicted insurance charge is: ${prediction[0]:.2f}')
    
    st.subheader('Prediction Analysis')
    fig, ax = plt.subplots()
    ax.bar(['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region'], input_data.flatten(), color='skyblue')
    ax.axhline(y=prediction[0], color='b', linestyle='--', label='Predicted Charge')
    ax.set_ylabel('Input Value / Prediction Charge')
    ax.set_title('Feature Contribution to Prediction')
    ax.legend()
    st.pyplot(fig)

    st.write("### Explanation of Inputs")
    st.write(f"- **Age**: {age} years old")
    st.write(f"- **Sex**: {sex}")
    st.write(f"- **BMI**: {bmi} kg/m²")
    st.write(f"- **Children**: {children}")
    st.write(f"- **Smoker**: {'Yes' if smoker_encoded == 1 else 'No'}")
    st.write(f"- **Region**: {region}")
    
st.write("### How the Model Works")
st.info(""" 
The model predicts insurance charges based on various personal factors:
- Age: Older individuals typically pay more.
- Sex: Males and females may have different average charges.
- BMI: A higher BMI may increase charges.
- Children: More children might lead to higher family insurance costs.
- Smoking: Smokers usually have higher insurance premiums.
- Region: Insurance rates can vary significantly by geographic location.
""")


