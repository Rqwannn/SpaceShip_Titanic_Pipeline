import streamlit as st
import requests
import json

st.set_page_config(page_title="SpaceShip Titanic Prediction", page_icon="Dicoding", layout="centered")

API_URL = "http://localhost:8001/invocations"

st.title("SpaceShip Titanic - Passenger Transport Prediction")
st.markdown("Aplikasi memprediksi apakah seorang penumpang akan diangkut. Masukkan data di sidebar atau muat salah satu contoh di bawah ini.")

if 'home_planet' not in st.session_state:
    st.session_state.home_planet = "Europa"
    st.session_state.age = 30
    st.session_state.deck = "B"
    st.session_state.cabin_num = 5
    st.session_state.side = "P"
    st.session_state.room_service = 0.0
    st.session_state.food_court = 0.0
    st.session_state.shopping_mall = 0.0
    st.session_state.spa = 0.0
    st.session_state.vr_deck = 0.0

transported_example = {
    "home_planet": "Europa", "age": 34, "deck": "B", "cabin_num": 5, "side": "P",
    "room_service": 0.0, "food_court": 0.0, "shopping_mall": 0.0, "spa": 0.0, "vr_deck": 0.0
}
not_transported_example = {
    "home_planet": "Earth", "age": 25, "deck": "F", "cabin_num": 150, "side": "S",
    "room_service": 0.0, "food_court": 700.0, "shopping_mall": 100.0, "spa": 300.0, "vr_deck": 10.0
}

def load_example(example_data):
    for key, value in example_data.items():
        st.session_state[key] = value

col1, col2 = st.columns(2)
with col1:
    if st.button("Muat Contoh: Transported", use_container_width=True):
        load_example(transported_example)
with col2:
    if st.button("Muat Contoh: Not Transported", use_container_width=True):
        load_example(not_transported_example)

st.sidebar.header("Passenger Input Features")

with st.sidebar.form(key='prediction_form'):
    st.subheader("Passenger Info")
    home_planet = st.selectbox("Home Planet", ("Earth", "Europa", "Mars"), key="home_planet")
    age = st.number_input("Age", min_value=0, max_value=100, key="age")

    st.subheader("Cabin Details")
    deck = st.selectbox("Deck", ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'), key="deck")
    cabin_num = st.number_input("Cabin Number", min_value=0, key="cabin_num")
    side = st.selectbox("Side", ("P", "S"), key="side")

    st.subheader("Passenger Spending ($)")
    room_service = st.number_input("Room Service", min_value=0.0, step=100.0, key="room_service")
    food_court = st.number_input("Food Court", min_value=0.0, step=100.0, key="food_court")
    shopping_mall = st.number_input("Shopping Mall", min_value=0.0, step=100.0, key="shopping_mall")
    spa = st.number_input("Spa", min_value=0.0, step=100.0, key="spa")
    vr_deck = st.number_input("VR Deck", min_value=0.0, step=100.0, key="vr_deck")
    
    submit_button = st.form_submit_button(label='Predict Transport Status')

if submit_button:
    payload = {
        "home_planet": st.session_state.home_planet,
        "deck": st.session_state.deck,
        "side": st.session_state.side,
        "age": float(st.session_state.age),
        "cabin_num": float(st.session_state.cabin_num),
        "room_service": st.session_state.room_service,
        "food_court": st.session_state.food_court,
        "shopping_mall": st.session_state.shopping_mall,
        "spa": st.session_state.spa,
        "vr_deck": st.session_state.vr_deck
    }
    
    with st.spinner('Processing data and sending to model...'):
        try:
            response = requests.post(API_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('predictions', [0])[0]
                
                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.success("Transported - Penumpang ini kemungkinan besar akan diangkut!")
                else:
                    st.error("Not Transported - Penumpang ini kemungkinan besar akan tetap di sini.")
                
                with st.expander("Show Raw API Response"):
                    st.json(result)
            else:
                st.error(f"Error from API: Status {response.status_code}")
                st.json(response.json())

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the prediction service at {API_URL}.")
            st.error(f"Error details: {e}")