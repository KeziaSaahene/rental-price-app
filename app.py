import json
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import streamlit as st
 
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "rental_model.pkl"
LOCALITY_MAP_PATH = APP_DIR / "locality_map.json"
 
CATEGORIES = ["Detached", "Duplex", "Flats", "Mansion", "Semi-Detached", "Townhouse"]
FURNISHING = ["Unfurnished", "Semi-Furnished", "Furnished"]
AMENITIES = [
    "24-hour Electricity", "Air Conditioning", "Balcony", "Chandelier",
    "Dining Area", "Dishwasher", "Hot Water", "Kitchen Cabinets",
    "Kitchen Shelf", "Microwave", "Pop Ceiling", "Pre-Paid Meter",
    "Refrigerator", "TV", "Tiled Floor", "Wardrobe", "Wi-Fi",
]
 
 
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
 
 
@st.cache_data
def load_locality_map():
    with open(LOCALITY_MAP_PATH) as f:
        return json.load(f)
 
 
def predict_price(model, bedrooms, bathrooms, floor_area, region, locality, category, is_furnished, amenities):
    total_rooms = bedrooms + bathrooms
    amenities_count = len(amenities)
 
    input_df = pd.DataFrame([{
        "total_rooms": total_rooms,
        "amenities_count": amenities_count,
        "floor_area": floor_area,
        "bathrooms": bathrooms,
        "locality": locality,
        "is_furnished": is_furnished,
        "region": region,
        "category": category,
    }])
 
    log_price = model.predict(input_df)[0]
    return float(np.expm1(log_price))
 
 
def main():
    st.set_page_config(page_title="Ghana Rental Price Predictor", page_icon="🏠", layout="centered")
 
    st.title("🏠 Ghana Rental Price Predictor")
    st.write(
        "Estimate monthly rental price for a property in Ghana based on its "
        "characteristics. Model: XGBoost trained on Tonaton rental listings."
    )
 
    model = load_model()
    locality_map = load_locality_map()
 
    # Region lives OUTSIDE the form so changing it immediately refreshes the
    # Locality options below. Fields inside st.form only update on submit,
    # so a dependent dropdown like Locality must react to Region before that.
    region = st.selectbox("Region", sorted(locality_map.keys()))
    locality_options = sorted(locality_map.get(region, []))
 
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
 
        with col1:
            locality = st.selectbox("Locality", locality_options)
            category = st.selectbox("Property Category", CATEGORIES)
            is_furnished = st.selectbox("Furnishing Status", FURNISHING)
 
        with col2:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=2, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)
            floor_area = st.number_input("Floor Area (sq. m)", min_value=10.0, max_value=5000.0, value=100.0, step=5.0)
 
        amenities = st.multiselect("Amenities", AMENITIES, default=["Tiled Floor", "24-hour Electricity"])
 
        submitted = st.form_submit_button("Predict Rental Price", use_container_width=True)
 
    if submitted:
        with st.spinner("Predicting..."):
            price = predict_price(
                model, bedrooms, bathrooms, floor_area,
                region, locality, category, is_furnished, amenities,
            )
        st.success(f"### Estimated Monthly Rent: GH₵ {price:,.0f}")
        st.caption(
            "This is a model estimate based on historical listings and should be "
            "used as a rough guide, not an exact valuation."
        )
 
    st.divider()
    st.caption("Built with a tuned XGBoost regression pipeline (log-price target).")
 
 
if __name__ == "__main__":
    main()
 
