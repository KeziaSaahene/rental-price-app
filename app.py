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


def inject_theme(dark: bool):
    """Build the full stylesheet for the chosen theme and inject it.

    Streamlit reruns the whole script on every interaction, so rather than
    toggling classes client-side, we regenerate the stylesheet with the
    right palette baked in and re-inject it each run.
    """
    if dark:
        bg, surface, border = "#0E1526", "#161F36", "#2A3757"
        text, text_muted = "#EDEAE2", "#9BA6C4"
        input_bg = "#101A30"
        shadow = "0 20px 60px rgba(0,0,0,0.45)"
    else:
        bg, surface, border = "#FAF7F1", "#FFFFFF", "#E4DDCB"
        text, text_muted = "#151B2E", "#6B6456"
        input_bg = "#FBF9F4"
        shadow = "0 20px 50px rgba(20,15,5,0.08)"

    gold, teal = "#B08D3E", "#2C5F58"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Manrope:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: {bg} !important;
        font-family: 'Manrope', sans-serif;
        color: {text};
    }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stToolbar"] {{ display: none; }}

    .block-container {{ max-width: 760px; padding-top: 3rem; padding-bottom: 4rem; }}

    .domora-brand {{
        display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.1rem;
    }}
    .domora-mark {{
        width: 11px; height: 11px; background: {gold};
        transform: rotate(45deg); flex-shrink: 0; border-radius: 2px;
    }}
    .domora-wordmark {{
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.15rem;
        letter-spacing: -0.01em; color: {text}; margin: 0;
    }}
    .domora-rule {{
        border: none; height: 1px; background: linear-gradient(90deg, {gold} 0%, {border} 40%);
        margin: 0.9rem 0 1.1rem 0;
    }}
    .domora-tagline {{
        font-size: 0.98rem; color: {text_muted}; margin-bottom: 2.1rem; max-width: 46ch;
        line-height: 1.55;
    }}

    .domora-card {{
        background: {surface}; border: 1px solid {border}; border-radius: 14px;
        padding: 2.1rem 2.3rem 1.7rem 2.3rem; box-shadow: {shadow};
    }}

    label, .stSelectbox label, .stNumberInput label, .stMultiSelect label {{
        font-family: 'Manrope', sans-serif !important; font-weight: 600 !important;
        font-size: 0.82rem !important; color: {text_muted} !important;
        letter-spacing: 0.01em;
    }}

    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stMultiSelect"] > div > div {{
        background: {input_bg} !important; border: 1px solid {border} !important;
        border-radius: 8px !important; color: {text} !important;
    }}
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
        background: {teal} !important; border-radius: 6px !important;
    }}

    [data-testid="stFormSubmitButton"] button {{
        background: {gold} !important; color: #1A1204 !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important; padding: 0.7rem 0 !important;
        font-size: 0.98rem !important; letter-spacing: 0.01em; transition: opacity 0.15s ease;
    }}
    [data-testid="stFormSubmitButton"] button:hover {{ opacity: 0.88; }}

    .domora-result {{
        margin-top: 1.6rem; padding-top: 1.6rem; border-top: 1px solid {border};
    }}
    .domora-result-label {{
        font-size: 0.8rem; font-weight: 600; color: {text_muted}; letter-spacing: 0.02em;
        margin-bottom: 0.3rem;
    }}
    .domora-result-value {{
        font-family: 'Fraunces', serif; font-weight: 500; font-size: 2.6rem; color: {gold};
        line-height: 1.1;
    }}
    .domora-result-note {{
        font-size: 0.86rem; color: {text_muted}; margin-top: 0.6rem; line-height: 1.5; max-width: 52ch;
    }}

    .domora-footer {{
        margin-top: 2.6rem; font-size: 0.8rem; color: {text_muted}; line-height: 1.6;
        border-top: 1px solid {border}; padding-top: 1.1rem; max-width: 52ch;
    }}
    </style>
    """


def main():
    st.set_page_config(page_title="Domora — Rental Valuation", layout="centered")

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    top_l, top_r = st.columns([5, 1.3])
    with top_r:
        st.session_state.dark_mode = st.toggle(
            "Dark", value=st.session_state.dark_mode, key="theme_toggle"
        )

    st.markdown(inject_theme(st.session_state.dark_mode), unsafe_allow_html=True)

    with top_l:
        st.markdown(
            """
            <div class="domora-brand">
                <div class="domora-mark"></div>
                <p class="domora-wordmark">Domora</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="domora-rule" />', unsafe_allow_html=True)
    st.markdown(
        '<p class="domora-tagline">Property valuation intelligence for the '
        "Ghanaian rental market. Enter a property's details for an instant, "
        "data-driven rent estimate.</p>",
        unsafe_allow_html=True,
    )

    model = load_model()
    locality_map = load_locality_map()

    st.markdown('<div class="domora-card">', unsafe_allow_html=True)

    # Region lives outside the form so changing it immediately refreshes the
    # Locality list below — fields inside st.form only update on submit.
    region = st.selectbox("Region", sorted(locality_map.keys()))
    locality_options = sorted(locality_map.get(region, []))

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            locality = st.selectbox("Locality", locality_options)
            category = st.selectbox("Property type", CATEGORIES)
            is_furnished = st.selectbox("Furnishing", FURNISHING)

        with col2:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=2, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)
            floor_area = st.number_input("Floor area (sq. m)", min_value=10.0, max_value=5000.0, value=100.0, step=5.0)

        amenities = st.multiselect("Amenities", AMENITIES, default=["Tiled Floor", "24-hour Electricity"])

        submitted = st.form_submit_button("Estimate rent", use_container_width=True)

    if submitted:
        price = predict_price(
            model, bedrooms, bathrooms, floor_area,
            region, locality, category, is_furnished, amenities,
        )
        st.markdown(
            f"""
            <div class="domora-result">
                <div class="domora-result-label">Estimated monthly rent</div>
                <div class="domora-result-value">GH₵ {price:,.0f}</div>
                <div class="domora-result-note">
                    Based on comparable listings for this property type and location.
                    Treat this as a guide alongside current market listings, not an exact valuation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="domora-footer">Domora estimates are generated from historical '
        "rental listing data across Ghana and are indicative only.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
