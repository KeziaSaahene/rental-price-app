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
 
# A tiny house-silhouette tile, URL-encoded inline (no external asset needed).
# The fill color/opacity is injected per-theme where this template is used.
_HOUSE_TILE = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' "
    "height='160'%3E%3Cg fill='{fill}'%3E%3Cpath d='M80 24 L124 62 L112 62 L112 118 "
    "L48 118 L48 62 L36 62 Z'/%3E%3C/g%3E%3C/svg%3E"
)
 
 
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
        bg_top, bg_bottom = "#0B1120", "#0E1526"
        glow_a, glow_b = "rgba(59,102,192,0.16)", "rgba(176,141,62,0.08)"
        house_fill = "rgba(255,255,255,0.03)"
        surface, border = "#161F36", "#2A3757"
        text, text_muted = "#EDEAE2", "#8D97B4"
        input_bg = "#101A30"
        shadow = "0 25px 70px rgba(0,0,0,0.5)"
    else:
        bg_top, bg_bottom = "#FDFBF7", "#F3EEE3"
        glow_a, glow_b = "rgba(176,141,62,0.10)", "rgba(44,95,88,0.06)"
        house_fill = "rgba(20,18,10,0.035)"
        surface, border = "#FFFFFF", "#E4DDCB"
        text, text_muted = "#151B2E", "#726B5B"
        input_bg = "#FBF9F4"
        shadow = "0 20px 50px rgba(20,15,5,0.08)"
 
    gold, gold_light, teal = "#B08D3E", "#E4C77A", "#2C5F58"
    house_tile = _HOUSE_TILE.format(fill=house_fill)
 
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Manrope:wght@400;500;600;700&display=swap');
 
    html, body {{ font-family: 'Manrope', sans-serif; color: {text}; }}
 
    [data-testid="stAppViewContainer"] {{
        background:
            url("{house_tile}") repeat,
            radial-gradient(ellipse 900px 500px at 8% -8%, {glow_a} 0%, transparent 60%),
            radial-gradient(ellipse 700px 500px at 100% 10%, {glow_b} 0%, transparent 55%),
            linear-gradient(180deg, {bg_top} 0%, {bg_bottom} 55%) !important;
        color: {text};
    }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stToolbar"] {{ display: none; }}
 
    .block-container {{ max-width: 760px; padding-top: 3rem; padding-bottom: 4rem; }}
 
    .mikasa-brand {{
        display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.2rem;
        perspective: 400px;
    }}
    .mikasa-mark {{
        width: 20px; height: 20px; flex-shrink: 0;
        background: linear-gradient(135deg, {gold} 0%, {gold_light} 45%, {gold} 100%);
        border-radius: 3px;
        animation: mikasa-spin 6s linear infinite;
    }}
    @keyframes mikasa-spin {{
        from {{ transform: rotate(45deg) rotateY(0deg); }}
        to   {{ transform: rotate(45deg) rotateY(360deg); }}
    }}
    .mikasa-wordmark {{
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 3.2rem;
        letter-spacing: -0.01em; color: {text}; margin: 0; line-height: 1;
    }}
    .mikasa-tagline {{
        font-size: 0.92rem; color: {text_muted}; margin: 0.5rem 0 2.4rem 0; max-width: 44ch;
        line-height: 1.55; font-weight: 500;
    }}
    .mikasa-instruction {{
        font-size: 0.95rem; color: {text_muted}; margin-bottom: 1.1rem; font-weight: 600;
    }}
 
    div[data-testid="stVerticalBlockBorderWrapper"].st-key-mikasa_card,
    .st-key-mikasa_card {{
        background: {surface}; border: 1px solid {border}; border-radius: 14px;
        padding: 0.6rem 1.7rem 1.4rem 1.7rem; box-shadow: {shadow};
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
        border-radius: 8px !important;
    }}
    /* Force readable text everywhere inside selects/multiselect — BaseWeb sets
       its own low-contrast color on inner spans that a parent-level rule
       above doesn't override, which is why closed dropdowns looked dim. */
    [data-testid="stSelectbox"] *,
    [data-testid="stMultiSelect"] * {{
        color: {text} !important;
    }}
    [data-testid="stMultiSelect"] span[data-baseweb="tag"],
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] * {{
        background: {teal} !important; border-radius: 6px !important; color: #F3F1EA !important;
    }}
    ul[role="listbox"] {{ background: {input_bg} !important; }}
    ul[role="listbox"] li {{ color: {text} !important; }}
 
    [data-testid="stFormSubmitButton"] button {{
        background: {gold} !important; color: #1A1204 !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important; padding: 0.7rem 0 !important;
        font-size: 0.98rem !important; letter-spacing: 0.01em; transition: opacity 0.15s ease;
    }}
    [data-testid="stFormSubmitButton"] button:hover {{ opacity: 0.88; }}
 
    .mikasa-result {{
        margin-top: 1.6rem; padding-top: 1.6rem; border-top: 1px solid {border};
    }}
    .mikasa-result-label {{
        font-size: 0.8rem; font-weight: 600; color: {text_muted}; letter-spacing: 0.02em;
        margin-bottom: 0.3rem;
    }}
    .mikasa-result-value {{
        font-family: 'Fraunces', serif; font-weight: 500; font-size: 2.6rem; color: {gold};
        line-height: 1.1;
    }}
    .mikasa-result-note {{
        font-size: 0.86rem; color: {text_muted}; margin-top: 0.6rem; line-height: 1.5; max-width: 52ch;
    }}
 
    .mikasa-footer {{
        margin-top: 2.6rem; font-size: 0.8rem; color: {text_muted}; line-height: 1.6;
        border-top: 1px solid {border}; padding-top: 1.1rem; max-width: 52ch;
    }}
    </style>
    """
 
 
def main():
    st.set_page_config(page_title="Mi Casa — Rental Valuation", layout="centered")
 
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
            <div class="mikasa-brand">
                <div class="mikasa-mark"></div>
                <p class="mikasa-wordmark">Mi Casa</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
    st.markdown(
        '<p class="mikasa-tagline">Property valuation intelligence for the '
        "Ghanaian real estate market.</p>",
        unsafe_allow_html=True,
    )
 
    model = load_model()
    locality_map = load_locality_map()
 
    # A real Streamlit container (with a stable key) rather than a raw
    # markdown <div>, so the fields actually render *inside* the styled box
    # instead of leaving it as an empty bar with the form floating below it.
    with st.container(key="mikasa_card"):
        st.markdown(
            '<p class="mikasa-instruction">Enter your property\'s details for '
            "an instant valuation.</p>",
            unsafe_allow_html=True,
        )
 
        # Region lives outside the form so changing it immediately refreshes
        # the Locality list below — fields inside st.form only update on submit.
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
                <div class="mikasa-result">
                    <div class="mikasa-result-label">Estimated monthly rent</div>
                    <div class="mikasa-result-value">GH₵ {price:,.0f}</div>
                    <div class="mikasa-result-note">
                        Based on comparable listings for this property type and location.
                        Treat this as a guide alongside current market listings, not an exact valuation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
    st.markdown(
        '<div class="mikasa-footer">Mi Casa estimates are generated from historical '
        "rental listing data across Ghana and are indicative only.</div>",
        unsafe_allow_html=True,
    )
 
 
if __name__ == "__main__":
    main()
 
