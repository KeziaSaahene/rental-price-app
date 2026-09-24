import json
import time
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

# The Mi Casa house-outline mark, embedded as base64 so no extra asset file
# is needed in the repo. Black line-art on a transparent background; dark
# mode inverts it to white via CSS filter (see inject_theme's mark_filter).
_LOGO_B64 = (
    "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"
    "432KHbuIZ5vltSf1NIRn5aYVpZ+BuK7yXaQz4FnGur41MDGOO4v7MT3sfj20sU1MMlTEMOGkFtrpzcQkeyvbxu7usdulK35lPdNx"
    "G9vwsLfmxcTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExKRbxLKSih3bnqeNbXpoY5uYmJiY"
    "9LJFHXTp2HnvfIGNnfv8mx6aHpqOm5iYmJiYmJiYmJi0QYy0t7FND21sG9uU3wDHxraxTQ9tbLtxWyQ2to1tOt6nFrmJiYmJiYmJ"
    "iYmJiYmJiYlJhmK8T7Fj2/O0sU0PbWwTExMTExMTExMTExMTExMTExMTExOTdshgaHNgYmJi0pEyYgBtYmJi0pkSGkCbmJiYdCpC"
    "2xSYmJiYGECbmJiYmBhAm5iYmBhAm5iYmJgUANCW117s2HmLzbmNbXrY5WOXbFJtkdjYNrbpeEeOPVKq8x+t3X3y2NaSPv/5Nz00"
    "Pew3HQ/qAWgTExMTk/aJZRKamJiYdKpkDdDGvdnYnSA25zZ2T+h4KYebCnKcsE4du92AE9jYNnYBgBSYjhc6dssctO2cZh3a2Kbj"
    "Ni/5jG1OQhMTE5MOFXMSmpiYmHSqGECbmJiYdAFAG+9T7Nh5i825jW162OVjGwdtYmJi0pky8v8DWJqIX/X+Fl0AAAAASUVORK5C"
    "YII="
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
    # The embedded logo is black line-art on transparent; invert it to white
    # line-art in dark mode so it stays legible against the dark background.
    mark_filter = "filter: invert(1);" if dark else ""

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

    .block-container {{ max-width: 760px; padding-top: 0.45rem; padding-bottom: 0.45rem; }}

    .mikasa-brand {{
        display: flex; align-items: center; gap: 1.1rem; margin-bottom: 0.05rem;
        flex-wrap: wrap; position: relative; z-index: 0;
    }}
    .mikasa-mark {{
        width: clamp(60px, 10vw, 125px); max-width: 100%; height: auto; flex-shrink: 0;
        {mark_filter}
    }}
    [data-testid="stToggle"] {{
        position: relative; z-index: 10;
    }}
    .mikasa-wordmark {{
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    font-size: 3rem !important;
    letter-spacing: -0.04em !important;
    line-height: 0.85 !important;

    margin: 0 !important;
    padding: 0 !important;

    white-space: nowrap !important;
    width: max-content !important;
    flex-shrink: 0 !important;

    background: linear-gradient(
        110deg,
        #8A6A24 0%,
        #C9A23A 18%,
        #FFE9A3 38%,
        #B88920 52%,
        #F5D76E 68%,
        #A97816 82%,
        #E8C65A 100%
    ) !important;

    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;

    text-shadow:
        0 2px 8px rgba(176, 141, 62, 0.25),
        0 0 25px rgba(228, 199, 122, 0.18) !important;
}}
    .mikasa-tagline {{
        font-size: 0.84rem; color: {text_muted}; margin: 0.2rem 0 0.55rem 0;
        line-height: 1.55; font-weight: 500; max-width: none; width: 100%;
        white-space: nowrap;
    }}
    .mikasa-instruction {{
        font-size: 0.86rem; color: {text_muted}; margin-bottom: 0.45rem; font-weight: 600;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"].st-key-mikasa_card,
    .st-key-mikasa_card {{
        background: {surface}; border: 1px solid {border}; border-radius: 14px;
        padding: 0.3rem 1.3rem 0.5rem 1.3rem; box-shadow: {shadow};
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

    /* Match the number-input text to the white text used in the other inputs. */
    [data-testid="stNumberInput"] label {{
        color: {text} !important;
    }}
    [data-testid="stNumberInput"] input {{
        color: {text} !important;
        -webkit-text-fill-color: {text} !important;
        caret-color: {text} !important;
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
        margin-top: 0.45rem; padding-top: 0.45rem; border-top: 1px solid {border};
    }}
    .mikasa-result-label {{
        font-size: 0.8rem; font-weight: 600; color: {text_muted}; letter-spacing: 0.02em;
        margin-bottom: 0.3rem;
    }}
    .mikasa-result-value {{
        font-family: 'Fraunces', serif; font-weight: 500; font-size: 2.05rem; color: {gold};
        line-height: 1.1;
    }}
    .mikasa-result-note {{
        font-size: 0.86rem; color: {text_muted}; margin-top: 0.35rem; line-height: 1.3;
        width: 100%; max-width: none;
    }}

    .mikasa-estimating {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        padding: 0.35rem 0 0.3rem;
    }}
    .mikasa-spinner {{
        width: 34px; height: 34px;
        border: 3px solid rgba(255,255,255,0.22);
        border-top-color: {gold};
        border-radius: 50%;
        animation: mikasa-spin 0.8s linear infinite;
    }}
    .mikasa-estimating-text {{
        margin-top: 0.45rem; font-size: 0.75rem; font-weight: 700;
        color: {text}; letter-spacing: 0.16em;
    }}
    @keyframes mikasa-spin {{
        to {{ transform: rotate(360deg); }}
    }}

    .mikasa-footer {{
        margin-top: 1rem; font-size: 0.76rem; color: {text_muted}; line-height: 1.6;
        border-top: 1px solid {border}; padding-top: 1.1rem; max-width: 52ch;
    }}
    </style>
    """


def main():
    st.set_page_config(page_title="Mi Casa — Rental Valuation", layout="centered")

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    top_l, top_r = st.columns([5, 1.3], vertical_alignment="center")
    with top_r:
        st.session_state.dark_mode = st.toggle(
            "Dark" if st.session_state.dark_mode else "Light", value=st.session_state.dark_mode, key="theme_toggle"
        )

    st.markdown(inject_theme(st.session_state.dark_mode), unsafe_allow_html=True)

    with top_l:
        st.markdown(
            f"""
            <div class="mikasa-brand">
                <img class="mikasa-mark" src="data:image/png;base64,{_LOGO_B64}" alt="" />
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
            rental_period = st.selectbox(
                "Rental period",
                ["6 months", "1 year", "1.5 years", "2 years"],
            )

            submitted = st.form_submit_button("Estimate rent", use_container_width=True)

        if submitted:
            estimating = st.empty()
            estimating.markdown(
                """
                <div class="mikasa-estimating">
                    <div class="mikasa-spinner"></div>
                    <div class="mikasa-estimating-text">Estimating</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            time.sleep(1)

            price = predict_price(
                model, bedrooms, bathrooms, floor_area,
                region, locality, category, is_furnished, amenities,
            )

            period_months = {
                "6 months": 6,
                "1 year": 12,
                "1.5 years": 18,
                "2 years": 24,
            }
            total_rent = price * period_months[rental_period]

            estimating.empty()
            st.markdown(
                f"""
                <div class="mikasa-result">
                    <div class="mikasa-result-label">Estimated rent — {rental_period}</div>
                    <div class="mikasa-result-value">GH₵ {total_rent:,.0f}</div>
                    <div class="mikasa-result-note">
                        <p class="mikasa-monthly">Monthly estimate: GH₵ {price:,.0f}</p>
                        <p class="mikasa-guidance">Based on comparable listings for this property type and location. Treat this as a guide alongside current market listings, not an exact valuation.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="mikasa-footer" style="margin-top:0.25rem !important;">Mi Casa estimates are generated from historical '
        "rental listing data across Ghana and are indicative only.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
