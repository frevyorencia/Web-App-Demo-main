import streamlit as st

st.set_page_config(page_title="Mini Layout Playground", page_icon="🎨", layout="wide")

st.markdown(
    """
    <style>
    .placeholder-box {
        border: 2px dashed #cbd5f5;
        border-radius: 10px;
        padding: 1.6rem 1rem;
        text-align: center;
        color: #64748b;
        background-color: #f8fafc;
        font-weight: 600;
    }
    div[data-testid="stButton"] > button {
        border-radius: 999px;
        padding: 0.45rem 1.2rem;
        background: #6366f1;
        color: #ffffff;
        border: none;
        font-weight: 600;
    }
    div[data-testid="stButton"] > button:hover {
        background: #4f46e5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Mini Website Playground")
st.caption("Adjust a few controls in the sidebar and watch the page change right away.")

with st.sidebar:
    st.header("Quick tweaks")
    page_heading = st.text_input("Page heading", "Sunny Landing Page")
    tagline = st.text_input("Tagline", "Keep your copy short and cheerful.")
    hero_headline = st.text_input("Hero headline", "Make your layout your own")
    hero_body = st.text_area("Hero text", "Use this mini playground to move text and buttons around.")
    primary_label = st.text_input("Main button label", "Get Started")
    secondary_label = st.text_input("Secondary link", "Learn more")
    hero_layout = st.radio("Hero layout", ("Text left", "Centered"))

    st.markdown("---")
    show_highlights = st.checkbox("Show highlight cards", True)
    highlights_raw = st.text_area(
        "Highlights (one per line)",
        "Fast setup\nSimple buttons\nClean text",
    ) if show_highlights else ""

st.subheader(page_heading)
if tagline.strip():
    st.write(tagline)

if hero_layout == "Text left":
    left, right = st.columns([2, 1])
    with left:
        st.write(f"### {hero_headline}")
        st.write(hero_body)
        st.button(primary_label, key="hero_left_button")
        if secondary_label.strip():
            st.write(f"[{secondary_label}](#)")
    with right:
        st.markdown("<div class='placeholder-box'>Add your image here</div>", unsafe_allow_html=True)
else:
    st.write(f"### {hero_headline}")
    st.write(hero_body)
    st.button(primary_label, key="hero_center_button")
    if secondary_label.strip():
        st.write(f"[{secondary_label}](#)")

if show_highlights:
    items = [item.strip() for item in highlights_raw.splitlines() if item.strip()]
    if not items:
        items = ["Add highlight lines in the sidebar"]
    max_items = min(len(items), 3)
    cols = st.columns(max_items)
    for idx in range(max_items):
        with cols[idx]:
            st.write(f"**{items[idx]}**")
            st.write("Keep the description to two short sentences.")
            st.button("Action", key=f"highlight_button_{idx}")

st.write("---")
st.info(
    "Tip: Try deleting text or renaming the button to feel how small tweaks change the page vibe."
)
