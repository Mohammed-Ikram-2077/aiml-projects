from __future__ import annotations

import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"


def load_site_html() -> str:
    """Inline Vite's built assets so the React site can run in a Streamlit iframe."""
    html_path = DIST / "index.html"
    if not html_path.exists():
        raise FileNotFoundError("The built site is missing. Run `npm run build` first.")

    html = html_path.read_text(encoding="utf-8")

    stylesheet_pattern = re.compile(
        r'<link\b[^>]*href=["\'](?:\./)?assets/([^"\']+\.css)["\'][^>]*>'
    )
    script_pattern = re.compile(
        r'<script\b[^>]*src=["\'](?:\./)?assets/([^"\']+\.js)["\'][^>]*>\s*</script>'
    )

    def inline_css(match: re.Match[str]) -> str:
        css_file = DIST / "assets" / match.group(1)
        return f"<style>\n{css_file.read_text(encoding='utf-8')}\n</style>"

    def inline_js(match: re.Match[str]) -> str:
        js_file = DIST / "assets" / match.group(1)
        return f'<script type="module">\n{js_file.read_text(encoding="utf-8")}\n</script>'

    html = stylesheet_pattern.sub(inline_css, html)
    html = script_pattern.sub(inline_js, html)
    return html


st.set_page_config(page_title="Aussies Cafe", page_icon="☕", layout="wide")
st.markdown(
    """
    <style>
      #MainMenu, footer, header { visibility: hidden; }
      .block-container { padding: 0 !important; max-width: 100% !important; }
      div[data-testid="stAppViewContainer"] { background: #fff; }
      iframe { border: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    site_html = load_site_html()
except FileNotFoundError as error:
    st.error(str(error))
else:
    components.html(site_html, height=850, scrolling=True)
