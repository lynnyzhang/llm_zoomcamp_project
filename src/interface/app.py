import argparse
import sys
import uuid
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env explicitly — do not rely on Streamlit's implicit CWD-based dotenv
# (only runs at startup and misses .env created later / different CWD).
load_dotenv(project_root / ".env")

import streamlit as st

from src.interface.cards import CardRenderer
from src.interface.chat_page import ChatPage
from src.interface.messages import MessageRenderer
from src.interface.turn_saver import TurnSaver


def parse_cli_flags(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--show-confidence", action="store_true", default=False)
    args, _ = parser.parse_known_args(argv)
    return args.show_confidence


# Read once at import; --show-confidence toggles the grounding-confidence bar.
SHOW_CONFIDENCE = parse_cli_flags(sys.argv[1:])

st.set_page_config(
    page_title="Pokémon Assistant",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

cards = CardRenderer()
saver = TurnSaver()
messages = MessageRenderer(cards, saver, SHOW_CONFIDENCE)
chat_page = ChatPage(cards, messages, saver)

chat_page.load_history()

st.title("🤖 Pokémon Assistant")
st.caption("Agentic RAG assistant for Pokémon knowledge")

# Chat history
for message in st.session_state.messages:
    chat_page.render(message)

# Chat input
if prompt := st.chat_input("Ask a question about Pokémon..."):
    chat_page.run_turn(prompt)
