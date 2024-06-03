from pydantic import BaseModel, Field, computed_field
from PIL import Image

import streamlit as st
import instructor
import anthropic
import openai

from pathlib import Path
import base64
import json


def get_image_data():
    global DATAPATH
    with open(DATAPATH / "notes/data.json", "r") as f:
        data = json.load(f)
    return {note["name"]: Note(**note, image=Image.open(note["path"])) for note in data}


def get_recording_data():
    global DATAPATH
    with open(DATAPATH / "recordings/data.json", "r") as f:
        data = [Recording(**d) for d in json.load(f)]
    return {d.name: d for d in data}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        b64str = base64.b64encode(image_file.read()).decode("utf-8")
        return f"{b64str}"


class HandwrittenText(BaseModel):
    """Text extracted from a handwritten doctor's note.
    The document may be a printed form with handwritten information or a page
    of handwritten notes.
    """

    raw_text: str = Field(
        ..., description="The raw text extracted from the handwritten document."
    )
    md_text: str = Field(
        ...,
        description="""\
            A markdown formatted version of the handwritten text.
            The text is formatted to be suitable for a markdown viewer.
            The text should be organized in a way that is easy to read.
        """,
    )


class Note(BaseModel):
    name: str
    path: Path
    extracted_gpt: HandwrittenText | None = None
    extracted_claude: HandwrittenText | None = None

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    def image(self) -> Image.Image:
        return Image.open(self.path)

    @computed_field
    def image_base64(self) -> str:
        return encode_image(self.path)


class Recording(BaseModel):
    path: Path
    raw_transcript: str
    summary: str

    @computed_field
    def name(self) -> str:
        return self.path.stem


DATAPATH = Path("/mnt/arrakis/meddibia/meddibia-demos/data/")

GPT = instructor.from_openai(openai.OpenAI(), mode=instructor.Mode.MD_JSON)
CLAUDE = instructor.from_anthropic(
    anthropic.Client(), mode=instructor.Mode.ANTHROPIC_JSON
)

IMAGE_DATA = get_image_data()
RECORDING_DATA = get_recording_data()

st.set_page_config(layout="wide")
st.logo("/mnt/arrakis/meddibia/meddibia-demos/src/ai-mvp/meddibia-logo.png")

handwritten_text_tab, recording_tab = st.tabs(
    ["Handwritten Text Extraction", "Audio Transcription"]
)

with handwritten_text_tab:
    selected_note = st.selectbox(
        "Select a note", options=[note.name for note in IMAGE_DATA.values()]
    )

    image_col, text_col = st.columns(2)

    with image_col:
        st.image(IMAGE_DATA[selected_note].image, width=600)

    with text_col:
        # gpt_tab, claude_tab = st.tabs(["GPT", "Claude"])

        # with gpt_tab:
        with st.expander("Formatted Text", expanded=True):
            st.markdown(IMAGE_DATA[selected_note].extracted_gpt.md_text)

        with st.expander("Raw Text", expanded=False):
            st.write(IMAGE_DATA[selected_note].extracted_gpt.raw_text)

    # with claude_tab:
    #     with st.expander("Formatted Text", expanded=True):
    #         st.markdown(IMAGE_DATA[selected_note].extracted_claude.md_text)

    #     with st.expander("Raw Text", expanded=False):
    #         st.write(IMAGE_DATA[selected_note].extracted_claude.raw_text)

with recording_tab:
    selected_recording = st.selectbox(
        "Select a recording",
        options=[recording.name for recording in RECORDING_DATA.values()],
    )

    st.audio(
        str(RECORDING_DATA[selected_recording].path), format="audio/mpeg", loop=True
    )

    with st.expander("Summary", expanded=True):
        st.write(RECORDING_DATA[selected_recording].summary)

    with st.expander("Raw Transcript", expanded=False):
        st.write(RECORDING_DATA[selected_recording].raw_transcript)
