from pydantic import BaseModel, RootModel
from groq import Groq, AsyncGroq
from telegram import File
from loguru import logger
from tqdm import tqdm

from settings import Settings

from typing import Generator
from pathlib import Path
import datetime as dt
import sys
import os

settings = Settings()

MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 500


def get_logger() -> logger:
    base_path = Path(__file__).parent.parent.parent
    logger.remove()
    logger_config = {
        "format": "{time:%Y%M%d-%H%M%S} - {level} - {message}",
        "colorize": True,
        "level": "INFO",
    }
    logger.add(
        sink=base_path / "logs" / f"{dt.datetime.now().strftime('%Y%m%d-%H%M')}.log",
        **logger_config,
    )
    logger.add(
        sink=sys.stderr,
        **logger_config,
    )
    return logger


def get_job_name(chat_id: int, file_id: str) -> str:
    return f"C{chat_id}-F{file_id}"


def create_tmpfile(file: File) -> str:
    global settings
    uid = file.file_unique_id
    ext = file.file_path.split(".")[-1]
    return f"{settings.TMP_DIR}/{uid}.{ext}"


class FileGenerator:
    def __init__(self, audio_path: Path):
        self.generator = upload_file(audio_path)

    def read(self, size=-1):
        try:
            return next(self.generator)
        except StopIteration:
            return b""


def upload_file(audio_path: Path) -> Generator[bytes, None, None]:
    with open(audio_path, "rb") as f:
        file_size = os.path.getsize(audio_path)
        chunk_size = 1024

        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Uploading {audio_path.name}",
        ) as pg:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
                pg.update(len(chunk))


class Message(BaseModel):
    role: str
    content: str


class Conversation(RootModel):
    root: list[Message]

    def model_dump(self, *args, **kwargs):
        return [message.model_dump(*args, **kwargs) for message in self.root]

    def add(self, role: str, content: str):
        self.root.append(Message(role=role, content=content))

    def clear(self):
        self.root = [m for m in self.root if m.role == "system"]


one_shot_convo = Conversation(
    [
        Message(
            role="system",
            content="""\
you are a medical scribe. you will be provided transcripts from doctors' consultations with patients. '
review each transcript then write out a concise report that summarizes important findings.

only respond with the summary of the consultation. do not include any other information.
""",
        ),
        Message(
            role="user",
            content="""\
## consultation_transcript

This is the second panel. I had one in 2017, another one last year. So let me take it again. You have abstract 2016 and 2018. Last year no. Okay. 2023. 2023. 1623. Okay. So after the last one, I started, I hit my head on the wall before I fell. Oh. That one was last year. Did you hit with your front head or the back head? Here. And I really pushed here and here. So since then my vision has been clouded. Have you gone to see any eye doctor? No. This is the first visit? Yes. Okay, fine. Apart from the bloody vision, is there any other complaints? This one, I can't see this one. I can't see. Not the left eye, huh? Okay, fine. Can you put your chin here? So, for how long now you have BP problem? Since July. Last year or 2016? Last year. So what did... BP problem, it's been a long time. Like how many years? Since 2001. No, BP. No, no, no, no, no. 2023. It's not true. You've had stroke in 2016. Yes. So your problem should have been there by that time 2001 and she's having bp yes does she take her drugs so she is okay fine okay does she have sugar for you no no only bp okay and now i problem. Just hold on. Relax, relax. Open your eye. history of two previous strokes 2015 and 2023 known hypertensive of 23 years duration compliant on medication complaints of blurry vision, both eyes. Thank you. Look straight mother, look at my hair. Examination findings please. Just let me know if her VA is 6 over 12 on the right, one has glasses, this is over 15. Then I will be, and this is this woman, 13 and 14. Okay. Are you ready for the investigations? Yes, ma'am. investigations. Mama, look straight to don't bring your face down, please. Now, anterior segment. Cognac tawa is brown. Cornea is clear. Pimple is round. Sun turn reactive. Lens opacity 2+. Eyelids are cornea is clear, pupil is round, sun-turned, reactive, less opacity, two pluses, iris arthrophic. In the cornea you have acus senilis. Media is hazy so I can't see anything at the back. For the left eye, For the left side, conjunctiva is brown and injected. There is anterior ble pluses here. Media is lazy, I can't see the media. Okay. Now, mama, remove your face. What we are going to do... What we are going to do for her is to send her for B-scan, random blood sugar, and dilated eye examination also and refraction. Okay, let's do all the investigation. Dilated eye examination and refraction. Refraction before dilated eye examination.
""",
        ),
        Message(
            role="assistant",
            content="""\
The patient is an elderly individual with a history of two previous strokes, one in 2015 and another in 2023. They have been a known hypertensive for 23 years and are currently compliant with their medication. The patient reports experiencing blurry vision in both eyes for an unspecified duration. The patient also has a history of high blood pressure since 2001.\n\nExamination findings indicate that the patient's visual acuity (VA) is 6/12 on the right eye, with the left eye being worse at 13/14. The patient has cataracts in both eyes, with a lens opacity of 2+. The left eye has a brown and injected conjunctiva, with anterior blepharitis and a hazy media, making it difficult to see the back of the eye.\n\nInvestigations ordered include a B-scan, random blood sugar, and dilated eye examination, as well as refraction.
""",
        ),
    ]
)


def create_client():
    global settings
    return AsyncGroq(api_key=settings.GROQ_API_KEY)


async def generate_response(client: Groq, user_message: str):
    global one_shot_convo
    convo = one_shot_convo.model_copy()
    convo.add(role="user", content=user_message)

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=convo.model_dump(),
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        stream=False,
    )

    return completion.choices[0].message.content
