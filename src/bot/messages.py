def welcome():
    return """\
Hello! I'm the demo bot for meddibia.
You can use me to test experimental features for meddibia.
"""


def unknown(messages: str = ""):
    txt = "I'm sorry, I didn't understand that command."
    if messages:
        txt += f"\nPlease {messages}"
    return txt


def transcribe():
    return """\
Please send me a consultation recording to transcribe.
"""
