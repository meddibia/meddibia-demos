from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    ConversationHandler,
    filters,
    MessageHandler,
)
import asyncio
import httpx

from settings import Settings
import bot_utils as utils
import messages as msg

settings = Settings()
logger = utils.get_logger()
GROQ = utils.create_client()

WAITING_FOR_AUDIO, TRANSCRIBING = range(2)
JOBS = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("/start command from user_id: %s" % update.effective_user.id)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg.welcome())


async def start_transcribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(msg.transcribe())
    return WAITING_FOR_AUDIO


async def audio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global JOBS

    if update.message.audio or update.message.voice or update.message.document:
        audio_file = (
            update.message.audio or update.message.voice or update.message.document
        )
        file_size = audio_file.file_size / 1024 / 1024
        duration = audio_file.duration

        logger.info(
            f"Audio file received: {audio_file.file_id} ({file_size:.2f} MB, {duration:.2f} seconds)"
        )
        await update.message.reply_text("Audio received. Transcription started.")

        job = {
            "id": None,
            "chat_id": update.effective_chat.id,
            "user_id": update.effective_user.id,
            "status": "starting",
            "audio_file": audio_file,
            "name": utils.get_job_name(update.effective_chat.id, audio_file.file_id),
        }
        context.job_queue.run_once(
            transcription_job, when=0, data=job, name=job["name"]
        )

        JOBS[job["user_id"]] = job

        return TRANSCRIBING
    else:
        breakpoint()
        await update.message.reply_text("Please send an audio file or voice message.")
        return WAITING_FOR_AUDIO


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    return ConversationHandler.END


async def transcription_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    global settings, JOBS, GROQ

    job = context.job.data
    user_id = job["user_id"]
    chat_id = job["chat_id"]

    audio_file = await job["audio_file"].get_file()
    local_file = await audio_file.download_to_drive(
        custom_path=utils.create_tmpfile(audio_file)
    )

    # send the audio file
    async with httpx.AsyncClient() as client:
        response = await client.post(
            settings.transcribe_url,
            headers=settings.transcribe_headers,
            files={
                "file": (local_file.name, utils.FileGenerator(local_file), "audio/mpeg")
            },
            timeout=10,
        )

    if response.status_code == 200:
        job_id = response.json()["call_id"]

        JOBS[user_id].update({"id": job_id, "status": "running"})

        while True:
            logger.info(f"Checking status of job {job_id}")
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.transcribe_status_url}/{job_id}",
                    timeout=10,
                )

            if response.status_code == 200:
                status = response.json()["status"]
                if status == "complete":
                    result = response.json()["transcript"]["text"].strip()
                    JOBS.pop(user_id)

                    logger.info(f"Transcription complete for job {job_id}")

                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"Transcription complete:\n{result}",
                    )

                    summary = await utils.generate_response(
                        client=GROQ,
                        user_message=f"""
## consultation transcript

{result}
""",
                    )

                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"Consultation summary:\n{summary}",
                    )

                    break

            await asyncio.sleep(10)
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Error transcribing audio file.",
        )


async def job_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in JOBS:
        job = JOBS[user_id]
        await update.message.reply_text(f"Transcription job status: {job['status']}")


def main() -> None:
    global settings

    app = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("transcribe", start_transcribe)],
        states={
            WAITING_FOR_AUDIO: [
                MessageHandler(
                    filters.AUDIO | filters.VOICE | filters.Document.AUDIO,
                    audio_handler,
                )
            ],
            TRANSCRIBING: [
                CommandHandler("status", job_status),
            ],
        },
        fallbacks=[MessageHandler(filters.Regex("^done$"), audio_handler)],
    )
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("status", job_status))

    logger.info("Starting the bot")
    app.run_polling()
    logger.info("Bot stopped")


if __name__ == "__main__":
    main()
