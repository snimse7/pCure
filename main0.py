from telegram.ext import *
from image_process0 import *

BOT_API_TOKEN = "6044452467:AAEb2OrWuFXgIp5ve1mcUn5GCTYyqJ0pbmM"


def start(update, context):
    update.message.reply_text(
        f"Welcome to the P-cure Bot\n" f"\n" f"Send an image to detect disease\n" f"\n"
    )


def image_hamdler(update, context):
    file = update.message.photo[-1].get_file()
    image_processing(file)
    update.message.reply_text(get_response_english())


if __name__ == "__main__":
    updater = Updater(BOT_API_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, image_hamdler))

    updater.start_polling(1.0)
    updater.idle()
