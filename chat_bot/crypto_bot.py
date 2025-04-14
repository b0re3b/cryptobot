import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from utils.config import TELEGRAM_API

bot = Bot(token=TELEGRAM_API)
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "👋 Привіт!\n\n"
        "Я бот, який допоможе тобі у трейдингу, прогнозуючи курс криптовалют 📈.\n"
        "Вибери валюту, яка тебе цікавить, і часовий проміжок ⏳."
    )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
