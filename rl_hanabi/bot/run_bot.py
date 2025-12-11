import logging
import os

from dotenv import load_dotenv
from rl_hanabi.bot.random_bot import RandomBot



def main():
    # Expect env: HANABI_USERNAME, HANABI_PASSWORD (optionally with suffixes)
    load_dotenv()

    # Basic logging configuration for the bot and connection.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not os.environ.get("HANABI_USERNAME") or not os.environ.get("HANABI_PASSWORD"):
        print("Set HANABI_USERNAME and HANABI_PASSWORD in env.")
        return

    bot = RandomBot()
    logger = logging.getLogger("rl_hanabi.run_bot")
    bot.on_open = lambda: logger.info("WebSocket open")
    bot.on_close = lambda code, reason: logger.info("WebSocket closed: %s %s", code, reason)
    bot.on_error = lambda e: logger.error("WebSocket error: %s", e)

    bot.connect()

    # Keep process alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        bot.close()


if __name__ == "__main__":
    main()
