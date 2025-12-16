import logging
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from dotenv import load_dotenv
from rl_hanabi.bot.mcts_bot import MCTSBot

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

    # You can adjust time_limit_ms here
    bot = MCTSBot(time_limit_ms=1000)
    logger = logging.getLogger("rl_hanabi.run_mcts_bot")
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
