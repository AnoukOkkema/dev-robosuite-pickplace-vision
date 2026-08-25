import logging
import logging.config
from pathlib import Path

import yaml


class LoggingConfigurator:
    """Configures the application's logging system from config/logging.yaml."""

    _initialized = False

    @classmethod
    def setup(cls, log_filename: str = "main.log") -> logging.Logger:
        """
        Configures the logging system using the YAML configuration file.

        Only takes effect on the first call per process - later calls just
        return the already-configured logger.

        Args:
            log_filename (str): Name of the log file inside logs/, e.g.
                "train_yolo.log" -- lets each entrypoint script write to
                its own file instead of sharing logs/main.log.

        Returns:
            logging.Logger: Configured logger instance.

        Example:
            logger = LoggingConfigurator.setup("train_yolo.log")
            logger.info("Application started")
        """

        if cls._initialized:
            return logging.getLogger(__name__)

        # Ensure the logs directory exists
        Path("logs").mkdir(exist_ok=True)

        # Load the YAML configuration
        config_path = Path("config", "logging.yaml")

        try:
            if config_path.exists():
                with open(config_path, "r") as file:
                    config = yaml.safe_load(file)

                    if "file" in config.get("handlers", {}):
                        config["handlers"]["file"]["filename"] = str(Path("logs", log_filename))

                    logging.config.dictConfig(config)
            else:
                raise FileNotFoundError(f"Logging config not found at {config_path}")
        except Exception as e:
            print(f"Failed to load logging configuration: {e}")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-8s | - %(message)s",
                datefmt="%H:%M:%S",
            )
        finally:
            cls._initialized = True

        cls._route_ultralytics_logs_to_file()

        return logging.getLogger(__name__)

    @staticmethod
    def _route_ultralytics_logs_to_file() -> None:
        """
        Attaches this app's file handler(s) to Ultralytics' own logger.

        Ultralytics sets `propagate = False` on its "ultralytics" logger
        and attaches its own stdout handler (see
        `ultralytics.utils.set_logging`), so its training/validation
        output normally only ever prints to the console and never reaches
        our log file. This adds our file handler directly to that logger
        (leaving its console handler untouched) so the same output also
        lands in logs/*.log.

        Safe to call regardless of whether `ultralytics` has been imported
        yet - `logging.getLogger("ultralytics")` just fetches/creates the
        logger by name either way, and the handler stays attached even if
        Ultralytics itself imports and configures that logger afterwards.

        Returns:
            None
        """

        file_handlers = [
            handler
            for handler in logging.getLogger().handlers
            if isinstance(handler, logging.FileHandler)
        ]

        ultralytics_logger = logging.getLogger("ultralytics")

        for handler in file_handlers:
            if handler not in ultralytics_logger.handlers:
                ultralytics_logger.addHandler(handler)
