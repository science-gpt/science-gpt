import json
import logging
from typing import Optional

import toml
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer
from orchestrator.utils import SingletonMeta

SURVEY_LEVEL = 25  # Custom log level for survey responses
AZURE_LOGS_CONNECTION_STRING = (
    toml.load("secrets.toml").get("azure-logging", {}).get("connection_string", None)
)


class LogManager(logging.Logger, metaclass=SingletonMeta):
    def __init__(
        self,
        name,
        azure_connection_string=AZURE_LOGS_CONNECTION_STRING,
        level=logging.NOTSET,
    ):
        """
        Custom logging class.
        Log level (DEBUG, INFO, WARNING, etc..) must be specified at instantiation. Note that INFO corresponds to
        a log level of 20, so to capture user survey responses, the level must at least INFO.
        When writing a log, the xtra parameter can be used to add extra information to the log.
        Example usage:
        logger.info("This is an info message", xtra={"user": "john_doe", "action": "login", "status": "success"})
        """
        super().__init__(name, level)
        logging.addLevelName(SURVEY_LEVEL, "SURVEY")
        self.extra_info = {"user": "unknown"}

        # configure console logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(user)s - %(message)s")
        handler.setFormatter(formatter)
        self.addHandler(handler)

        # configure azure logging
        if not azure_connection_string:
            logging.warning(
                "Azure connection string not provided. Azure logging will not be enabled."
            )
        else:
            logging.info("Azure logging enabled.")
            azure_handler = AzureLogHandler(connection_string=azure_connection_string)
            azure_handler.setFormatter(formatter)
            self.addHandler(azure_handler)

            self.tracer = Tracer(
                exporter=AzureExporter(
                    connection_string=azure_connection_string,
                    sampler=ProbabilitySampler(1.0),
                )
            )

    def set_user(self, user: str):
        self.extra_info["user"] = user

    def set_configs(self, configs: dict[dict]):
        """takes a json dump of the configs"""
        flatdict = {
            subkey: subvalue
            for key, value in configs.items()
            for subkey, subvalue in value.items()
        }
        self.extra_info = self.extra_info | flatdict

    def info(self, msg, *args, xtra=None, configs: Optional[dict] = None, **kwargs):
        if configs is not None:
            self.set_configs(configs)
        extra_info = {
            "user": self.extra_info["user"],
            "custom_dimensions": self.extra_info | (xtra if xtra is not None else {}),
        }

        super().info(msg, *args, extra=extra_info, **kwargs)

    def survey(self, msg, *args, xtra=None, configs: Optional[dict] = None, **kwargs):
        extra_info = self.extra_info | (xtra if xtra is not None else {})
        if self.isEnabledFor(SURVEY_LEVEL):
            self._log(SURVEY_LEVEL, msg, args, extra=extra_info, **kwargs)


logger = LogManager(
    __name__,
    azure_connection_string=AZURE_LOGS_CONNECTION_STRING,
    level=logging.INFO,
)
