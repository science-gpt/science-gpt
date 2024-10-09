import logging

from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer


class LogManager(logging.Logger):
    def __init__(self, name, azure_connection_string, level=logging.NOTSET):
        """
        Custom logging class.
        Log level (DEBUG, INFO, WARNING, etc..) must be specified at instantiation.
        """
        super().__init__(name, level)
        self.extra_info = None

        # configure console logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.addHandler(handler)

        # configure azure logging
        azure_handler = AzureLogHandler(azure_connection_string)
        azure_handler.setFormatter(formatter)
        self.addHandler(azure_handler)

        self.tracer = Tracer(
            exporter=AzureExporter(
                connection_string=azure_connection_string,
                sampler=ProbabilitySampler(1.0),
            )
        )

    def info(self, msg, *args, xtra=None, **kwargs):
        extra_info = xtra if xtra is not None else self.extra_info
        super().info(msg, *args, extra=extra_info, **kwargs)

    def survey(self, msg, *args, **kwargs):
        pass
