# Copyright (c) Microsoft. All rights reserved.

import logging
import os
import sys

from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter, AzureMonitorTraceExporter
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import set_tracer_provider
from opentelemetry.trace import NoOpTracerProvider
from opentelemetry.trace.span import format_trace_id
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

load_dotenv()

APPINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

# Create resource with service name from environment or default
service_name = os.getenv("OTEL_SERVICE_NAME", "TravelPlanningSystemDemo")
resource = Resource.create({
    ResourceAttributes.SERVICE_NAME: service_name,
    ResourceAttributes.SERVICE_VERSION: "1.0.0",
    "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development")
})



def set_up_logging():
    class KernelFilter(logging.Filter):
        """A filter to not process records from semantic_kernel."""

        # These are the namespaces that we want to exclude from logging for the purposes of this demo.
        namespaces_to_exclude: list[str] = [
            "semantic_kernel.functions.kernel_plugin",
            "semantic_kernel.prompt_template.kernel_prompt_template",
        ]

        def filter(self, record):
            return not any([record.name.startswith(namespace) for namespace in self.namespaces_to_exclude])

    exporters = []
    exporters.append(AzureMonitorLogExporter(connection_string=APPINSIGHTS_CONNECTION_STRING))

    # Create and set a global logger provider for the application.
    logger_provider = LoggerProvider(resource=resource)
    # Log processors are initialized with an exporter which is responsible
    # for sending the telemetry data to a particular backend.
    for log_exporter in exporters:
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    # Sets the global default logger provider
    set_logger_provider(logger_provider)

    # Create a logging handler to write logging records, in OTLP format, to the exporter.
    handler = LoggingHandler()
    # Add filters to the handler to only process records from semantic_kernel.
    handler.addFilter(logging.Filter("semantic_kernel"))
    handler.addFilter(KernelFilter())
    # Attach the handler to the root logger. `getLogger()` with no arguments returns the root logger.
    # Events from all child loggers will be processed by this handler.
    logger = logging.getLogger()
    logger.addHandler(handler)
    # Set the logging level to NOTSET to allow all records to be processed by the handler.
    logger.setLevel(logging.NOTSET)


class CustomBatchSpanProcessor(BatchSpanProcessor):
    @override
    def on_end(self, span: ReadableSpan):
        if span.name.startswith("agent_runtime"):  # group_chat_manager
            # Skip spans that are part of the agent runtime.
            return
        # Skip empty streaming message spans
        if span.name == "streaming_message_final":
            attributes = span.attributes or {}
            content_length = attributes.get("message.content_length", 0)
            if content_length == 0:
                return
        super().on_end(span)


def set_up_tracing():
    exporters = []
    
    # Add Azure Monitor exporter if connection string is available
    if APPINSIGHTS_CONNECTION_STRING:
        exporters.append(AzureMonitorTraceExporter(
            connection_string=APPINSIGHTS_CONNECTION_STRING
        ))
        print("📊 Azure Monitor trace export enabled")
    
    # Add Jaeger/OTLP exporter for local development
    jaeger_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                                "http://localhost:4318/v1/traces")
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=jaeger_endpoint)
        exporters.append(otlp_exporter)
        print(f"📊 Jaeger/OTLP trace export enabled: {jaeger_endpoint}")
    except Exception as e:
        print(f"⚠️  Failed to initialize OTLP exporter: {e}")

    # Initialize a trace provider for the application
    tracer_provider = TracerProvider(resource=resource)
    
    # Add all available exporters
    for exporter in exporters:
        # tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        tracer_provider.add_span_processor(CustomBatchSpanProcessor(exporter))
    
    # Sets the global default tracer provider
    set_tracer_provider(tracer_provider)
    
    if not exporters:
        print("⚠️  No trace exporters configured - traces will not be sent")
    else:
        print(f"✅ Tracing initialized with {len(exporters)} exporter(s)")


def enable_observability(func):
    """A decorator to enable observability for the demo."""

    async def wrapper(*args, **kwargs):
        set_up_logging()
        set_up_tracing()

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("main") as current_span:
            print(f"Trace ID: {format_trace_id(current_span.get_span_context().trace_id)}")
            return await func(*args, **kwargs)

    return wrapper
