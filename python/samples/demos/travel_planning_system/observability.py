# Copyright (c) Microsoft. All rights reserved.

# Copyright (c) Microsoft. All rights reserved.

import logging
import os
import sys

from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter, AzureMonitorTraceExporter
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import set_tracer_provider
from opentelemetry.trace.span import format_trace_id

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
    "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
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


# class CustomBatchSpanProcessor(BatchSpanProcessor):
#     @override
#     def on_end(self, span: ReadableSpan):
#         if span.name.startswith("agent_runtime"):
#             # Skip spans that are part of the agent runtime.
#             return
#         super().on_end(span)



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
    
    # 1. Azure Monitor Exporter
    if APPINSIGHTS_CONNECTION_STRING:
        try:
            exporters.append(AzureMonitorTraceExporter(connection_string=APPINSIGHTS_CONNECTION_STRING))
            print("✅ Azure Monitor trace export enabled.")
        except Exception as e:
            print(f"❌ Failed to initialize Azure Monitor exporter: {e}")


    # 4. Langfuse Exporter (using proper Langfuse SDK)

    import base64
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry import trace
    from langfuse import Langfuse

    # Langfuse Configuration
    langfuse_public_key = "pk-lf-ec644ab5-f243-4ea1-83fa-0255543d3e50"
    langfuse_secret_key = "sk-lf-5876b73d-3c8c-4dd0-94c5-d4c1acb5eb0e"
    langfuse_host = "https://us.cloud.langfuse.com"

    exporters = []

    if langfuse_public_key and langfuse_secret_key:
        try:

            # OTLP Exporter Setup
            auth_header = "Basic " + base64.b64encode(f"{langfuse_public_key}:{langfuse_secret_key}".encode()).decode()
            langfuse_exporter = OTLPSpanExporter(
                endpoint=f"{langfuse_host}/api/public/otel/v1/traces",
                headers={"Authorization": auth_header}
            )
            exporters.append(langfuse_exporter)
            print(f"🎯 Langfuse OTLP trace export enabled to: {langfuse_host}")

        except ImportError as ie:
            print(f"⚠️ Langfuse SDK not installed: {ie}")


        except Exception as e:
            print(f"❌ Failed to initialize Langfuse exporter: {e}")

    else:
        print("⚠️ Langfuse credentials not found. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")

    if not exporters:
        print("⚠️ No trace exporters configured - traces will not be sent.")
        # Add console exporter as fallback
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporters.append(ConsoleSpanExporter())
            print("📊 Fallback: Console exporter enabled")
        except Exception as e:
            print(f"❌ Even console exporter failed: {e}")
            return
    
    
    # Initialize tracer provider with filtering
    tracer_provider = TracerProvider(resource=resource)

    for exporter in exporters:
        tracer_provider.add_span_processor(CustomBatchSpanProcessor(exporter))
        #tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    set_tracer_provider(tracer_provider)
    print(f"✅ Tracing initialized with {len(exporters)} exporter(s).")

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