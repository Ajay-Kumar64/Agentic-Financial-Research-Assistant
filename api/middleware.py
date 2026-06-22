"""
api/middleware.py — Request logging and error handling.
"""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("agent.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        latency = (time.time() - start) * 1000

        logger.info(
            f"{request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Latency: {latency:.0f}ms"
        )
        response.headers["X-Response-Time"] = f"{latency:.0f}ms"
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error: {e}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(e)}
            )