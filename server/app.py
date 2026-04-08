from __future__ import annotations

import os

import uvicorn

from app import app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    main()
