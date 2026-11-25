# -*- coding: utf-8 -*-
"""Start the Auralis web app: python -m webapp.server [--port 8765]."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser(description="Auralis web app server (local only)")
    ap.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ap.add_argument("--no-resume", action="store_true",
                    help="mark the jobs a previous server left in flight as failed "
                         "instead of queueing them again")
    args = ap.parse_args()
    os.environ["AURALIS_RESUME_JOBS"] = "0" if args.no_resume else "1"
    import uvicorn
    uvicorn.run("webapp.server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
