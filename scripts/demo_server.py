# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""Static dev server for the browser demo (docs/).

Same as `python -m http.server` but sends Cache-Control: no-store, so the
browser always picks up freshly exported weights and edited JS — plain
http.server sends no cache headers and browsers heuristically cache the
module files, which serves stale code after an update.

Usage: python scripts/demo_server.py [--port 8080]
"""
import argparse
import functools
import http.server


class NoStoreHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dir", default="docs")
    args = ap.parse_args()
    handler = functools.partial(NoStoreHandler, directory=args.dir)
    with http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler) as srv:
        print(f"demo → http://localhost:{args.port}/", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
