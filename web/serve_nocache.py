import http.server, socketserver, sys, os
os.chdir(sys.argv[2] if len(sys.argv)>2 else '.')
PORT=int(sys.argv[1])
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma','no-cache')
        super().end_headers()
socketserver.TCPServer.allow_reuse_address=True
with socketserver.TCPServer(('0.0.0.0',PORT),H) as s:
    print(f'serving {os.getcwd()} on :{PORT} (no-cache)'); s.serve_forever()
