"""One local motion-plan service for the isolated MoveIt runtime."""
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
from tempfile import TemporaryDirectory
import yaml


class Planner(socketserver.StreamRequestHandler):
    def handle(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/'request.json'
            result=Path(folder)/'result.yaml'
            source.write_bytes(self.rfile.readline())
            subprocess.run(['/moveit_build/plan_pair',str(source),str(result)],timeout=40,check=False)
            response=yaml.safe_load(result.read_text())
            self.wfile.write(json.dumps(response,allow_nan=False).encode()+b'\n')


if __name__=='__main__':
    address=sys.argv[1]
    with socketserver.UnixStreamServer(address,Planner) as server:
        os.chmod(address,0o600)
        server.serve_forever()
