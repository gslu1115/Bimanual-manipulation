"""Loopback JSON adapter: Isaac imports no ZMQ or GraspGen dependencies.

This diagnostic transport returns raw gripper-base poses, never executable TCPs.
Local IDs/timestamps are echoed by this adapter, not invented official fields.
"""
import base64
import json
import time
import uuid
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from workstation.models.graspgen_client import GraspGenClient, VisualClouds, validate_grasps

MAX_BODY=1024*1024

def dispatch(request,client):
    if request.get('action') != 'infer': raise ValueError('Only diagnostic infer is supported')
    count=request['count']
    if type(count) is not int or not 200 <= count <= 20000: raise ValueError('Invalid point count')
    raw=base64.b64decode(request['points'],validate=True)
    if len(raw) != count*12: raise ValueError('Malformed float32 cloud payload')
    points=np.frombuffer(raw,dtype='<f4').reshape(count,3)
    result=client.infer(VisualClouds(points,np.empty((0,3),np.float32),float(request['time_s'])),
                        num_grasps=request.get('num_grasps',64))
    return dict(id=request['id'],time_s=request['time_s'],
        poses=result['base_poses_workcell'].tolist(),scores=result['scores'].tolist(),
        metadata=result['metadata'],timing=result['timing'])

def serve(args):
    client=GraspGenClient(args.graspgen_endpoint,timeout_ms=30000)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.path != '/infer': raise ValueError('Unknown route')
                count=int(self.headers.get('Content-Length','0'))
                if not 0<count<=MAX_BODY: raise ValueError('Request exceeds bounded payload')
                self.connection.settimeout(35)
                reply=dispatch(json.loads(self.rfile.read(count)),client)
                status=200
            except Exception as error:
                reply={'error':str(error)};status=400
            payload=json.dumps(reply,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers()
            self.wfile.write(payload)
    HTTPServer(('127.0.0.1',args.graspgen_proxy_port),Handler).serve_forever()

class GraspGenProxy:
    def __init__(self,endpoint='http://127.0.0.1:5558/infer',timeout_s=35):
        self.endpoint,self.timeout_s=endpoint,timeout_s

    def infer(self,clouds,num_grasps=64):
        if clouds.frame!='workcell' or clouds.unseen_space_is_free: raise ValueError('Invalid cloud frame')
        points=np.asarray(clouds.target_m,dtype='<f4')
        if points.ndim!=2 or points.shape[1]!=3 or not 200<=len(points)<=20000 or not np.isfinite(points).all():
            raise ValueError('Invalid visual target cloud')
        request=dict(action='infer',id=uuid.uuid4().hex,time_s=clouds.time_s,count=len(points),
            points=base64.b64encode(points.tobytes()).decode(),num_grasps=num_grasps)
        started=time.monotonic()
        http=urllib.request.Request(self.endpoint,json.dumps(request,allow_nan=False).encode(),
            {'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(http,timeout=self.timeout_s) as response:
            raw=response.read(MAX_BODY+1)
        if len(raw)>MAX_BODY:raise ValueError('Oversized proxy response')
        reply=json.loads(raw)
        return self.validate_reply(reply,request,time.monotonic()-started)

    def validate_reply(self,reply,request,elapsed):
        if elapsed>self.timeout_s:raise RuntimeError('Late GraspGen response')
        if reply.get('id')!=request['id'] or reply.get('time_s')!=request['time_s']:
            raise ValueError('Response does not belong to this observation request')
        if reply.get('metadata',{}).get('gripper_name')!='franka_panda':raise ValueError('Wrong gripper')
        poses,scores=validate_grasps(reply['poses'],reply['scores'])
        return dict(base_poses_workcell=poses,scores=scores,metadata=reply['metadata'],
            timing=reply.get('timing',{}),request_id=request['id'],observation_time_s=request['time_s'],
            round_trip_s=elapsed,TCP_calibration_verified=False,robot_path_checked=False,
            executable=False,scores_are_success_probability=False)
