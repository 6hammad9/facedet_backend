from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import requests
import json

app = Flask(__name__)
@app.route('/api', methods=['POST'])
def api():
    r = request
    # convert string of image data to uint8
    body = jsonpickle.decode(r.data)

    # nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # build a response dict to send back to client
    response = {'message': 'image received'}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")
app.run(host="0.0.0.0", port=5001)

test_url = 'http://localhost/emacsPanel/api/getConnected.php'
# headers = {'content-type': 'text/html'}
# body = jsonpickle.encode({'cam_id_req': '4568ErrtFLli3s298zZVv'})
body = {'cam_id_req': '4568ErrtFLli3s298zZVv'}
# response = requests.post(test_url, data=body, headers=headers)
res = requests.post(test_url, data=body)
print(res.text)

#%% threading
#
# import time, threading
# def foo():
#     print(time.ctime())
#     threading.Timer(10, foo).start()
#
# foo()
#
#
# #%% image io
# im_file = request.files["image"]
# im_bytes = im_file.read()
# im = Image.open(io.BytesIO(im_bytes))


#%% json response dierectly from df

# results.pandas().xyxy[0].to_json(orient="records")