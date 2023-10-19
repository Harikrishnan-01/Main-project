from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import pred3
from pred3 import *
import pred4
from pred4 import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my-channel'  # Set a secret key for secure communication
socketio = SocketIO(app, cors_allowed_origins='*')  # Enable CORS for WebSocket connections
CORS(app)

final_string = ''   # Global variable to store the final string

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('process_frame')
def handle_process_frame(data):
    image_data = data['image']
    sign_language = data['sign_language']

    # Process the frame here
    encoded_data = image_data.split(',')[1].encode()
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if sign_language == 'indian':
        encoded_frame, final_string = process_frame_indian(frame)
    elif sign_language == 'american':
        encoded_frame, final_string = process_frame_american(frame)
    else:
        emit('process_frame_error', {'error': 'Invalid sign language'})
        return

    emit('process_frame_result', {'signText': final_string, 'encodedFrame': encoded_frame})



def process_frame_indian(frame):
    output_frame, final_string = pred4.fun(frame)
    ret, buffer = cv2.imencode('.jpg', output_frame)
    encoded_frame = base64.b64encode(buffer)
    return encoded_frame.decode(), final_string

def process_frame_american(frame):
    output_frame, final_string = pred3.fun(frame)
    ret, buffer = cv2.imencode('.jpg', output_frame)
    encoded_frame = base64.b64encode(buffer)
    return encoded_frame.decode(), final_string

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    image_data = request.json['image']
    sign_language = request.json['sign_language']

    encoded_data = image_data.split(',')[1].encode()
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if sign_language == 'indian':
        encoded_frame, final_string = process_frame_indian(frame)
    elif sign_language == 'american':
        encoded_frame, final_string = process_frame_american(frame)
    else:
        return jsonify(error='Invalid sign language')
    
    return jsonify(signText=final_string, encodedFrame=encoded_frame)

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global frame, final_string
    sign_language = request.json['sign_language']

    print(sign_language)

    if sign_language == 'indian':
        frame, final_string = pred3.fun('', clear_text=True)
    elif sign_language == 'american':
        frame, final_string = pred4.fun('', clear_text=True)
    else:
        return jsonify(error='Invalid sign language')

    print("Cleared text:", final_string)

    # Return the cleared text in the response
    return jsonify(clearedText=final_string)

if __name__ == '__main__':
    socketio.run(app, port=8000, debug=True)
