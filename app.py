from flask import Flask, render_template, Response
import cv2
import cloak

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Initialize background once at the start
cloak.init_background(camera)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success or frame is None:
            continue  # Skip bad frames

        processed_frame = cloak.apply_cloak_effect(frame)
        if processed_frame is None:
            continue  # Skip if cloak logic failed

        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

