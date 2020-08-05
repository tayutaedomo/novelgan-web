import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
APP_SETTINGS = os.getenv('APP_SETTINGS', 'config.DevelopmentConfig')
app.config.from_object(APP_SETTINGS)

auth = HTTPBasicAuth()


from lib.cyclegan import CycleGAN, IMG_COLS, IMG_ROWS


gan = None

try:
    gan = CycleGAN()
    gan.init()
    gan.load_models()
except Exception as e:
    print(e)


@auth.verify_password
def verify_password(username, password):
    u = os.getenv('BASICAUTH_USERNAME', 'novelgan')
    p = os.getenv('BASICAUTH_PASSWORD', 'novels')
    return username == u and password == p


@app.route('/')
@auth.login_required
def index():
    #return 'Hello World!'
    return redirect(url_for('cyclegan'))


@app.route('/cyclegan', methods=['GET', 'POST'])
def cyclegan():
    local = {
        'predicted': False
    }

    if request.files.get('image'):
        file = request.files['image']
        local['file'] = file

        content = file.read()
        local['file_base64'] = str(base64.b64encode(content), 'utf-8')

        img = Image.open(BytesIO(content))
        img_resize = img.resize((IMG_COLS, IMG_ROWS))
        img_np = np.array(img_resize) / 127.5 - 1.
        img_reshape = img_np.reshape(1, IMG_ROWS, IMG_COLS, 3)

        x = img_reshape
        y = gan.generate_image(x)
        y = (0.5 * y + 0.5) * 255
        y_img = Image.fromarray(np.uint8(y[0]))

        buffered = BytesIO()
        y_img.save(buffered, format='JPEG')
        y_img_str = str(base64.b64encode(buffered.getvalue()), 'utf-8')

        local['predicted'] = True
        local['y_proba'] = y_img_str

    return render_template('cyclegan.html', local=local)


#@app.errorhandler(403)
#@app.errorhandler(404)
@app.errorhandler(500)
@app.errorhandler(503)
def error_handler(error):
    print(error)
    msg = 'Error: {code}\n'.format(code=error.code)
    return msg, error.code


if __name__ == '__main__':
    print('Config:', APP_SETTINGS)
    app.run()

