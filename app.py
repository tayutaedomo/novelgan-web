import os
import base64
from flask import Flask, render_template, request, redirect, url_for
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
APP_SETTINGS = os.getenv('APP_SETTINGS', 'config.DevelopmentConfig')
app.config.from_object(APP_SETTINGS)

auth = HTTPBasicAuth()


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
    local = {}

    if request.files.get('image'):
        file = request.files['image']
        local['file'] = file

        content = file.read()
        local['file_base64'] = str(base64.b64encode(content), 'utf-8')

    return render_template('cyclegan.html', local=local)


if __name__ == '__main__':
    print('Config:', APP_SETTINGS)
    app.run()

