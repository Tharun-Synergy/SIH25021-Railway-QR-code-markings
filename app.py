from flask import Flask, request, redirect, url_for, send_file, render_template, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from db import init_db, get_db, verify_user, is_unique, save_form_data
import uuid
import qrcode
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

init_db()

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            username = username.strip()
            password = password.strip()
            if verify_user(username, password):
                login_user(User(username))
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error="❌ Login failed: unauthorized user")
        else:
            return render_template('login.html', error="❌ Login failed: username or password missing")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
@login_required
def generate():
    data = request.get_json()
    required = ['companyName', 'idNo', 'registration', 'date', 'time', 'productName', 'filetype']
    if not all(data.get(k) for k in required):
        return jsonify({'error': 'Missing fields'}), 400

    unique_str = str(uuid.uuid4())
    while not is_unique(unique_str):
        unique_str = str(uuid.uuid4())

    save_form_data(unique_str, data['companyName'], data['idNo'], data['registration'],
                   data['date'], data['time'], data['productName'], data['filetype'])

    message = f"Unique ID: {unique_str}\n" + "\n".join([f"{k}: {v}" for k, v in data.items()])
    if data['filetype'] == 'png':
        qr = qrcode.make(message)
        buffer = io.BytesIO()
        qr.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png', as_attachment=True, download_name=f"{unique_str}.png")
    elif data['filetype'] == 'g.code':
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(message)
        qr.make(fit=True)
        matrix = qr.modules
        gcode = ["G21", "G90", "G00 Z5"]
        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                if cell:
                    gcode += [f"G00 X{x} Y{y}", "G01 Z-1 F100", "G00 Z5"]
        gcode.append("M30")
        buffer = io.BytesIO()
        buffer.write('\n'.join(gcode).encode())
        buffer.seek(0)
        return send_file(buffer, mimetype='text/plain', as_attachment=True, download_name=f"{unique_str}.gcode")
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == "__main__":
    app.run(debug=True)
