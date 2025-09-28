import sqlite3
from flask import g
import os

DATABASE = os.path.join(os.path.dirname(__file__), '..', 'qr_api.db')

def get_db():
    if '_database' not in g:
        g._database = sqlite3.connect(DATABASE)
        g._database.row_factory = sqlite3.Row
    return g._database

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS qr_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_string TEXT UNIQUE,
            company_name TEXT,
            product_id TEXT,
            registration_no TEXT,
            date TEXT,
            time TEXT,
            product_name TEXT,
            file_type TEXT
        )''')

        # Insert two fixed users
        users = [('admin', 'admin123'), ('staff', 'staff456')]
        for username, password in users:
            try:
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            except sqlite3.IntegrityError:
                pass  # User already exists

        conn.commit()

def verify_user(username, password):
    row = get_db().execute('SELECT password FROM users WHERE username = ?', (username,)).fetchone()
    return row and row['password'] == password

def is_unique(unique_str):
    return get_db().execute('SELECT 1 FROM qr_data WHERE unique_string = ?', (unique_str,)).fetchone() is None

def save_form_data(*args):
    db = get_db()
    db.execute('''INSERT INTO qr_data (unique_string, company_name, product_id, registration_no, date, time, product_name, file_type)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', args)
    db.commit()