from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

# Function to create users table (only run once)
def create_users_table():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Run table creation
create_users_table()

# Function to check login credentials
def check_login(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

# Function to create a new user
def create_user(username, password):
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:  # If username already exists
        return False

@app.route("/")
def home():
    return render_template("login.html")  # Ensure you have login.html in templates folder

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    if check_login(username, password):
        session["username"] = username  # Store username in session
        return redirect(url_for("monitor"))  # Redirect to monitoring page
    else:
        return "Invalid credentials! Try again. <a href='/'>Go Back</a>"

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if create_user(username, password):
            return "Signup successful! <a href='/'>Go to Login</a>"
        else:
            return "Username already exists! <a href='/signup'>Try Again</a>"

    return render_template("signup.html")

@app.route("/monitor")
def monitor():
    if "username" not in session:
        return redirect(url_for("home"))  # If not logged in, redirect to login

    return render_template("monitor.html")  # Ensure you have monitor.html

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
