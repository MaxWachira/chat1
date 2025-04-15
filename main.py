from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import nltk
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
import keras
from keras import models
from functools import wraps
from flask import make_response
import json
import pickle
from datetime import timedelta
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configure session to expire quickly and not be permanent
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Session expires after 30 minutes of inactivity
app.config['SESSION_COOKIE_SECURE'] = True  # Only send cookie over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent client-side JS from accessing the cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Protection against CSRF

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Max@123'
app.config['MYSQL_DB'] = 'login_details'

# Initialize MySQL
mysql = MySQL(app)

# Load NLP chatbot components
try:
    lemmatizer = WordNetLemmatizer()
    model = models.load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
except Exception as e:
    print(f"Error loading NLP components: {e}")
    # Handle error appropriately (maybe disable chatbot functionality)

# Decorator to add no-cache headers and check session
def nocache_and_session(view):
    @wraps(view)
    def decorated_function(*args, **kwargs):
        # Clear session if expired
        if 'loggedin' in session:
            session.permanent = True
            app.permanent_session_lifetime = timedelta(minutes=30)
        
        response = make_response(view(*args, **kwargs))

        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = ""
        return response
    return decorated_function

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    confidence_threshold = 0.7
    return classes[np.argmax(res)] if np.max(res) > confidence_threshold else "fallback"

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."


genai.configure(api_key="AIzaSyBvoKd16m3vPHJZvEBGCA8AW3fS4bDGbrs")

model_gemini = genai.GenerativeModel('gemini-1.5-flash')

def ask_gpt(question):
  try:
        response = model_gemini.generate_content(question)
        return response.text.strip()
  except Exception as e:
        return f"Error using Gemini: {str(e)}"


@app.route('/')
@nocache_and_session
def home():
    if 'loggedin' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/chatbot')
@nocache_and_session
def chatbot():
    if 'loggedin' in session:
        return render_template('chatbot.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
@nocache_and_session
def chat():
    if 'loggedin' not in session:
        return jsonify({'response': 'Please login to use the chatbot'})
    
    user_message = request.json.get('message')
    tag = predict_class(user_message)
    if tag == "fallback":
         response = ask_gpt(user_message)
    else:
        response = get_response(tag)
    return jsonify({'response': response})

@app.route('/login', methods=['GET', 'POST'])
@nocache_and_session
def login():

    session.clear()
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM new_table WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        
        if account:
            # Create new session
            session.permanent = False
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return redirect(url_for('home'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/register', methods=['GET', 'POST'])
@nocache_and_session
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM new_table WHERE username = %s', (username,))
        account = cursor.fetchone()
        
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        elif len(username) < 3:
            msg = 'Username must be at least 3 characters long!'
        elif username.isdigit():
            msg = 'Username cannot consist of only numbers!'
        elif not email.endswith(('.com', '.net', '.org', '.edu', '.co', '.ke')): 
            msg = 'Email must have a valid domain (e.g., .com, .net, .org)!'
        elif len(password) < 8:
            msg = 'Password must be at least 8 characters long!'
        else:
            cursor.execute('INSERT INTO new_table VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('register.html', msg=msg)

@app.route('/logout')
@nocache_and_session
def logout():
    # Clear the session and all session data
    session.clear()
    # Ensure the session cookie is deleted
    response = make_response(redirect(url_for('login')))
    response.set_cookie('username', '',expires=0)
    response.set_cookie('password', '',expires=0)
    #response.set_cookie('session', '', expires=0)
    #response.set_cookie('session', '', expires=0, path='/', secure=True, httponly=True, samesite='Lax')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
@app.route('/training')
def training():
    if 'loggedin' in session:
        return render_template('training.html')
    return redirect(url_for('login'))
@app.route('/add-intent', methods=['POST'])
@nocache_and_session
def add_intent():
    if 'loggedin' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    try:
        data = request.get_json()
        # Validate required fields
        if not data or 'tag' not in data or 'patterns' not in data or 'responses' not in data:
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
        
        # Create new intent structure
        new_intent = {
            'tag': data['tag'],
            'patterns': data['patterns'],
            'responses': data['responses']
        }
        
        # Add to existing intents
        intents['intents'].append(new_intent)
        
        # Save to file
        with open('intents.json', 'w') as file:
            json.dump(intents, file, indent=4)
            
        return jsonify({
            'success': True,
            'message': 'Intent added successfully! Retrain the model to apply changes.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to add intent: {str(e)}'
        }), 500

@app.route('/retrain', methods=['POST'])
@nocache_and_session
def retrain_model():
    # Ensure only logged-in users can trigger retraining; optionally restrict further by user role.
    if 'loggedin' not in session:
        return jsonify({"error": "Unauthorized access!"}), 403
    try:
        # Load new intents data from the intents.json file
        with open('intents.json', 'r') as f:
            new_intents = json.load(f)

        new_words = []
        new_classes = []
        documents = []
        ignore_letters = ['?', '!', '.', ',']

        # Process the intents file for training data
        for intent in new_intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                new_words.extend(word_list)
                documents.append((word_list, intent['tag']))
                if intent['tag'] not in new_classes:
                    new_classes.append(intent['tag'])

        new_words = sorted(set(lemmatizer.lemmatize(word.lower()) for word in new_words if word not in ignore_letters))
        new_classes = sorted(set(new_classes))
        pickle.dump(new_words, open("words.pkl", "wb"))
        pickle.dump(new_classes, open("classes.pkl", "wb"))

        training = []
        output_empty = [0] * len(new_classes)
        for document in documents:
            bag = []
            word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
            for word in new_words:
                bag.append(1 if word in word_patterns else 0)
            output_row = list(output_empty)
            output_row[new_classes.index(document[1])] = 1
            training.append(bag + output_row)

        random.shuffle(training)
        training = np.array(training)
        train_x = training[:, :len(new_words)]
        train_y = training[:, len(new_words):]

        # Define and train the model
        new_model = models.Sequential()
        new_model.add(keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
        new_model.add(keras.layers.Dropout(0.5))
        new_model.add(keras.layers.Dense(64, activation="relu"))
        new_model.add(keras.layers.Dense(len(train_y[0]), activation="softmax"))
        sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        new_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
        new_model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        new_model.save("chatbot_model.h5")

        # Update the global variables so the application uses the newly trained model immediately
        global model, words, classes, intents
        model = new_model
        words = new_words
        classes = new_classes
        intents = new_intents

        return jsonify({"message": "Chatbot model retrained successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)