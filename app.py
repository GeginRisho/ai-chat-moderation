from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room
import pickle

# =========================
# 🚀 INIT APP
# =========================

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)

# =========================
# 🤖 LOAD TRAINED MODEL
# =========================

model = pickle.load(open("cyberbullying_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

print("✅ Model Loaded Successfully")

# =========================
# 🏠 ROUTES
# =========================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/chat")
def chat():
    room = request.args.get("room")
    username = request.args.get("username")
    return render_template("chat.html", room=room, username=username)

# =========================
# 🔌 SOCKET EVENTS
# =========================

@socketio.on("join")
def handle_join(data):
    room = data["room"]
    join_room(room)


@socketio.on("send_message")
def handle_message(data):
    room = data["room"]
    username = data["username"]
    message = data["message"]

    try:
        # 🔹 Transform message
        transformed = vectorizer.transform([message])

        # 🔹 Predict toxicity
        prediction = model.predict(transformed)[0]

        print("Prediction:", prediction)

        # ⚠ If dataset uses 1 = toxic
        if prediction == 1:
            emit("blocked", {
                "username": "System",
                "message": "⚠ Message blocked by AI"
            }, room=request.sid)
            return

        # ✅ Safe → Broadcast to room
        emit("message", {
            "username": username,
            "message": message
        }, room=room)

    except Exception as e:
        print("Error during prediction:", e)

# =========================
# 🏁 RUN SERVER
# =========================

if __name__ == "__main__":
    socketio.run(app, debug=True)