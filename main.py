import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# =======================================================
# 1. IMPORTURI ȘI RESURSE (Configurația)
# =======================================================

# Dimensiunea de intrare așteptată de model
INPUT_SIZE = (64, 64) 

# Numele fisierului cu model și construirea căii relative
MODEL_FILENAME = 'fine_tuned_model.hdf5'
script_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(script_dir, 'models', MODEL_FILENAME) 

# Calea pentru clasificatorul de față (Haar Cascade)
HAARCASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
HAARCASCADE_PATH = os.path.join(script_dir, 'haarcascade_files', HAARCASCADE_FILENAME)

# Etichetele emoțiilor (7 clase)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# Încărcarea Clasificatorului de Față (Haar Cascade)
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
if face_cascade.empty():
    print("Eroare: Nu s-a putut incarca clasificatorul de fata de la calea:", HAARCASCADE_PATH)
    exit()

# Încărcarea Modelului ML
try:
    model = load_model(MODEL_PATH, compile=False) 
    print("Modelul TensorFlow/Keras a fost încărcat cu succes.")
except Exception as e:
    print("Eroare fatală la încărcarea modelului: Nu s-a putut încărca de la calea:", MODEL_PATH)
    print("Detalii eroare:", e)
    exit()

# Pornirea camerei web
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Eroare: Nu s-a putut deschide camera web.")
    exit()

# =======================================================
# 2. CLASA APLICAȚIEI GUI (Tkinter)
# =======================================================

class EmotionApp:
    def __init__(self, master):
        self.master = master # master e fereastra fizica de pe ecran, self e partea din spate, memoria
        master.title("Emotion recognition on faces in real time")
        master.geometry("980x820")
        master.minsize(900, 720)

        BACKGROUND_COLOR = "#FFE6EA"
        TEXT_COLOR = "white"
        TITLE_FONT = ("Segoe Script", 26, "bold")   # caligrafic (fallback automat daca nu exista)
        MAIN_FONT = ("Helvetica", 18, "bold") 
        BUTTON_FONT = ("Helvetica", 14, "bold") 

        # Setează fundalul ferestrei principale
        master.config(bg=BACKGROUND_COLOR)

        # --- DEFINIREA CULORILOR DINAMICE ---
        self.TK_EMOTION_COLORS = { # pt Thinker - fereastra butoane text
            "Angry": "red",
            "Disgust": "green",
            "Fear": "purple",
            "Happy": "yellow", 
            "Sad": "blue",
            "Surprise": "orange",
            "Neutral": "gray",
            "Fără Față Detectată": "#AFAFAF"
        }
        self.CV_EMOTION_COLORS = {  # pt pixeli - patratul de pe fata
            "Angry": (255, 0, 0),      # Roșu (BGR)
            "Disgust": (0, 128, 0),    # Verde (BGR)
            "Fear": (128, 0, 128),     # Mov (BGR)
            "Happy": (255, 255, 0),    # Galben (BGR)
            "Sad": (0, 0, 255),        # Albastru (BGR)
            "Surprise": (255, 165, 0),  # Portocaliu (BGR)
            "Neutral": (128, 128, 128),  # Gri (BGR)
            "Fără Față Detectată": (160, 160, 160)  # Gri deschis
        }

        # Emoji specifice fiecarei emotii
        self.EMOJI = {
            "Angry": "😠",
            "Disgust": "🤢",
            "Fear": "😨",
            "Happy": "😄",
            "Sad": "😢",
            "Surprise": "😲",
            "Neutral": "😐",
            "Fără Față Detectată": "🫥"
        }
        # -----------------------------------
        
        # TITLUL APLICAȚIEI (încadrat + emoji)
        self.title_frame = tk.Frame(
            master,
            bg=BACKGROUND_COLOR,
            highlightthickness=4,
            highlightbackground="#FD3F92"
        )
        self.title_frame.pack(pady=(18, 10), padx=70, fill="x")

        self.title_label = tk.Label(
            self.title_frame,
            text="✨💖  Recunoașterea Emoțiilor în Timp Real  💖✨",
            font=TITLE_FONT,
            bg=BACKGROUND_COLOR,
            fg="#FD3F92"
        )
        self.title_label.pack(pady=10)

        # Variabilă pentru a stoca emoția curentă
        self.emotion_var = tk.StringVar()
        self.emotion_var.set("Așteaptă flux video...")

        # VIDEO ÎNCADRAT într-un chenar care își schimbă culoarea în funcție de emoție
        self.video_frame = tk.Frame(
            master,
            bg=BACKGROUND_COLOR,
            highlightthickness=8,
            highlightbackground=self.TK_EMOTION_COLORS["Fără Față Detectată"]
        )
        self.video_frame.pack(padx=70, pady=15)

        # Widget pentru afișarea fluxului video (în chenar)
        self.video_label = tk.Label(self.video_frame, bg=BACKGROUND_COLOR) 
        self.video_label.pack(padx=10, pady=10)

        # Widget pentru afișarea emoției prezise (cu emoji)
        self.emotion_label = tk.Label(
            master, 
            textvariable=self.emotion_var, 
            font=MAIN_FONT, 
            fg="#AFAFAF", # Culoare inițială
            bg=BACKGROUND_COLOR
        )
        self.emotion_label.pack(padx=10, pady=5)

        # Buton de Ieșire
        self.exit_button = tk.Button(
            master,
            text="💗  Închide Aplicația",
            command=self.on_closing,
            bg="#FD3F92",
            fg=TEXT_COLOR,
            font=BUTTON_FONT,
            padx=30,
            pady=14,
            relief="flat",
            activebackground="#ff6fa7",
            activeforeground="white"
        )
        self.exit_button.pack(pady=10)
            
        # Pornirea buclei de actualizare video
        self.delay = 10 
        self.update_frame()
        
    def on_closing(self):
        """ Funcția apelată la închiderea ferestrei """
        if messagebox.askokcancel("Ieșire", "Doriți să închideți aplicația?"):
            cap.release() 
            self.master.destroy() 

    # =======================================================
    # 3. LOGICA VIDEO ȘI AI (UPDATE_FRAME)
    # =======================================================
    def update_frame(self): #Funcția care citește un cadru, procesează emoția și actualizează GUI-ul
        ret, frame = cap.read() 
        
        if ret:
            # 1. PRE-PROCESARE INIȚIALĂ
            frame = cv2.flip(frame, 1) 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

            predicted_emotion = "Fără Față Detectată"
            
            # 2. DETECȚIA FEȚEI (OpenCV Haar Cascade)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                # 3. PREGĂTIREA IMAGINII PENTRU MODEL
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, INPUT_SIZE, interpolation=cv2.INTER_AREA)

                if roi_gray.size > 0:
                    # Normalizarea (0-255 -> 0.0-1.0)
                    roi = roi_gray.astype("float") / 255.0
                    roi = np.expand_dims(roi, axis=-1) 
                    roi = np.expand_dims(roi, axis=0) 

                    # 4. PREDICTȚIA EMOȚIEI
                    preds = model.predict(roi, verbose=0)[0]
                    emotion_index = preds.argmax()
                    predicted_emotion = EMOTIONS[emotion_index]
                    
                    # --- ACTUALIZARE VIZUALĂ DINAMICĂ (OpenCV) ---
                    box_color_bgr = self.CV_EMOTION_COLORS.get(predicted_emotion, (0, 0, 0))
                    
                    # 5. AFIȘARE VIZUALĂ (Desenează dreptunghi și text pe cadrul RGB)
                    cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), box_color_bgr, 2)

                    # Textul să rămână în cadru (sus dacă încape, altfel jos)
                    text_y = (y - 10) if (y - 10) > 25 else (y + h + 30)

                    cv2.putText(
                        rgb_frame, predicted_emotion, (x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color_bgr, 2
                    )
                    # -----------------------------------------------
                
                break 

            # --- ACTUALIZARE GUI DINAMICĂ (Tkinter) ---
            tkinter_color_name = self.TK_EMOTION_COLORS.get(predicted_emotion, "white")
            emoji = self.EMOJI.get(predicted_emotion, "🙂")
            
            # 6. ACTUALIZARE GUI (Tkinter)
            self.emotion_var.set(f"{emoji}  Emoție: {predicted_emotion}")
            
            # Actualizează culoarea etichetei Tkinter
            self.emotion_label.config(fg=tkinter_color_name)

            # Chenar video care "luminează" pe culoarea emoției
            self.video_frame.config(highlightbackground=tkinter_color_name)
            # -----------------------------------------------
            
            # Conversia OpenCV (NumPy array) la formatul Tkinter (PIL)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # 7. REPETAREA BUCLEI
        self.master.after(self.delay, self.update_frame)

# =======================================================
# 4. PUNCT DE INTRARE (MAIN)
# =======================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing) 
    root.mainloop()