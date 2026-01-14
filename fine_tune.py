import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys

# =======================================================
# CONFIGURAȚII DE MEDIU ȘI CĂI
# =======================================================

# Dezactivează optimizările oneDNN (Intel) pentru stabilitate sporită în antrenare.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# Obține directorul scriptului, asigurând căile absolute pentru resurse.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Cazul în care scriptul este rulat într-un mediu fără __file__ (e.g., Jupyter)
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# --- 1. CONFIGURAȚIE ---
NEW_DATA_DIR = os.path.join(script_dir, 'new_training_data') 
MODEL_FILENAME = '_mini_XCEPTION.102-0.66.hdf5'
SAVE_FILENAME = 'fine_tuned_model.hdf5'

MODEL_PATH = os.path.join(script_dir, 'models', MODEL_FILENAME) 
SAVE_PATH = os.path.join(script_dir, 'models', SAVE_FILENAME)      

INPUT_SIZE = (64, 64)
BATCH_SIZE = 16 
CUSTOM_LEARNING_RATE = 0.00001 
EPOCHS = 15 # Numărul de epoci de antrenare

# =======================================================
# 2. ÎNCĂRCAREA ȘI PREGĂTIREA DATELOR
# =======================================================

# Generator pentru setul de antrenare
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(NEW_DATA_DIR, 'train'),
    target_size=INPUT_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Generator pentru setul de validare (doar rescalare, fără augmentare)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(NEW_DATA_DIR, 'validation'),
    target_size=INPUT_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# =======================================================
# 3. PREGĂTIREA MODELULUI (FINE-TUNING)
# =======================================================

print(f"\nSe încarcă modelul existent de la: {MODEL_PATH}")
try:
    # Încărcăm modelul fără starea optimizatorului vechi
    model = load_model(MODEL_PATH, compile=False) 
except Exception as e:
    print(f"Eroare fatală la încărcarea modelului: {e}")
    sys.exit(1)

# Îngheață majoritatea straturilor pentru a menține cunoștințele de bază.
# Lăsăm ultimele 4 straturi (clasificatoarele) antrenabile.
for layer in model.layers[:-4]:
    layer.trainable = False
    
print(f"Număr de straturi înghețate: {len(model.layers) - 4} / {len(model.layers)}")

# Configurează optimizatorul cu rata de învățare foarte mică
optimizer = tf.keras.optimizers.Adam(learning_rate=CUSTOM_LEARNING_RATE)

# Recompilează modelul cu noul optimizator și straturile selectate ca antrenabile.
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary() 

# =======================================================
# 4. ANTRENAMENTUL PROPRIU-ZIS ȘI SALVAREA
# =======================================================

print("\n🚀 Start Fine-Tuning...")
try:
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Salvează modelul nou
    model.save(SAVE_PATH)
    print(f"\n✅ Antrenamentul s-a terminat. Modelul nou salvat la: {SAVE_PATH}")

except Exception as e:
    print(f"\n❌ A apărut o eroare în timpul antrenamentului: {e}")
    print("Sugestie: Asigură-te că ai suficientă memorie RAM/GPU sau încearcă să micșorezi BATCH_SIZE.")