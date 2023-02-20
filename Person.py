import cv2

# Chargement du classificateur de visage
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ouvrir la vidéo
cap = cv2.VideoCapture('ma_video.mp4')

# Initialisation du compteur
count = 0

# Boucle sur chaque frame de la vidéo
while cap.isOpened():
    # Lecture d'une frame
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion en niveau de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection de visages dans la frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Incrémenter le compteur du nombre de visages détectés
    count += len(faces)

    # Afficher la frame avec les visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()

print("Nombre de personnes : ", count)
