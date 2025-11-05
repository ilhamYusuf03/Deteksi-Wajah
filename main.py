import cv2
import os
import sys

# Pastikan path file XML diambil dari lokasi file ini sendiri
base_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(base_path, "face_ref.xml")

# Load classifier
face_ref = cv2.CascadeClassifier(xml_path)

# Cek kalau XML tidak ditemukan
if face_ref.empty():
    print(f"[ERROR] File XML tidak ditemukan di: {xml_path}")
    sys.exit()

# Aktifkan kamera
camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1)
    return faces

def drawer_box(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("UcupFace", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()

if __name__ == "__main__":
    main()
