import cv2
import mediapipe as mp
import numpy as np

# Görselleştirme için sabitler
COLOR = (255, 0, 0)  # Mavi renk (BGR formatında)

# MediaPipe Yüz Algılama modülünü başlat
mp_face_detection = mp.solutions.face_detection

# Çerçeve işleme fonksiyonu: Yüzleri algılar ve işlenmiş çerçeveyi ve koordinatları döner
def process_frame(image, draw_box=True):
    """
    Bu fonksiyon, bir görüntüde yüz algılaması yapar ve yüzlerin etrafına sınırlayıcı kutular çizer.
    
    Parametreler:
        - image: İşlenecek görüntü (BGR formatında bir OpenCV çerçevesi).
        - draw_box (bool, opsiyonel): Yüzlerin etrafına sınırlayıcı kutular çizilip çizilmeyeceğini belirler.
          Varsayılan olarak True (kutular çizilir).
    
    Dönüş Değerleri:
        - annotated_image: Yüzlerin etrafına kutular çizilmiş (veya çizilmemiş) işlenmiş görüntü.
        - coordinates: Algılanan her yüzün (origin_x, origin_y, bbox_width, bbox_height) formatında
          koordinatlarını içeren bir liste.
    """
    
    # MediaPipe Yüz Algılama nesnesini başlat
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Çerçevenin kopyasını oluştur (orijinal çerçeveyi değiştirmemek için)
    annotated_image = image.copy()
    height, width, _ = image.shape  # Çerçevenin yüksekliğini, genişliğini ve kanal sayısını al
    coordinates = []  # Yüz koordinatlarını saklamak için liste

    # Yüz algılama işlemini gerçekleştir (RGB formatında)
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Yüzler algılandıysa kontrol et
    if results.detections:
        for detection in results.detections:
            # sınırlayıcı kutunun (bounding box) bilgilerini al
            bboxC = detection.location_data.relative_bounding_box
            origin_x = int(bboxC.xmin * width)  # Sol üst köşe x koordinatı
            origin_y = int(bboxC.ymin * height)  # Sol üst köşe y koordinatı
            bbox_width = int(bboxC.width * width)  # Kutunun genişliği
            bbox_height = int(bboxC.height * height)  # Kutunun yüksekliği

            # Koordinatları sınırların içinde kalacak şekilde ayarla
            origin_x = max(0, min(origin_x, width - 1))
            origin_y = max(0, min(origin_y, height - 1))
            bbox_width = max(0, min(bbox_width, width - origin_x))
            bbox_height = max(0, min(bbox_height, height - origin_y))

            # Koordinatları listeye ekle
            coordinates.append((origin_x, origin_y, bbox_width, bbox_height))

            # Eğer draw_box True ise sınırlayıcı kutuyu çiz
            if draw_box:
                start_point = (origin_x, origin_y)  # sınırlayıcı kutunun başlangıç noktası
                end_point = (origin_x + bbox_width, origin_y + bbox_height)  # sınırlayıcı kutunun bitiş noktası
                cv2.rectangle(annotated_image, start_point, end_point, COLOR, 3)  # Mavi kutu çiz

    # İşlenmiş çerçeveyi ve koordinatları döndür
    return annotated_image, coordinates

def filtre_uygulama(frame, coordinates, filter_type):
    filtered_frame = frame.copy()
    for (origin_x, origin_y, bbox_width, bbox_height) in coordinates:
        face_region = frame[origin_y:origin_y + bbox_height, origin_x:origin_x + bbox_width]

        if filter_type == "Gauss Filtresi":
            face_region = cv2.GaussianBlur(face_region, (15, 15), 0)
        elif filter_type == "Median Filtresi":
            face_region = cv2.medianBlur(face_region, 15)
        elif filter_type == "Ortalama Filtresi":
            face_region = cv2.blur(face_region, (15, 15))
        elif filter_type == "Sobel Filtresi":
            sobelx = cv2.Sobel(face_region, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(face_region, cv2.CV_64F, 0, 1, ksize=5)
            face_region = cv2.convertScaleAbs(sobelx + sobely)
        elif filter_type == "Prewitt Filtresi":
            prewittx = cv2.filter2D(face_region, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
            prewitty = cv2.filter2D(face_region, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
            face_region = cv2.convertScaleAbs(prewittx + prewitty)
        elif filter_type == "Laplacian Filtresi":
            laplacian = cv2.Laplacian(face_region, cv2.CV_64F)
            face_region = cv2.convertScaleAbs(laplacian)
        elif filter_type == "Bulanıklaştırma":
            face_region = cv2.GaussianBlur(face_region, (101,101),0)
            face_region = cv2.GaussianBlur(face_region, (101, 101), 0)

        filtered_frame[origin_y:origin_y + bbox_height, origin_x:origin_x + bbox_width] = face_region

    return filtered_frame

cap = cv2.VideoCapture(0)  # Webcami başlat
filter_mode = None
while cap.isOpened():  # Webcam açık olduğu sürece
    ret, frame = cap.read()  # Bir çerçeve oku
    if not ret:  # Eğer çerçeve okunamadıysa
        print("Hata: Çerçeve yakalanamadı.")
        break

    # process_frame fonksiyonunu çağır, draw_box parametresiyle
    annotated_frame, face_coordinates = process_frame(frame, draw_box=True)

    # Yüz koordinatlarını kullanabilirsiniz
    print(f"Algılanan yüz koordinatları: {face_coordinates}")
    if filter_mode:
        filtered_frame = filtre_uygulama(frame, face_coordinates, filter_mode)
    else:
        filtered_frame = frame

    cv2.imshow("Orjinal",annotated_frame)
    cv2.imshow("Filtre", filtered_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("1"):
        filter_mode = "Ortalama Filtresi" if filter_mode != "Ortalama Filtresi" else None
    if key == ord("2"):
        filter_mode = "Median Filtresi" if filter_mode != "Median Filtresi" else None
    if key == ord("3"):
        filter_mode = "Gauss Filtresi" if filter_mode != "Gauss Filtresi" else None
    if key == ord("4"):
        filter_mode = "Sobel Filtresi" if filter_mode != "Sobel Filtresi" else None
    if key == ord("5"):
        filter_mode = "Prewitt Filtresi" if filter_mode != "Prewitt Filtresi" else None
    if key == ord("6"):
        filter_mode = "Laplacian Filtresi" if filter_mode != "Laplacian Filtresi" else None
    if key == ord("7"):
        filter_mode = "Bulanıklaştırma" if filter_mode != "Bulanıklaştırma" else None
    # 'q' tuşuna basıldığında döngüyü kır (çık)
    if key == ord("q"):
        break

# Webcami serbest bırak ve tüm OpenCV pencerelerini kapat
cap.release()
cv2.destroyAllWindows()
