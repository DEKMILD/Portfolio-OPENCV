import cv2
import numpy as np
import matplotlib.pyplot as plt

# ฟังก์ชันสำหรับประมวลผลภาพ (เปลี่ยนเป็นขอบภาพ)
def convertImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # แปลงภาพเป็นสีเทา
    blur = cv2.GaussianBlur(gray, (5, 5), 0)       # เบลอภาพเพื่อลด noise
    canny = cv2.Canny(blur, 100, 200)              # ใช้ Canny edge detection เพื่อหาขอบภาพ
    return canny

# โหลดภาพจากไฟล์
img = cv2.imread("MM/4.jpg")                       # อ่านภาพจากไฟล์
processed_img = convertImage(img)                 # เรียกฟังก์ชัน convertImage เพื่อหาขอบภาพ
original_img = img.copy()                         # เก็บสำเนาภาพต้นฉบับไว้สำหรับการตัดส่วน

contour_img = processed_img.copy()                # สร้างสำเนาภาพที่ประมวลผลสำหรับการหาขอบเขต (contour)

# หาขอบเขต (contours) ในภาพ
contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # เรียงขอบเขตตามพื้นที่ และเลือก 10 อันดับแรก

# วนลูปสำหรับการตรวจสอบ contour
for contour in contours:
    p = cv2.arcLength(contour, True)                       # คำนวณความยาวเส้นรอบ contour
    approx = cv2.approxPolyDP(contour, 0.02 * p, True)     # ลดจำนวนจุดของ contour ให้เหลือเฉพาะจุดสำคัญ

    if len(approx) == 4:                                   # หาก contour มี 4 ด้าน (เป็นรูปสี่เหลี่ยม)
        x, y, w, h = cv2.boundingRect(contour)             # คำนวณกรอบสี่เหลี่ยมล้อมรอบ contour
        license_img = original_img[y:y + h, x:x + w]       # ตัดภาพเฉพาะกรอบที่ล้อมรอบ contour
        cv2.imshow("License Detected : ", license_img)     # แสดงภาพที่ตัดได้ (พื้นที่กรอบสี่เหลี่ยม)
        cv2.drawContours(img, [contour], -1, (0, 255, 255), 3)  # วาด contour สีเหลืองบนภาพต้นฉบับ

# แสดงภาพผลลัพธ์
cv2.imshow("Image", img)                                  # แสดงภาพต้นฉบับพร้อม contour ที่ตรวจจับได้
cv2.waitKey(0)                                            # รอให้ผู้ใช้กดปุ่มเพื่อปิดหน้าต่าง
