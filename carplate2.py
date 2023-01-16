import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

import pytesseract

base_path = r'C:\DevRoot\dataset\ComputerVision\carplate2'

# ↓ Google Colab 에서 
# from google.colab.patches import cv2_imshow  # 이미지를 보여줄 경우 필요
# from google.colab import files # 파일을 업로드 할때 필요

def cv2_imshow(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plt_imshow_bgr(bgr_img):
    plt.figure(figsize=(12,10))
    cvtImg = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plt.imshow(cvtImg)
    plt.show()

def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img

def find_chars(contour_list):
    matched_result_idx = []  # 최종적으로 남는 index 값들 담아서 리턴될 것이다.

    # contour_list 에서 2개의 countour (d1, d2) 의 모든 조합을 비교
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:  # d1 과 d2 가 같으면 넘어가구. (비교할 필요 없으니까)
                continue

            # 두 contour 사이의 거리를 구하기 위해 dx, dy 계산 (아래 그림 참조)
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

            # d1 의 대각선 길이
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            # 각도 구하기 (아래 그림 참조)
            if dx == 0:  # dx 가 0 이면 걍 90도
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx)) # arctan() 의 리턴값은 radian 값이다 
                                                            # 이를 호도법으로 변환 np.degrees()

            # 면적 비율, 폭 비율, 높이 비율
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            # 초기 설정값에 따른기분에 맞는 contour 만 골라서 
            # matched_contours_idx 에 contour 의 index 추가
            # 즉 d1 과 같은 번호판 후보군 d2(들)을 추가
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # 마지막에는 d1 도 넣어준다
        # append this contour
        matched_contours_idx.append(d1['idx'])

        # 그리하였는데, MIN_N_MATCHED 보다 개수가 적으면 번호판으로 인정안함
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        # 위 조건들을 다 통과하면 최종 후보군 matched_result_idx 에 인덱스 추가
        matched_result_idx.append(matched_contours_idx)

        # 그런데, 여기서 끝내지 않고, 
        # 최종 후보군에 들이 않은 것들을 한번 더 비교해볼거다

        # 일단 matched_contours_idx 에 들지 않은 것들을 unmatched_contour_idx 에 넣고
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        # np.take(a, idx)
        # a 에서 idx 와 같은 인덱스의 값만 추출

        # possible_contours ?? ▼▼ ?? 이상하다?  전역을 사용한다고?  contour_list
        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        # 재귀 호출 recursive
        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx  # 최종적으로 남는 index 값들이 담겨져 리턴된다.



import os
import sys
import numpy as np
import cv2
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# 파일 경로
base_path = r'.'

# 파일 불러오기
cap = cv2.VideoCapture(os.path.join(base_path, 'carplate2.mp4'))

# 너비 높이 등 파라미터 설정
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

if not cap.isOpened():
    print('camera open failed')
    sys.exit()
    
frame_cnt = 0

while True:
    ret, frame = cap.read()
    if not ret:  break
        
    height, width, channel = frame.shape
    
    bright = increase_brightness(frame, 80)
    
    # Grayscale
    gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
    
    # blur and threshold
    img_blurred = cv2.GaussianBlur(
        gray, # 원본 이미지
        ksize=(1, 1), 
        sigmaX=5, 
    )
    
    img_blur_thresh = cv2.adaptiveThreshold(
        img_blurred,   
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=29,
        C=9
    )


    contours, _ = cv2.findContours(
        img_blur_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    

    # contours 정보 전부 저장
    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 데이터를 만들고 insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),  # center X
            'cy': y + (h / 2)   # center Y
        })

    MIN_AREA = 50  # 최소 넓이 
    MAX_AREA = 500 # 
    MIN_WIDTH, MIN_HEIGHT = 2, 8  # 최소 너비, 높이
    MIN_RATIO, MAX_RATIO = 0.2, 1.0  # 너비-높이 비율의 최대/최소 

    possible_contours = []  # 위 조건에 맞는 것들을 걸러낸 것들을 담아보겠습니다

    cnt = 0
    # d = dict type
    for d in contours_dict:  # 위에서 저장했었던 contours_dict 를 순환하면서
        area = d['w'] * d['h']   # 넓이 계산
        ratio = d['w'] / d['h']  # 너비-높이 비율 계산

        # 조건에 맞는 것들만 골라서 possible_contours 에 담는다.
        if area > MIN_AREA and area < MAX_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    temp_result = np.zeros((height, width, 3), dtype=np.uint8)
    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        
        
    MAX_DIAG_MULTIPLYER = 7 # 
    MAX_ANGLE_DIFF = 4.0 #
    MAX_AREA_DIFF = 0.75 #
    MAX_WIDTH_DIFF = 0.65
    MAX_HEIGHT_DIFF = 0.2
    MIN_N_MATCHED = 5
    
    result_idx = find_chars(possible_contours)
    
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    PLATE_WIDTH_PADDING = 1.1 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5

    # 가로세로 비율
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []
    cnt = 0

    for i, matched_chars in enumerate(matched_result):
        # 일단 center X 기준으로 정렬 해줍니다 (직전까진 순서가 뒤죽박죽이었을테니)
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        # 번호판의 center 좌표를 계산해봅니다. (처음 contour ~ 마지막 countour 의 center 거리)
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        # 번호판의 width 계산
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        # 번호판의 height 계산
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        # 현재 번호판이 비뚤어져 있다. 
        # 회전 각도를 함 구해보자.  (아래 그림)
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']  # 높이
        triangle_hypotenus = np.linalg.norm(  # 빗변 구하기
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        # 높이와 빗변을 사용해서 arcsin() 함수로 각도 계산
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        

        # 위의 각도가 나오면 회전하는 transformation matrix 를 구하고
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        # 이를 이미지에 적용하고
        img_rotated = cv2.warpAffine(img_blur_thresh, M=rotation_matrix, dsize=(width, height))

        # 번호판 잘라내기
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )

        # 위 결과가 번호판의 가로세로 비율조건에 맞지 않으면 저장하지 않고 스킵함
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
        
        # 번호판 문자열 인식후 이를 담기 위한 변수들
        plate_chars = []  # <- 여기에 번호판의 문자들이 담길거다.
        longest_idx, longest_text = -1, 0  # <- 그 중에서 가장 긴 문자열을 번호판이라 할것이다.

        
        for i, plate_img in enumerate(plate_imgs):
            longest_text = 0
            # x1.6배 확대
            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            plate_img = cv2.GaussianBlur(plate_img, ksize=(3, 3), sigmaX=5) # 노이즈 한번 없애도

            plate_img = cv2.adaptiveThreshold(  # threshold 도 한번 더 해보고
                plate_img,
                maxValue=255.0,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                thresholdType=cv2.THRESH_BINARY, 
                blockSize=19, 
                C=9
            )

            # 또 한번더 contour를 찾는다
            contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

            # '번호판 부분' 의 좌표값을 일단 초기화
            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
            plate_max_x, plate_max_y = 0, 0

            for contour in contours:  # 각 contour 별로
                x, y, w, h = cv2.boundingRect(contour)  # boundingRect 구하고 

                area = w * h  # 면적과
                ratio = w / h  # 가로세로비율 구하고

                # 설정기준에 맞는지 체크해서
                if area > MIN_AREA \
                and w > MIN_WIDTH and h > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
                    # '번호판 부분' 좌표의 최대, 최소값 구하기
                    if x < plate_min_x: plate_min_x = x
                    if y < plate_min_y: plate_min_y = y
                    if x + w > plate_max_x: plate_max_x = x + w
                    if y + h > plate_max_y: plate_max_y = y + h

            # 위 에서 결정된 '번호판 부분' 좌표를 사용하여 '번호판 부분' 만 잘라내기 (crop)
            img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

            img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=5) # 노이즈 한번 없애도

            img_result = cv2.adaptiveThreshold(  # threshold 도 한번 더 해보고
                img_result,
                maxValue=255.0,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                thresholdType=cv2.THRESH_BINARY, 
                blockSize=19, 
                C=9
            )

            # 이미지에 패딩을 준다.
            img_result = cv2.copyMakeBorder(
                img_result, 
                top=10, bottom=10, left=10, right=10, 
                borderType=cv2.BORDER_CONSTANT, value=(0,0,0))   # 검정색 패딩(여백?)

            # 드디어 OCR 문자 인식!
            chars = pytesseract.image_to_string(
                img_result,
                lang='eng',
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            )

            result_chars = ''
            has_digit = False
            for c in chars:
                if ord('A') <= ord(c) <= ord('Z') or c.isdigit():
                    if c.isdigit():
                        has_digit = True
                    result_chars += c
                    
            
            
            plate_chars.append(result_chars)
            print("plate_chars:", plate_chars)

            

            # 가장 긴 문자열로 번호판 뽑음
            if len(result_chars) > longest_text:
#             if has_digit and len(result_chars) > longest_text:
                longest_text = len(result_chars)
                longest_idx = i


            plt.subplot(len(plate_imgs), 1, i+1)
            plt.imshow(img_result, cmap='gray')

            info = plate_infos[longest_idx]
            chars = plate_chars[longest_idx]
        

            # 사각형 그려주기
            cv2.rectangle(frame, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(0,0,255), thickness=2)

            label = chars
            cv2.putText(frame, label, (info['x'], info['y'] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            
    cv2.imshow('frame', frame)
    # cv2.imshow('temp_result', temp_result)

    if cv2.waitKey(delay) == 27:  # ESC 누르면 종료      
        break
        
cap.release()  
cv2.destroyAllWindows()    