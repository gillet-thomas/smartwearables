
import os
import math
import cv2
import imutils
import glob
import random

# package_directory = os.path.dirname(os.path.abspath(__file__))
package_directory = os.getcwd()
garments_directory = "data/garments/"

def setUpperBody(garment=None, index=None):
    global file1, file2, file3, file4, file5

    if index is None:
        if garment is None:
            files = glob.glob(garments_directory + '*_body_*.png')
        elif garment is not None:
            files = glob.glob(garments_directory + garment + '_body_*.png')
        index = random.randrange(0, len(files))
        file = files[index]
    else:
        file = glob.glob(garments_directory + garment + '_body_' + index + '.png')[0]

    parts = file.split("\\")[1].split('_')        # data/garments\tshirt_body_2.png
    if parts[0] == 'tshirt':
        file1 = 'tshirt_body_' + parts[2]
        file2 = 'tshirt_lsleeve_' + parts[2]
        file3 = 'tshirt_rsleeve_' + parts[2]
        file4 = file5 = None
    elif parts[0] == 'pullover':
        file1 = 'pullover_body_' + parts[2]
        file2 = 'pullover_lsleeve_' + parts[2]
        file3 = 'pullover_rsleeve_' + parts[2]
        file4 = 'pullover_lforearm_' + parts[2]
        file5 = 'pullover_rforearm_' + parts[2]
    elif parts[0] == 'dress':
        file1 = 'dress_body_' + parts[2]
        file2 = file3 = file4 = file5 = None

    print(file1, file2, file3, file4, file5)

def setLowerBody(garment=None, index=None):
    global file6

    if index is None:
        if garment is None:
            files = glob.glob(garments_directory + '*_full_*.png')
        elif garment is not None:
            files = glob.glob(garments_directory + garment + '_full_*.png')
        index = random.randrange(0, len(files))
        file = files[index]
    else:
        file = garments_directory + garment + '_full_' + index + '.png'
   
    file6 = file.split("\\")[1]
    print(file6)

def Distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # print(y1, y2, x1, x2, y1o, y2o, x1o, x2o)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (
            alpha * img_overlay[y1o:y2o, x1o:x2o, c] + alpha_inv * img[y1:y2, x1:x2, c])

def getPoints(joints):
    rshX = joints[33][0]
    rshY = joints[33][1]

    lshX = joints[34][0]
    lshY = joints[34][1]

    relbX = joints[32][0]
    relbY = joints[32][1]

    lelbX = joints[35][0]
    lelbY = joints[35][1]

    rhipX = joints[27][0]
    rhipY = joints[27][1]

    lhipX = joints[28][0]
    lhipY = joints[28][1]

    rkneeX = joints[26][0]
    rkneeY = joints[26][1]

    lkneeX = joints[29][0]
    lkneeY = joints[29][1]

    rfootX = joints[25][0]
    rfootY = joints[25][1]

    lfootX = joints[30][0]
    lfootY = joints[30][1]

    neckX = joints[37][0]
    neckY = joints[37][1]

    rwristX = joints[31][0]
    rwristY = joints[31][1]

    lwristX = joints[36][0]
    lwristY = joints[36][1]

    return rshX, rshY, lshX, lshY, relbX, relbY, lelbX, lelbY, rhipX, rhipY, lhipX, lhipY, rkneeX, rkneeY, lkneeX, lkneeY, rfootX, rfootY, lfootX, lfootY, neckX, neckY, rwristX, rwristY, lwristX, lwristY

def getRotatedPoints(img, rotationCenterX, rotationCenterY, offsetX = 0, offsetY = 0):
    (originX, originY) = (0,0)
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    rotate_cx = originX + cX                             # Center x of the rotate_bound image
    rotate_cy = originY + cY                             # Center y of the rotate_bound image

    originX = originX + (rotationCenterX-rotate_cx) + offsetX         # New top left X coordinates offsetting rotation
    originY = originY + (rotationCenterY-rotate_cy) + offsetY         # New top left Y coordinates offsetting rotation

    return originX, originY

def isPointInFrame(frame, x, y):
    return x > 0 and x < frame.shape[1] and y > 0 and y < frame.shape[0]

def getGarmentRatio(img):
    top_index = int(img.shape[1]/10)
    top = img[top_index]
    first = last = None
    for i in range(top.shape[0]):
        if(top[i][3]>0):
            if(first is None):
                first = i
            if(first is not None):
                last = i

    ratio = top.shape[0]/(last-first)
    return ratio
            
def dress(frame, joints, generateNewClothes):
    
    ######################### INIT #########################
    if generateNewClothes: setUpperBody(), setLowerBody()
    global cloth_body, cloth_lsleeve, cloth_rsleeve, cloth_lforearm, cloth_rforearm, is_dress
    global body_width, body_x, body_y, lsleeve_x, lsleeve_y, rsleeve_X, rsleeve_y, lforearm_x, lforearm_y, rforearm_x, rforearm_y
    global adjusted_rwidth, adjusted_lwidth
    
    rshX, rshY, lshX, lshY, relbX, relbY, lelbX, lelbY, rhipX, rhipY, lhipX, lhipY, rkneeX, rkneeY, lkneeX, lkneeY, rfootX, rfootY, lfootX, lfootY, neckX, neckY, rwristX, rwristY, lwristX, lwristY = getPoints(joints)
    
    umidX = (rshX+lshX)/2
    umidY = (rshY+lshY)/2
    lmidX = (rhipX+lhipX)/2
    lmidY = (rhipY+lhipY)/2    

    is_dress = False   
    body_width = 0

    ######################### BODY #########################
    if file1 and isPointInFrame(frame, rshX, rshY) and isPointInFrame(frame, rhipX, rhipY) and isPointInFrame(frame, lshX, lshY) and isPointInFrame(frame, lhipX, lhipY):
        file = os.path.join(package_directory, garments_directory, file1)
        s_img = cv2.imread(file, -1)
        is_dress = file1.startswith("dress")
        
        body_width = Distance(rshX, rshY, lshX, lshY)
        body_height = Distance(neckX, neckY, lmidX, lmidY)

        scaling = body_scaling if not is_dress else getGarmentRatio(s_img)
        
        adjusted_width = int(body_width*scaling)
        adjusted_height = int(body_height*scaling)

       ### RESIZE + ROTATE + ADJUST ORIGIN ###
        if(adjusted_height > 0 and adjusted_width > 0):
            s_img = cv2.resize(s_img, (int(adjusted_width), int(adjusted_height)))

        angle = math.degrees(math.atan2(umidY-lmidY, umidX-lmidX))+90
        
        rotationY = neckY + s_img.shape[0]/2 - abs(angle)/2
        rotationPointBelowHip = (rotationY >= lmidY*1.1) # Multiply by 1.1 to exclude clothes with rotation point near the lmidY point
        rotationX = lmidX + (lmidX - umidX)*(1-(1/scaling)) if rotationPointBelowHip else lmidX - (lmidX - umidX)*(1-(1/scaling))

        s_img = imutils.rotate_bound(s_img, angle)

        body_x, body_y = getRotatedPoints(s_img, rotationX, rotationY)

        cloth_body = s_img
    else: 
        cloth_body = None

    ######################### RIGHT SLEEVE ###########################
    if file2 and isPointInFrame(frame, rshX, rshY) and isPointInFrame(frame, relbX, relbY):
        file = os.path.join(package_directory, garments_directory, file2)
        s_img = cv2.imread(file, -1)  

        rsleeve_height = Distance(rshX, rshY, relbX, relbY)

        adjusted_height = int(rsleeve_height*body_scaling)
        adjusted_rwidth = int(adjusted_height/3)           # Set the width to the 3rd of the length
                
        ### RESIZE + ROTATE + ADJUST ORIGIN ###
        if(adjusted_height > 0 and adjusted_rwidth > 0):
            s_img = cv2.resize(s_img, (adjusted_rwidth, adjusted_height))

        angle = math.degrees(math.atan2(rshY-relbY, rshX-relbX))+90
        s_img = imutils.rotate_bound(s_img, angle)

        rotationCenterX = (rshX+relbX)/2
        rotationCenterY = (rshY+relbY)/2
        rsleeve_X, rsleeve_y = getRotatedPoints(s_img, rotationCenterX, rotationCenterY)
        
        cloth_rsleeve = s_img
    else: 
        cloth_rsleeve = None
        
    ########################## LEFT SLEEVE ####################
    if file3 and isPointInFrame(frame, lshX, lshY) and isPointInFrame(frame, lelbX, lelbY):
        file = os.path.join(package_directory, garments_directory, file3)
        s_img = cv2.imread(file, -1)  

        lsleeve_height = Distance(lshX, lshY, lelbX, lelbY)

        adjusted_height = int(lsleeve_height*body_scaling)
        adjusted_lwidth = int(adjusted_height/3) 

        ### RESIZE + ROTATE + ADJUST ORIGIN ###
        if(adjusted_height > 0 and adjusted_lwidth > 0):
            s_img = cv2.resize(s_img, (adjusted_lwidth, adjusted_height))
        
        angle = math.degrees(math.atan2(lelbY-lshY, lelbX-lshX))-90
        s_img = imutils.rotate_bound(s_img, angle)

        rotationCenterX = (lshX+lelbX)/2
        rotationCenterY = (lshY+lelbY)/2
        lsleeve_x, lsleeve_y = getRotatedPoints(s_img, rotationCenterX, rotationCenterY)

        cloth_lsleeve = s_img
    else: 
        cloth_lsleeve = None

     ########################## RIGHT FOREARM ####################
    if file4 and isPointInFrame(frame, rwristX, rwristY) and isPointInFrame(frame, relbX, relbY):
        file = os.path.join(package_directory, garments_directory, file4)
        s_img = cv2.imread(file, -1)  

        rforearm_height = Distance(rwristX, rwristY, relbX, relbY)

        adjusted_height = int(rforearm_height*body_scaling)

        ### RESIZE + ROTATE + ADJUST ORIGIN ###
        if(adjusted_height > 0 and adjusted_rwidth > 0):
            s_img = cv2.resize(s_img, (adjusted_rwidth, adjusted_height))
        
        angle = math.degrees(math.atan2(relbY-rwristY, relbX-rwristX))+90
        s_img = imutils.rotate_bound(s_img, angle)

        rotationCenterX = (rwristX+relbX)/2
        rotationCenterY = (rwristY+relbY)/2
        rforearm_x, rforearm_y = getRotatedPoints(s_img, rotationCenterX, rotationCenterY)

        cloth_rforearm = s_img
    else:
        cloth_rforearm = None
    
    ########################## LEFT FOREARM ####################
    if file5 and isPointInFrame(frame, lwristX, lwristY) and isPointInFrame(frame, lelbX, lelbY):
        file = os.path.join(package_directory, garments_directory, file5)
        s_img = cv2.imread(file, -1)  

        lforearm_height = Distance(lwristX, lwristY, lelbX, lelbY)

        adjusted_height = int(lforearm_height*body_scaling)

        ### RESIZE + ROTATE + ADJUST ORIGIN ###
        if(adjusted_height > 0 and adjusted_lwidth > 0):
            s_img = cv2.resize(s_img, (adjusted_lwidth, adjusted_height))
        
        angle = math.degrees(math.atan2(lelbY-lwristY, lelbX-lwristX))+90
        s_img = imutils.rotate_bound(s_img, angle)

        rotationCenterX = (lwristX+lelbX)/2
        rotationCenterY = (lwristY+lelbY)/2
        lforearm_x, lforearm_y = getRotatedPoints(s_img, rotationCenterX, rotationCenterY)

        cloth_lforearm = s_img
    else:
        cloth_lforearm = None

    ########################## PANTS ####################
    if file6 and not is_dress and isPointInFrame(frame, rhipX, rhipY) and isPointInFrame(frame, lhipX, lhipY):
        file = os.path.join(package_directory, garments_directory, file6)
        s_img = cv2.imread(file, -1)
        is_short = file6.startswith("shorts") or file6.startswith("skirt")

        scaling = getGarmentRatio(s_img)
        
        adjusted_width = int(body_width*scaling)
        adjusted_height = abs(rfootY-rhipY) if not is_short else abs(rkneeY-rhipY)

        ### RESIZE + ROTATE + ADJUST ORIGIN ###
        if(adjusted_height > 0 and adjusted_width > 0):
            s_img = cv2.resize(s_img, (int(adjusted_width), int(adjusted_height)))

        pants_x = lmidX - adjusted_width/2
        pants_y = lmidY

        cloth_pants = s_img
    else: 
        cloth_pants = None

############################################################

    if file6 and type(cloth_pants) != type(None):
        overlay_image_alpha(frame, cloth_pants[:, :, 0:3], (int(pants_x), int(pants_y)), cloth_pants[:, :, 3] / 255.0)
    
    if file5 and type(cloth_rforearm) != type(None):
        overlay_image_alpha(frame, cloth_rforearm[:, :, 0:3], (int(rforearm_x), int(rforearm_y)), cloth_rforearm[:, :, 3] / 255.0)

    if file4 and type(cloth_lforearm) != type(None):
        overlay_image_alpha(frame, cloth_lforearm[:, :, 0:3], (int(lforearm_x), int(lforearm_y)), cloth_lforearm[:, :, 3] / 255.0)
    
    if file3 and type(cloth_rsleeve) != type(None):
        overlay_image_alpha(frame, cloth_rsleeve[:, :, 0:3], (int(rsleeve_X), int(rsleeve_y)), cloth_rsleeve[:, :, 3] / 255.0)

    if file2 and type(cloth_lsleeve) != type(None):
        overlay_image_alpha(frame, cloth_lsleeve[:, :, 0:3], (int(lsleeve_x), int(lsleeve_y)), cloth_lsleeve[:, :, 3] / 255.0)

    if file1 and type(cloth_body) != type(None):
        overlay_image_alpha(frame, cloth_body[:, :, 0:3], (int(body_x), int(body_y)), cloth_body[:, :, 3] / 255.0)
    
    # frame = cv2.circle(frame, (int(neckX), int(neckY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(rshX), int(rshY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(lshX), int(lshY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(umidX), int(umidY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(rhipX), int(rhipY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(lhipX), int(lhipY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(lmidX), int(lmidY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(relbX), int(relbY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(lelbX), int(lelbY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(rwristX), int(rwristY)), radius=0, color=(0, 0, 255), thickness=5)
    # frame = cv2.circle(frame, (int(lwristX), int(lwristY)), radius=0, color=(0, 0, 255), thickness=5)

#### SET SETTINGS FROM PARAMETERS ####
body_scaling = 1.2

file1 = None
file2 = None
file3 = None
file4 = None
file5 = None   
file6 = None
setUpperBody()
setLowerBody()