import numpy as np
import cv2

cam_id = 0  # camera id

# for default webcam, cam_id is usually 0
# try out other numbers (1,2,..) if this does not work

cap = cv2.VideoCapture(cam_id)

mode = 'o' # show the original image at the beginning

sigma = 5

while True:
    ret, I = cap.read();
    #I = cv2.imread("agha-bozorg.jpg") # can use this for testing 
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) # convert to grayscale
    Ib = cv2.GaussianBlur(I, (sigma,sigma), 0); # blur the image
    
    if mode == 'o':
        # J = the original image
        J = I
    elif mode == 'x':
        # J = Sobel gradient in x direction
        J = np.abs(cv2.Sobel(Ib,cv2.CV_64F,1,0));
        
    elif mode == 'y':
        # J = Sobel gradient in y direction
        pass 

    
    elif mode == 'm':
        # J = magnitude of Sobel gradient
        pass
    
    elif mode == 's':
        # J = Sobel + thresholding edge detection
        pass
    
    elif mode == 'l':
        # J = Laplacian edges
        pass

    
    elif mode == 'c':
        # J = Canny edges
        pass
    
    # we set the image type to float and the
    # maximum value to 1 (for a better illustration)
    # notice that imshow in opencv does not automatically
    # map the min and max values to black and white. 
    J = J.astype(np.float) / J.max();    
    cv2.imshow("my stream", J);

    key = chr(cv2.waitKey(1) & 0xFF)

    if key in ['o', 'x', 'y', 'm', 's', 'c', 'l']:
        mode = key
    if key == '-' and sigma > 1:
        sigma -= 2
        print "sigma = %d"%sigma
    if key in ['+','=']:
        sigma += 2    
        print "sigma = %d"%sigma
    elif key == 'q':
        break

cap.release()
cv2.destroyAllWindows()







