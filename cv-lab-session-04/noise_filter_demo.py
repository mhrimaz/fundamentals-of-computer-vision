import numpy as np
import cv2

I = cv2.imread('isfahan.jpg').astype(np.float64) / 255;

noise_sigma = 0.04 # initial standard deviation of noise

m = 1; # initial filter size,
# with m = 1 the input image will not change

filter = 'b' # box filter

while True:
   
    if filter == 'b':
        # filter with a box filter
        F = np.ones((m,m), np.float64)/(m*m)
        
    elif filter == 'g':  
        # filter with a Gaussian filter
        pass
    
    # add noise to image
    J = I + N;
    

    # filtered image
    K = cv2.filter2D(J, -1, F);
    

    cv2.imshow('img', K)
    key = cv2.waitKey(30) & 0xFF
        

    if key == ord('b'):
        filter = 'b' # box filter
        print 'Box filter'
        
    elif key == ord('g'):  
        filter = 'g' # filter with a Gaussian filter
        print 'Gaussian filter'

    elif key == ord('+'):
        # increase m
        m = m + 2
        print 'm=',m

    elif key == ord('-'):
        # increase m
        if m >= 3:
            m = m - 2
        print 'm=', m

    elif key == ord('u'):
        # increase noise
        pass

    elif key == ord('d'):
        # decrease noise
        pass

    elif key == ord('q'): 
        break # quit

cv2.destroyAllWindows()
