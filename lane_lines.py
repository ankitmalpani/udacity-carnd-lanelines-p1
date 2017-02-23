import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from math import floor
from scipy import stats
import imageio
from moviepy.editor import VideoFileClip
from IPython.display import HTML

class LaneLines(object):
    def __init__(self):
        #place holder. Needs improvements, validations

    def test_video(self, video_file):
        white_output = 'white.mp4'
        clip1 = VideoFileClip('viedos/{vf}'.format(vf=video_file))
        white_clip = clip1.fl_image(self.process_image) #NOTE: this function expects color images!!
        %time white_clip.write_videofile(white_output, audio=False)

    def process_image(self, img_loc):
        image = mpimg.imread('test_images/{loc}'.format(loc=img_loc))
        print('This image is:', type(image), 'with dimesions:', image.shape)
        plt.imshow(image)
        image_copy = np.copy(image)
        gray_image = self.grayscale(image_copy)
        gray_copy = np.copy(gray_image)
        plt.imshow(gray_copy, cmap='gray')
        gaussian_smoothed_gray = self.gaussian_blur(gray_copy, 5)
        plt.imshow(gaussian_smoothed_gray, cmap='gray')
        smoothed_edge_image = self.canny(gaussian_smoothed_gray, 100, 200 )
        plt.imshow(smoothed_edge_image)
        vertices = np.array([[(100,539),(430, 320), (520, 320), (900,539)]], dtype=np.int32)
        roi_image = self.region_of_interest(smoothed_edge_image, vertices)
        plt.imshow(roi_image)
        detected_lines_image_old = self.hough_lines(roi_image, 1, (np.pi*1)/180, 35, 5, 2, use_old=True)
        plt.imshow(detected_lines_image_old)
        pre_full_line_image = self.weighted_img(detected_lines_image_old, image_copy)
        plt.imshow(pre_full_line_image)

    def slope_and_y_intercept(self, x, y):
        """Helper method to find slope and y-intercept for given x,y pairs.
        takes (x1,x2) and (y1,y2) as input"""
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        return slope, intercept

    def is_under_threshold(self, x, opt_value):
        """Helper function to identify if a value can
        be considered as optimalon the image if its closer than 2 pixels."""
        return abs(opt_value - x) <= 2

    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def draw_lines_old(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        #shape that is handy in computing points for extrapolated lines
        imshape = self.grayscale(img).shape

        #starting parameters to keep track of minimum x,y values on left and right.
        min_x_left = min_x_right = imshape[1]
        max_x_left = max_x_right = 0
        min_y_left = min_y_right = imshape[0]

        #also keep track of x optima and y maxima
        min_x_for_max_y = max_x_for_max_y = 0

        #collect slope values for left and right lines for finding the average slope
        left_slopes = []
        right_slopes = []

        #for each line compute min,max values to identify vertices for a polygon (i.e. points for extrapolated lines)
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope, y_inter = self.slope_and_y_intercept([x1,x2],[y1,y2])
                if slope > 0 and slope < 1:
                    right_slopes.append(slope)
                    min_x_right = min(min_x_right, x1, x2)
                    min_y_right = min(min_y_right, y1, y2)
                    max_x_right = max(max_x_right, x1, x2)
                    if self.is_under_threshold(y1, imshape[0]):
                        max_x_for_max_y = max(max_x_for_max_y, x1) #store corresonding x for max y
                    elif self.is_under_threshold(y2, imshape[0]):
                        max_x_for_max_y = max(max_x_for_max_y, x2)
                elif slope < 0 and slope > -1:
                    left_slopes.append(slope)
                    min_x_left = min(min_x_left, x1, x2)
                    min_y_left = min(min_y_left, y1, y2)
                    max_x_left = max(max_x_left, x1, x2)
                    if self.is_under_threshold(y1, imshape[0]):
                        min_x_for_max_y = x1
                    elif self.is_under_threshold(y2, imshape[0]):
                        min_x_for_max_y = x2

        #find average slope
        avg_slope_left = avg_slope_right = 0
        if len(left_slopes) > 0:
            avg_slope_left = sum(left_slopes)/len(left_slopes)
        if len(right_slopes) > 0:
            avg_slope_right = sum(right_slopes)/len(right_slopes)


        #find upper left/right endpoints (mostly easily defined by hough lines themselves)
        poly_upper_left = [int(floor(max_x_left)), int(floor(min_y_left))]
        poly_upper_right = [int(floor(min_x_right)), int(floor(min_y_right))]

        #extrapolate lower left/right points depending on what points are available from hough transform
        if min_x_for_max_y and max_x_for_max_y:
            poly_lower_left = [int(floor(min_x_for_max_y)), int(floor(imshape[0]))]
            poly_lower_right = [int(floor(max_x_for_max_y)), int(floor(imshape[0]))]
        elif min_x_for_max_y:
            poly_lower_left = [int(floor(min_x_for_max_y)), int(floor(imshape[0]))]
            right_x = -(((poly_upper_right[1] - imshape[0])/avg_slope_right) - poly_upper_right[0])
            poly_lower_right = [int(floor(right_x)), int(floor(imshape[0]))]
        elif max_x_for_max_y:
            poly_lower_right = [int(floor(max_x_for_max_y)), int(floor(imshape[0]))]
            #left_x = (imshape[0] - y_inter_left)/avg_slope_left
            left_x = -(((poly_upper_left[1] - imshape[0])/avg_slope_left) - poly_upper_left[0])
            poly_lower_left = [int(floor(left_x)), int(floor(imshape[0]))]
        else:
            right_x = (imshape[0] - y_inter_right)/avg_slope_right
            poly_lower_right = [int(floor(right_x)), int(floor(imshape[0]))]
            left_x = (imshape[0] - y_inter_left)/avg_slope_left
            poly_lower_left = [int(floor(left_x)), int(floor(imshape[0]))]

        #draw the lines on the image
        cv2.line(img, tuple(poly_lower_left), tuple(poly_upper_left), color=[255, 0, 0], thickness=3)
        cv2.line(img, tuple(poly_lower_right), tuple(poly_upper_right), color=[255, 0, 0], thickness=3)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap, use_old=False):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if use_old:
            self.draw_lines_old(line_img, lines)
        else:
            self.draw_lines(line_img, lines)
        return line_img

    # Python 3 has support for cool math symbols.

    def weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)
