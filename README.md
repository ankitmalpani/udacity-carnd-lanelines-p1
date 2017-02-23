# udacity-carnd-lanelines-p1
Submission of Udacity CarND's first project: LaneLines

#**Finding Lane Lines on the Road**

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

[//]: # (Image References)

[image1]: ./sample_images/gray.png "gray"
[image2]: ./sample_images/smoothed_gray.png "smoothed_gray"
[image3]: ./sample_images/canny_edge.png "canny_edge"
[image4]: ./sample_images/roi_hough_old.png "roi_hough_old"

*My pipeline consisted of the following steps:*

1. First converted the image to grayscale

![alt text][image1]

2. Took a copy of the image to work with
3. Applied gaussian smoothing using kernel of size 5

![alt text][image2]

4. Applied canny_edge transform to identify edges

![alt text][image3]

5. Defined vertices that would help define the ROI
6. Applied region_of_interest filter using the vertices defined in step 4.
7. Applied the hough_transform to this filtered image

![alt text][image4]

8. Performed the weighted add to get the final results

By this time I had the first result where the lanes were detected without extrapolation.
(To reproduce this, one could use `draw_lines_old` method instead of the new `draw_lines` method)

*Approach to get a single line:*
After trying out a few different combinations of ROI filters and hough transforms, I noticed how we had it would be
very difficult to get the full lines with these parameters.
I approached this issue by trying to get 4 endpoints for the right and the left lines.
Since I was dealing with lines, I knew slope and Y-intercept would be important elements here.
I also wanted to calculate the min, max of x,y co-ordinates from the line segments from the hough output. This would help in
estimating the top 2 points for right and left lines.
Here is the summary of steps I took: (Though these are linearly written, there was a fair amount of iteration on these steps)

1. Wrapped scipy's `stats.lingress` function inside a helper function to get the slope and Y-intercept for a pair of points
2. Created a helper method to identify if a point was close enough to the max value of x,y axis. For eg: the max-Y value for the image could be 540, but I wanted to consider points that lie on 538/539 pixels as well
3. Realized that since Y is increasing as it goes down, slopes would be inverted
4. Modified the `draw_lines` function by looping through all the line segments returned by hough transform and collecting positive and negative slopes for right and left lanes respectively in a list. (This would help to calculate an average slope for all line segments later on)
5. Tracked min value of x-coordinate and y-coordinate for each left and right line (negative and positive slopes)
6. Also tracked the x-coordinate-optima for values where y-coordinate was close to 540. This would give me a ready set of endpoints at least for one of the lines (in general the solid lane line - assuming hough transform outputs correctly)
7. Once the looping was done, I calculated the average left, right slopes
8. Then, I estimated the 4 end points required to draw the line (4 vertices of a polygon). The upper points were easier to estimate based on the min/max values computed when looping. The lower endpoints required computation based on what information was already available to me in my variables mentioned in (6). The calculation used this formula: (y1 - y2) = m(x1 - x2) to estimate the missing point at y-optima(pixel 540)
9. Once I had the 4 points, I used `cv2.line` function to draw lines on the input image


###2. Identify potential shortcomings with your current pipeline

I think, one potential shortcoming is that the pipeline is based on some assumptions:

1. The camera is going to be always in the same position on the car. This made the calculations, parameter estimations a bit easier.
2. The lane line are consistently clear on the road. What if there is a bald patch (even if partially)
3. Did not work on the challenge video


###3. Suggest possible improvements to your pipeline

A possible improvement would be to improve accuracy. I would like to look deeper into more accurate methods of extrapolating
points given my current state of hough outputs. I would also like to see if there are better parameters for my canny_edge, ROI and hough functions.

Another potential improvement could be robustness. Building a pipeline that works despite of some assumptions and adapts with different road conditions
