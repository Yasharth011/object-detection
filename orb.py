import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np 

# take image path as input from cmd line 
parser = argparse.ArgumentParser()

parser.add_argument("-i", help="image to detect")
args = parser.parse_args()
img_path = args.i

# placeholder image to compare orb features 
query = cv2.imread("query.png")

# image to detect 
train = cv2.imread(img_path)

# gray scale both images 
gray_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
gray_train = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

# canny edge detection on both images
query_edges = cv2.Canny(gray_query, 100, 200)
train_edges = cv2.Canny(gray_train, 100, 200)

# object for cv2 ORB 
orb = cv2.ORB_create()

# extract orb features 
query_keypoints, query_descriptors = orb.detectAndCompute(gray_query, None)
train_keypoints, train_descriptors = orb.detectAndCompute(gray_query, None)

# object for matching orb features 
matcher = cv2.BFMatcher()

# extract matches in orb features 
matches = matcher.match(query_descriptors, train_descriptors)

# sort the matches according to distance 
matches = sorted(matches, key = lambda x:x.distance)

# taking first 10 matches 
good_matches = matches[:10]

# extract coordinates of the features 
src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([ train_keypoints[m.trainIdx].pt for m in good_matches])

# feature matching and homography estimation
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

h, w = query.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts,M)
dst += (w, 0)  # adding offset

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)

img3 = cv2.drawMatches(query, query_keypoints, train, train_keypoints,good_matches, None,**draw_params)

# Draw bounding box in Red
img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
img3 = cv2.resize(img3, (400,400))
cv2.imshow("result", img3)
cv2.waitKey()
