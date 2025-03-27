import argparse
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-i", help="image to detect")
args = parser.parse_args()
img_path = args.i

query = cv2.imread("query.png")

train = cv2.imread(img_path)

gray_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
gray_train = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

query_edges = cv2.Canny(gray_query, 100, 200)
train_edges = cv2.Canny(gray_train, 100, 200)

orb = cv2.ORB_create()

query_keypoints, query_descriptors = orb.detectAndCompute(gray_query, None)
train_keypoints, train_descriptors = orb.detectAndCompute(gray_query, None)

matcher = cv2.BFMatcher()
matches = matcher.match(query_descriptors, train_descriptors)

final_img = cv2.drawMatches(
    query, query_keypoints, train, train_keypoints, matches[:20], None
)


# query_with_orb = cv2.drawKeypoints(
#     query, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )

plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

plt.pause(10)
