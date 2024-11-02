import pdb
import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        if len(all_images) < 2:
            print("Not enough images to stitch.")
            return None, []

        homography_matrix_list = []

        left_img = cv2.imread(all_images[0])
        for i in range(1, len(all_images)):
            right_img = cv2.imread(all_images[i])

            key_points1, descriptor1, key_points2, descriptor2 = self.get_keypoint(left_img, right_img)
            good_matches = self.match_keypoint(key_points1, key_points2, descriptor1, descriptor2)

            good_pts = np.array([[key_points1[m.queryIdx].pt[0], key_points1[m.queryIdx].pt[1], 
                                   key_points2[m.trainIdx].pt[0], key_points2[m.trainIdx].pt[1]]
                                  for m in good_matches])

            final_H = self.ransac(good_pts)

            if final_H is None:
                print("Homography could not be computed for image pair {}.".format(i))
            
                del right_img
                continue
            
            homography_matrix_list.append(final_H)
            left_img = self.stitch_images(left_img, right_img, final_H)

            del right_img

        print("Stitching completed.")
        return left_img, homography_matrix_list

    def get_keypoint(self, left_img, right_img):
        sift = cv2.SIFT_create()
        key_points1, descriptor1 = sift.detectAndCompute(left_img, None)
        key_points2, descriptor2 = sift.detectAndCompute(right_img, None)

        print(f"Keypoints in left image: {len(key_points1)}")
        print(f"Keypoints in right image: {len(key_points2)}")

        return key_points1, descriptor1, key_points2, descriptor2

    def match_keypoint(self, key_points1, key_points2, descriptor1, descriptor2):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptor1, descriptor2, k=2)
        good_matches = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        print(f"Good matches found: {len(good_matches)}")
        return good_matches

    def ransac(self, good_pts):
        best_inliers = []
        final_H = None
        t = 5  # Threshold for distance to consider a point as an inlier
        for i in range(10):
            random_pts = random.sample(good_pts.tolist(), k=4)  # Randomly sample 4 points
            H = self.homography(random_pts)
            inliers = []
            for pt in good_pts:
                p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                Hp = np.dot(H, p)
                Hp /= Hp[2]  
                dist = np.linalg.norm(p_1 - Hp)

                if dist < t: 
                    inliers.append(pt)

            if len(inliers) > len(best_inliers):
                best_inliers, final_H = inliers, H

        return final_H

    def homography(self, points):
        A = []
        for pt in points:
            x, y = pt[0], pt[1]
            X, Y = pt[2], pt[3]
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

        A = np.array(A)
        u, s, vh = np.linalg.svd(A)
        H = (vh[-1, :].reshape(3, 3))
        H /= H[2, 2]  
        return H

    def stitch_images(self, left_img, right_img, final_H):
        rows1, cols1 = right_img.shape[:2]
        rows2, cols2 = left_img.shape[:2]

        points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        points2 = cv2.perspectiveTransform(points, final_H)
        list_of_points = np.concatenate((points1, points2), axis=0)

        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        H_translation = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(final_H)

        output_img = cv2.warpPerspective(left_img, H_translation, (x_max - x_min, y_max - y_min))
        output_img[(-y_min):rows1 + (-y_min), (-x_min):cols1 + (-x_min)] = right_img

        del left_img

        return output_img

    def say_hi(self):
        print('Hii From Ishva Patel..')

    def do_something(self):
        return None

    def do_something_more(self):
        return None

