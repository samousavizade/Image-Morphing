import dlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as pp


class ImageMorphing:
    def __init__(self,
                 first_image,
                 second_image,
                 frame_rate):

        self.first_image = first_image
        self.second_image = second_image
        self.height, self.width = first_image.shape[0], first_image.shape[1]
        self.frame_rate = frame_rate
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.frames = list()
        self.BORDER_MODE = cv.BORDER_REFLECT_101
        self.FLAGS = cv.INTER_LINEAR
        self.POINTS_NUMBER = 76

    def calculate_morphing(self):
        # get corresponding points
        first_image_points, second_image_points, corresponding_points = self.calculate_corresponding_points()

        # calculate triangulation on both image
        triangles = self.delaunay_triangulation(corresponding_points)

        # calculate frames based on triangles
        self.calculate_frames(first_image_points,
                              second_image_points,
                              triangles)

    def calculate_frames(self, first_image_points, second_image_points, triangles):

        def to_list(input):
            lst = list()
            for idx in range(self.POINTS_NUMBER):
                lst.append((int(input[idx, 0]),
                            int(input[idx, 1])))

            return lst

        # numpy array to list
        first_image_points_list = to_list(first_image_points)
        second_image_points_list = to_list(second_image_points)

        frames_number = int(self.frame_rate)

        weight = 0

        # generate frames
        for j in range(0, frames_number):
            points = list()
            weight += 1 / (frames_number - 1)

            # change uint8 to float32
            self.first_image = np.float32(self.first_image)
            self.second_image = np.float32(self.second_image)

            # for all 2 corresponding point calculate weighted mean point
            for i in range(0, len(first_image_points_list)):
                def weighted_sum(first_point, second_point, w):
                    return (1 - w) * first_point[1] + w * second_point[1], \
                           (1 - w) * first_point[0] + w * second_point[0],

                y, x = weighted_sum(first_image_points[i], second_image_points_list[i], w=weight)
                points.append((x, y))

            shape = self.first_image.shape
            dtype = self.first_image.dtype

            # initiate frame array
            self.current_frame = np.zeros(shape, dtype)

            # for all triangle calculate warped triangle
            for i in range(len(triangles)):
                x, y, z = int(triangles[i][0]), int(triangles[i][1]), int(triangles[i][2])

                def get_corresponding_triangles(x, y, z):
                    def get_triangle(lst, a, b, c):
                        return [lst[a], lst[b], lst[c]]

                    tri = get_triangle(points, x, y, z)
                    tri1 = get_triangle(first_image_points, x, y, z)
                    tri2 = get_triangle(second_image_points, x, y, z)

                    return tri, tri1, tri2

                triangle, triangle1, triangle2 = get_corresponding_triangles(x, y, z)
                self.morph_triangle(triangle1, triangle2, triangle, weight)

            # add frame to frame list
            self.frames.append(np.uint8(self.current_frame))

    def build(self, output_file_name, shape, circular=False):

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        file_name = output_file_name
        out = cv.VideoWriter(file_name, fourcc, 15, (shape[1], shape[0]))
        n = len(self.frames)
        for j in range(n - 1):
            pp.imsave('frames/frame' + str(j) + '.jpg', cv.cvtColor(self.frames[j], cv.COLOR_BGR2RGB))
            out.write(self.frames[j])

        if circular:
            for j in range(1, n):
                out.write(self.frames[n - 1 - j])

        out.release()

    def morph_triangle(self, first_tri, second_tri, triangle, a):
        t1r, t2r, tr = [], [], []
        dtype = np.float32
        first_rect = cv.boundingRect(np.float32([first_tri]))
        second_rect = cv.boundingRect(np.float32([second_tri]))
        rectangle = cv.boundingRect(np.float32([triangle]))

        for i in range(3):
            tr.append(((triangle[i][0] - rectangle[0]),
                       (triangle[i][1] - rectangle[1])))
            t1r.append(((first_tri[i][0] - first_rect[0]),
                        (first_tri[i][1] - first_rect[1])))
            t2r.append(((second_tri[i][0] - second_rect[0]),
                        (second_tri[i][1] - second_rect[1])))

        mask = np.zeros((rectangle[3], rectangle[2], 3), dtype=dtype)
        cv.fillConvexPoly(mask, np.int32(tr), (1.0, 1.0, 1.0), 16, 0)

        first_mask = ImageMorphing.get_mask(self.first_image, first_rect)
        second_mask = ImageMorphing.get_mask(self.second_image, second_rect)

        shape = (rectangle[2], rectangle[3])
        first_image_warp = ImageMorphing.affine_transform(first_mask, t1r, tr, shape)
        second_image_warp = ImageMorphing.affine_transform(second_mask, t2r, tr, shape)

        warped = (1.0 - a) * first_image_warp + a * second_image_warp

        self.current_frame[rectangle[1]:rectangle[1] + rectangle[3], rectangle[0]:rectangle[0] + rectangle[2]] = \
            self.current_frame[rectangle[1]:rectangle[1] + rectangle[3], rectangle[0]:rectangle[0] + rectangle[2]] * \
            (1 - mask) + warped * mask

    @staticmethod
    def get_mask(image, rect):
        return image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

    @staticmethod
    def affine_transform(mask, source, target, shape):
        transform_matrix = cv.getAffineTransform(np.float32(source), np.float32(target))

        dst = cv.warpAffine(mask,
                            transform_matrix,
                            (shape[0], shape[1]),
                            None,
                            flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_REFLECT_101)

        return dst

    def delaunay_triangulation(self, corresponding_points):
        rectangle = (0, 0, self.width, self.height)
        # initiate subdiv2d object
        sub_div = cv.Subdiv2D(rectangle)
        points_list = [(int(corresponding_points[i, 0]),
                        int(corresponding_points[i, 1]))
                       for i in range(self.POINTS_NUMBER)]

        points_dict = {points_list[i]: i
                       for i in range(self.POINTS_NUMBER)}

        for point in points_list:
            sub_div.insert(point)

        # get triangles
        triangles = self.get_all_triangles(sub_div, points_dict)

        return triangles

    def get_all_triangles(self, sub_div, pt_to_idx):
        triangles = list()
        triangles_list = sub_div.getTriangleList()

        def is_valid(pt1, pt2, pt3):
            return 0 <= pt1[0] <= self.width and 0 <= pt1[1] <= self.height and \
                   0 <= pt2[0] <= self.width and 0 <= pt2[1] <= self.height and \
                   0 <= pt3[0] <= self.width and 0 <= pt3[1] <= self.height

        for triangle in triangles_list:
            a = (int(triangle[0]), int(triangle[1]))
            b = (int(triangle[2]), int(triangle[3]))
            c = (int(triangle[4]), int(triangle[5]))
            if is_valid(a, b, c):
                triangles.append((pt_to_idx[a], pt_to_idx[b], pt_to_idx[c]))

        return triangles

    def get_points(self, image):
        height, width = self.height, self.width

        # detect main points in face
        detected = self.detector(image, 1)

        # initiate main points array
        points = np.zeros((self.POINTS_NUMBER, 2))

        # for all detected face (in this case: 1)
        for _, rect in enumerate(detected):
            # get points
            shape = self.predictor(image, rect)

            # add points to array
            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                points[i, :] = np.array([x, y])

            # add marginal points to array
            ImageMorphing.add_marginal_points(height, width, points)

        return points

    @staticmethod
    def add_marginal_points(height, width, points):
        points[68, :] = np.array([1, 1])
        points[69, :] = np.array([width - 1, 1])
        points[70, :] = np.array([(width - 1) // 2, 1])
        points[71, :] = np.array([1, height - 1])
        points[72, :] = np.array([1, (height - 1) // 2])
        points[73, :] = np.array([(height - 1) // 2, width - 1])
        points[74, :] = np.array([width - 1, height - 1])
        points[75, :] = np.array([(width - 1), (height - 1) // 2])

    def calculate_corresponding_points(self):
        # get main points in first image
        first_im_points = self.get_points(self.first_image)

        # get main points in second image
        second_im_points = self.get_points(self.second_image)

        # calculate corresponding points as mean
        corresponding_points = (first_im_points + second_im_points) / 2

        return first_im_points, second_im_points, corresponding_points


def main():
    # initiate configuration
    frame_rate = 120
    circular = False
    image1_file = 'Khamenei.jpg'
    image2_file = 'Khomeini.jpg'
    output_file_name = 'result_circular.mp4' if circular else 'result.mp4'

    # read images
    first_image = cv.imread(image1_file)
    second_image = cv.imread(image2_file)

    # initiate morphing object
    image_morphing = ImageMorphing(first_image,
                                   second_image,
                                   frame_rate)
    # calculate frames
    image_morphing.calculate_morphing()

    # build mp4 video with calculated frames
    image_morphing.build(output_file_name,
                         first_image.shape,
                         circular=circular)


if __name__ == '__main__':
    main()
