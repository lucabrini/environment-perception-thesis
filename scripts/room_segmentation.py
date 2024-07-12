import copy
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from skimage.morphology import medial_axis

def plot_image(image, filename):
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


class Origin:
    
    origin_x: float
    origin_y: float
    
    def __init__(self, origin_x: float, origin_y: float):
        self.origin_x = origin_x
        self.origin_y = origin_y

class RoomSegmentationJob:
    """
    This Job is responsible for segmenting a map into rooms and finding the door locations in order to produce
    a topological rooms graph

    The job is divided into the following steps:
        1. Binarize the map into free space and occupied space
        2. Analyze the free space to find the segments
        3. Analyze the occupied space to find the endpoints
        4. Find the door locations
        5. Separate the rooms
        6. Create a graph from the segmentation
    """

    map_resolution: float
    map_origin: List[float]
    free_space_image: np.array
    occupied_space_image: np.array

    free_space_skeleton: np.array
    occupied_space_skeleton: np.array

    robot_diameter_mt = 1.0

    def __init__(self, image: str, map_resolution: float, map_origin: Origin):

        """
        :param image: Bytes string representing the map image file
        :param map_resolution: float representing the resolution of the map in meters per pixel
        :param map_origin: Origin object representing the origin of the map
        """

        image = np.fromstring(image, np.uint8)
        self.image = cv.imdecode(image, cv.IMREAD_GRAYSCALE)
        self.map_resolution = map_resolution
        self.map_origin = map_origin

    def start(self):
        self.free_space_image, self.occupied_space_image = self.binarize()
        self.free_space_image = self.preprocess_free_space_image()
        self.occupied_space_image = self.preprocess_occupied_space_image()
        
        segments_list = self.analyze_free_space()
        # PLOT
            
        endpoints, endpoints_coords, threshold = self.analyze_occupied_space()

        # Find door locations
        door_locations = self.find_door_locations(segments_list, endpoints_coords, threshold)
        num_labels, labels, centroids, labels_segments_list = self.separate_rooms(door_locations)
        
        rooms_bboxs = RoomSegmentationUtils.get_bbs_from_rooms_labels(labels, num_labels)
        for bbox in rooms_bboxs:
            cv.rectangle(self.image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)

        rooms_segments = []
        for i in range(1, num_labels):
            mask = labels == i
            mask = mask.astype(np.uint8) * 255

            contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            print(hierarchy, contours)
            # hierarchy node: [next, previous, first child, parent]
            # next: Next contour at the same hierarchy level
            # previous: Previous contour at the same hierarchy level
            # first child: First child contour
            # parent: Parent contour
            room_segments = []
            for polyline in contours:
                if len(polyline) >= 3:
                    polyline = cv.approxPolyDP(polyline, 0.002 * cv.arcLength(polyline, True), True)
                    polyline = np.squeeze(polyline)
                    polyline = [np.round(self.to_world_pos(p[0], self.image.shape[0] - p[1]), 3) for p in polyline]
                    room_segments.append(polyline)

            rooms_segments.append(room_segments)

        #nodes, edges = RoomSegmentationUtils.segmentation_to_graph(labels, num_labels, centroids, labels_segments_list,
        #                                                           rooms_segments)

        return labels

    def binarize(self):
        _, free_space_image = cv.threshold(self.image, 220, 255, cv.THRESH_BINARY)
        _, occupied_space_image = cv.threshold(self.image, 10, 255, cv.THRESH_BINARY_INV)
        return free_space_image, occupied_space_image

    def preprocess_free_space_image(self):
        strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        map_image = cv.erode(self.free_space_image, strel)
        map_image = cv.medianBlur(map_image, 5)

        return map_image

    def preprocess_occupied_space_image(self):
        strel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        map_image = cv.dilate(self.occupied_space_image, strel)

        return map_image

    def voronoi_free_space(self, area_threshold=10):
        self.free_space_skeleton, skeleton_coords = RoomSegmentationUtils.skeletonize_image(self.free_space_image)
        
        # Find branching points
        branching_points = RoomSegmentationUtils.find_branching_points(self.free_space_skeleton)

        self.free_space_skeleton = self.free_space_skeleton * 255
        self.free_space_skeleton = self.free_space_skeleton - np.bitwise_and(self.free_space_skeleton, branching_points)

        # Find branches
        branches_num, branches_labels = RoomSegmentationUtils.filter_small_branches(self.free_space_skeleton, area_threshold)
        self.free_space_skeleton = np.zeros_like(self.free_space_skeleton)
        self.free_space_skeleton[branches_labels > 0] = 255
        
        return branches_num, branches_labels
        
    def analyze_free_space(self, area_threshold=10):
        branches_num, branches_labels = self.voronoi_free_space()
        # Compute distance transform
        distances = cv.distanceTransform(self.free_space_image, cv.DIST_L2, 0)
        # For each branch, find the point in the branch whose distance is the minimum
        segments_list = []
        for i in range(1, branches_num):
            segments = RoomSegmentationUtils.get_segments_from_branch(area_threshold, distances, branches_labels == i)
            segments_list.extend(segments)

        for segment in segments_list:
            # Find nearest point along perpendicular orientation
            r_centroid = np.round(segment.centroid).astype(np.uint16)
            dist = distances[r_centroid[1], r_centroid[0]]
            perpendicular_orientation = segment.orientation + np.pi / 2

            first_closest_point = np.round(
                segment.centroid + np.array(
                    [np.cos(perpendicular_orientation), np.sin(perpendicular_orientation)]) * dist).astype(np.uint16)

            perpendicular_orientation = segment.orientation - np.pi / 2
            second_closest_point = np.round(
                segment.centroid + np.array(
                    [np.cos(perpendicular_orientation), np.sin(perpendicular_orientation)]) * dist).astype(np.uint16)

            segment.set_intersection_points(first_closest_point, second_closest_point)

        return segments_list

    def analyze_occupied_space(self):
        self.occupied_space_skeleton, skeleton_coords = RoomSegmentationUtils.skeletonize_image(
            self.occupied_space_image)
        self.occupied_space_skeleton = self.occupied_space_skeleton * 255
        plot_image(self.occupied_space_skeleton, "occupied_space_voronoi_diagram")
        
        branches_num, branches_labels = RoomSegmentationUtils.filter_small_branches(self.occupied_space_skeleton, 20)
        color_labels = np.uint8(255 * branches_labels / np.max(branches_labels))
        color_labels = cv.applyColorMap(color_labels, cv.COLORMAP_HSV)
        color_labels[np.where(branches_labels == 0)] = [0, 0, 0]
        plot_image(color_labels, "occupied_space_diagram_branches_labeling")
        
        branches_labels_out = branches_labels.astype(np.uint8) * 255
        branches_labels_out = cv.applyColorMap(branches_labels_out, cv.COLORMAP_JET)

        branches_labels = branches_labels.astype(np.uint8)
        _, self.occupied_space_skeleton = cv.threshold(branches_labels, 0, 255, cv.THRESH_BINARY)

        distances = np.round(cv.distanceTransform(self.occupied_space_image, cv.DIST_L2, 0))
        # Get the distance values of the skeleton
        skeleton_distance_values = distances[np.where(self.occupied_space_skeleton)]
        wall_thickness = np.mean(skeleton_distance_values) * 2

        endpoints, endpoints_coords = RoomSegmentationUtils.find_end_points(self.occupied_space_skeleton)
        
        bg = np.zeros((self.occupied_space_skeleton.shape[0], self.occupied_space_skeleton.shape[1], 3), np.uint8)
        bg[np.where(self.occupied_space_skeleton == 255)] = [255, 255, 255]
        for endpoint in endpoints_coords:
            cv.circle(bg, (endpoint[0], endpoint[1]), 2, (255, 0, 0), -1)
        plot_image(bg, "occupied_space_diagram_endpoints")
        return endpoints, endpoints_coords, wall_thickness

    def find_door_locations(self, segment_list: List['Segment'], endpoints_coords: List[np.array],
                            wall_thickness: float):
        # TODO: Regarding efficiency, It could be a lot better to use a KDTree to find the closest endpoints
        for segment in segment_list:
            # Find the closest endpoint to each segment intersection point
            closest_endpoints = []
            for intersection_point in segment.intersection_points:
                min_distance = np.inf
                closest_endpoint = None
                for endpoint in endpoints_coords:
                    distance = np.linalg.norm(endpoint - intersection_point)
                    if distance < min_distance:
                        min_distance = distance
                        closest_endpoint = endpoint
                closest_endpoints.append(closest_endpoint)

            segment.set_closest_endpoints(closest_endpoints)
                        
            c_distance, ei_a_distance, ei_b_distance = segment.distances

            is_door_present = (
                    c_distance <= wall_thickness / 2 and
                    ei_a_distance <= wall_thickness and
                    ei_b_distance <= wall_thickness
            )

            if is_door_present:
                lines_img = np.zeros((segment.mask.shape[0], segment.mask.shape[1], 3), np.uint8)
                cv.line(lines_img, (segment.intersection_points[0][0], segment.intersection_points[0][1]),
                        (segment.intersection_points[1][0], segment.intersection_points[1][1]), (255, 255, 255), 1)

                cv.line(lines_img, (segment.closest_endpoints[0][0], segment.closest_endpoints[0][1]),
                        (segment.closest_endpoints[1][0], segment.closest_endpoints[1][1]), (255, 255, 255), 1)

                lines_img = cv.cvtColor(lines_img, cv.COLOR_BGR2GRAY)
                self.occupied_space_skeleton = self.occupied_space_skeleton - np.bitwise_and(
                    self.occupied_space_skeleton,
                    self.free_space_image
                )
                results = cv.bitwise_and(lines_img, self.occupied_space_skeleton, lines_img)
                results = cv.countNonZero(results)
                if results <= 2:
                    if abs(segment.door_length - self.robot_diameter_mt / self.map_resolution) <= 0.5 / self.map_resolution:
                        segment.set_door_location(True)
                    else:
                        segment.set_reason("Segment length is less than robot diameter")
                else:
                    segment.set_reason("Segment intersects with occupied space")
            else:
                segment.set_reason("Door not present")
                
        return segment_list

    def separate_rooms(self, segments_list: List['Segment']):
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(self.free_space_image, connectivity=4,
                                                                               ltype=cv.CV_32S)

        for i in range(1, num_labels):
            if stats[i][4] < 20:
                labels[labels == i] = 0

        out_segments_list = []
        for segment in segments_list:
            if segment.door_location:
                bg = np.zeros((self.free_space_image.shape[0], self.free_space_image.shape[1]), np.uint8)

                cv.line(bg, (segment.intersection_points[0][0], segment.intersection_points[0][1]),
                        (segment.intersection_points[1][0], segment.intersection_points[1][1]), (255, 255, 255), 2)
                cv.circle(bg, (segment.intersection_points[0][0], segment.intersection_points[0][1]), 2,
                          (255, 255, 255),
                          -1)
                cv.circle(bg, (segment.intersection_points[1][0], segment.intersection_points[1][1]), 2,
                          (255, 255, 255),
                          -1)

                temp_free_space_image = self.free_space_image - np.bitwise_and(self.free_space_image, bg)
                new_out = cv.connectedComponentsWithStats(temp_free_space_image, connectivity=4, ltype=cv.CV_32S)
                new_num_labels, labels, _, centroids = new_out
                if new_num_labels > num_labels:
                    out_segments_list.append(segment)
                    num_labels = new_num_labels
                    self.free_space_image = temp_free_space_image

        # plot the labels
        return num_labels, labels, centroids, out_segments_list

    def to_world_pos(self, map_x, map_y):
        world_x = map_x * self.map_resolution + self.map_origin.origin_x
        world_y = map_y * self.map_resolution + self.map_origin.origin_y
        return world_x, world_y


class RoomSegmentationUtils:

    @staticmethod
    def find_end_points(skeleton: np.array):
        t1 = np.array([[-1, -1, -1],
                       [-1, 1, -1],
                       [0, 1, 0]])
        t2 = np.array([[-1, -1, -1],
                       [-1, 1, 0],
                       [-1, 0, 1]])

        t3 = np.rot90(t1)
        t4 = np.rot90(t2)
        t5 = np.rot90(t3)
        t6 = np.rot90(t4)
        t7 = np.rot90(t5)
        t8 = np.rot90(t6)

        kernels = [t1, t2, t3, t4, t5, t6, t7, t8]
        
        results = np.zeros(skeleton.shape[:2], dtype=int)
        for kernel in kernels:
            results = np.logical_or(cv.morphologyEx(skeleton, op=cv.MORPH_HITMISS, kernel=kernel,
                                                    borderType=cv.BORDER_CONSTANT, borderValue=0), results)
        endpoints = results.astype(np.uint8) * 255
        endpoints_coords = list(zip(*np.where(endpoints == 255)))
        endpoints_coords = [np.array([x, y]) for y, x in endpoints_coords]

        return endpoints, endpoints_coords

    @staticmethod
    def find_branching_points(skeleton: np.array):
        t1 = np.array([
            [-1, 1, -1],
            [1, 1, 1],
            [-1, -1, -1]
        ])
        t2 = np.array([
            [1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1]
        ])

        t3 = np.rot90(t1)
        t4 = np.rot90(t2)
        t5 = np.rot90(t3)
        t6 = np.rot90(t4)
        t7 = np.rot90(t5)
        t8 = np.rot90(t6)

        # Y like branch points
        y1 = np.array([[1, -1, 1],
                       [0, 1, 0],
                       [0, 1, 0]])
        y2 = np.array([[-1, 1, -1],
                       [1, 1, 0],
                       [-1, 0, 1]])
        y3 = np.rot90(y1)
        y4 = np.rot90(y2)
        y5 = np.rot90(y3)
        y6 = np.rot90(y4)
        y7 = np.rot90(y5)
        y8 = np.rot90(y6)

        kernels = [t1, t2, t3, t4, t5, t6, t7, t8, y1, y2, y3, y4, y5, y6, y7, y8]
        branch_pts_img = np.zeros(skeleton.shape[:2], dtype=int)

        # Store branch points
        for kernel in kernels:
            branch_pts_img = np.logical_or(cv.morphologyEx(skeleton, op=cv.MORPH_HITMISS, kernel=kernel,
                                                           borderType=cv.BORDER_CONSTANT, borderValue=0),
                                           branch_pts_img)

        branch_pts_img = branch_pts_img.astype(np.uint8) * 255
        
        # Dilating the image to make the points more visible
        skel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) * 255
        branch_pts_img = cv.dilate(branch_pts_img, skel)


        return branch_pts_img

    @staticmethod
    def skeletonize_image(map_image: np.array):
        skeleton = medial_axis(map_image)
        skeleton = skeleton.astype(np.uint8)
        skeleton_coords = np.where(True)
        return skeleton, skeleton_coords

    @staticmethod
    def get_segments_from_branch(area_threshold, distances: np.array, branch_mask: np.array):
        segments_list = []

        # Mask the distance transform with the branch
        branch_distance_masked = cv.bitwise_and(
            distances,
            distances,
            mask=branch_mask.astype(np.uint8) * 255
        )
        # Get the distance values of the branch
        branch_distance_values = distances[np.where(branch_mask)]
        if len(branch_distance_values) != 0:
            # Find local minima
            min_distance = np.min(branch_distance_values)
            upperbound = min_distance + 2
            lowerbound = min_distance - 2 if min_distance - 2 > 0 else 1
            segments = cv.inRange(branch_distance_masked, lowerbound, upperbound)

            out = cv.connectedComponentsWithStats(segments, connectivity=8, ltype=cv.CV_32S)
            segments_num, segments_labels, segments_stats, segments_centroids = out

            for j in range(1, segments_num):
                if segments_stats[j][4] < area_threshold:
                    segments_labels[segments_labels == j] = 0
                else:
                    segment_mask = segments_labels == j
                    segment_mask = segment_mask.astype(np.uint8) * 255
                    segment_centroid = segments_centroids[j]

                    # Segment bounding box
                    x, y, w, h, _ = segments_stats[j]

                    # Slice the segment from the skeleton plus some padding
                    sliced_segment = segment_mask[y - 3:y + h + 3, x - 3:x + w + 3]
                    sliced_segment = sliced_segment.astype(np.uint8)

                    endpoints, endpoints_coords = RoomSegmentationUtils.find_end_points(sliced_segment)

                    a = endpoints_coords[0]
                    b = endpoints_coords[1]

                    if a[0] == b[0]:
                        segment_orientation = np.pi / 2
                    else:
                        m = (b[1] - a[1]) / (b[0] - a[0])
                        segment_orientation = np.arctan(m)

                        if a[0] < b[0] and a[1] < b[1]:
                            segment_orientation = np.pi - segment_orientation
                        else:
                            segment_orientation = -segment_orientation

                    segment_orientation = segment_orientation

                    segments_list.append(
                        Segment(segment_centroid, segment_orientation, segment_mask,
                                list(np.array(endpoints_coords) + [x, y]))
                    )

        return segments_list

    @staticmethod
    def filter_small_branches(skeleton: np.array, area_threshold: int):
        out = cv.connectedComponentsWithStats(copy.deepcopy(skeleton), connectivity=8, ltype=cv.CV_32S)
        branches_num, branches_labels, branches_stats, branches_centroids = out

        # Remove labels with less than 10 pixels
        for i in range(1, branches_num):
            if branches_stats[i][4] < area_threshold:
                branches_labels[branches_labels == i] = 0
        branches_num = np.max(branches_labels)
        return branches_num, branches_labels

    @staticmethod
    def segmentation_to_graph(rooms_labels, num_rooms_labels, rooms_centroids, segments_list, rooms_segments):
        # Create a graph
        nodes = []
        edges = []

        bg = rooms_labels / np.max(rooms_labels)
        bg = (bg * 255).astype(np.uint8)
        bg = cv.applyColorMap(bg, cv.COLORMAP_HSV)
        bg[np.where(rooms_labels == 0)] = [0, 0, 0]

        # For each segmented table, create a node
        for i in range(1, num_rooms_labels):

            room_color = bg[rooms_labels == i][0]
            room_color = '#%02x%02x%02x' % tuple(np.flip(room_color))

            if len(rooms_segments[i - 1]) == 0:
                continue

            main_outline_points = [{
                "x": s[0],
                "y": s[1],
                "z": 0,
            } for s in rooms_segments[i - 1][0]]

            subtraction_outlines = []
            for raw_outline in rooms_segments[i - 1][1:]:
                subtraction_outlines.append({
                    "points": [{
                        "x": s[0],
                        "y": s[1],
                        "z": 0,
                    } for s in raw_outline]
                })

            node = SemanticMapRoomNode.load({
                "id": str(i),
                "name": "Room " + str(i),
                "layer_area": {
                    "id": str(i),
                    "label": "Room " + str(i),
                    "color": room_color,
                    "outline": {
                        "points": main_outline_points
                    },
                    "subtraction": subtraction_outlines
                },
                "objects": {
                    "nodes": [],
                    "edges": []
                }
            })
            nodes.append(node)

            cv.circle(bg, (int(rooms_centroids[i][0]), int(rooms_centroids[i][1])), 2, (0, 0, 0), -1)
            cv.putText(bg, str(node["id"]), (int(rooms_centroids[i][0]), int(rooms_centroids[i][1])),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.4, (0, 0, 0), 1, cv.LINE_AA)

        for segment in segments_list:
            if segment.door_location:
                source = rooms_labels[segment.endpoints[0][1]][segment.endpoints[0][0]]
                target = rooms_labels[segment.endpoints[1][1]][segment.endpoints[1][0]]
                edges.append(SemanticMapEdge.load({
                    "source": str(source),
                    "target": str(target),
                    "label": str(round(segment.door_length, 2)),
                }))

        return nodes, edges

    @staticmethod
    def get_bbs_from_rooms_labels(rooms_labels, num_rooms_labels):
        rooms_bbs = []
        for i in range(1, num_rooms_labels):
            mask = rooms_labels == i
            mask = mask.astype(np.uint8) * 255
            x, y, w, h = cv.boundingRect(mask)
            rooms_bbs.append([x, y, w, h])
        return rooms_bbs


class Segment:
    centroid: np.array
    orientation: float
    mask: np.array

    endpoints: List[np.array] = []
    intersection_points: List[np.array] = []
    closest_endpoints: List[np.array] = []
    door_location: bool = False
    reason = ""

    def __init__(self, centroid: np.array, orientation: float, mask: np.array, endpoints: List[np.array]):
        self.centroid = centroid
        self.orientation = orientation
        self.mask = mask
        self.endpoints = endpoints

    def set_endpoints(self, endpoints: List[np.array]):
        assert len(endpoints) == 2, "Endpoints must be two"
        self.intersection_points = endpoints

    def set_intersection_points(self, point_a: np.array, point_b: np.array):
        self.intersection_points = [point_a, point_b]

    def set_closest_endpoints(self, closest_endpoints: List[np.array]):
        assert len(closest_endpoints) == 2, "Endpoints must be two"
        self.closest_endpoints = closest_endpoints

    @property
    def distances(self):
        closest_endpoints_centroid = np.mean(self.closest_endpoints, axis=0)
        c_distance = np.linalg.norm(closest_endpoints_centroid - self.centroid)
        ei1_distance = np.linalg.norm(self.closest_endpoints[0] - self.intersection_points[0])
        ei2_distance = np.linalg.norm(self.closest_endpoints[1] - self.intersection_points[1])

        return np.round([c_distance, ei1_distance, ei2_distance])

    @property
    def door_length(self):
        return np.linalg.norm(
            self.intersection_points[0].astype(np.int16) - self.intersection_points[1].astype(np.int16))

    def set_door_location(self, door_location: bool):
        self.door_location = door_location

    def set_reason(self, reason: str):
        self.reason = reason

if __name__ == "__main__":
    
    with open("./images/room_recognition/grid_map.png", "rb") as f:
        image = f.read()
    print("ciao")
    RoomSegmentationJob(image, 0.05, Origin(0, 0)).start()