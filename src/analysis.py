import os
from abc import ABC, abstractmethod

import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt

from .functionality import load_images, image_preproc, image_fractions_preproc
from .config import (
    DESTINATION_FOLDER_SEM_PATCHED,
    FOLDER_SEM,
    parameters,
    DESTINATION_FOLDER_FRACTIONS,
    FOLDER_FRACTION,
)
from .plot import StitchedSemPlotter, FractionPlotter


class BaseProcessor(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _pre_process(self):
        pass

    @abstractmethod
    def _process(self):
        pass

    @abstractmethod
    def _post_process(self):
        pass

    def run(self):
        self._pre_process()
        self._process()
        self._post_process()


class StitchedSemProcessor(BaseProcessor):

    def __init__(self):
        self.plotter = StitchedSemPlotter()
        self.all_kde_curves = []
        self.all_grain_data = []
        self.k = 9  # Degree of blurring - must be an odd number

    def _pre_process(self):
        self.images = load_images(FOLDER_SEM)

    def _process(self):
        """
        Perform SEM segmentation
        """
        for image, filename in self.images:
            # Filters & artifact correction
            filtered_image = image_preproc(image, self.k)
            og_shape = filtered_image.shape # store the shape before any flattening

            sorted_labels, sorted_centers = self._apply_kmeans(filtered_image)

            # Statistics
            mineral_prctage = self._calc_mineral_percentages(sorted_labels)
            iron_prctage_oxides = self._calc_iron_prctage(mineral_prctage, sorted_centers)

            # Drawing contours on brightest particles
            display_image, contours = sem_contour_on_magnetite(
                og_shape,
                filtered_image,
                sorted_labels,
                sorted_centers,
                filename,
                DESTINATION_FOLDER_SEM_PATCHED,
            )

            self.plotter.visualize_contours(display_image)

            self.plotter.display_orig_and_sorted_images(
                image, sorted_labels, sorted_centers, og_shape
            )

            # Calculating distances and centroids
            distances = distances_of_centroids(contours)

            data = self.plotter.precompute_sem_plot_data(contours, distances, filename)

            self.plotter.plot_data(mineral_prctage, iron_prctage_oxides, filename, **data)

            self._save_some_data_for_later(data, filename)

    def _post_process(self):
        self.plotter.plot_kde(self.all_kde_curves)
        self.plotter.plot_grain_data(self.all_grain_data)

    def _save_some_data_for_later(self, data, filename):
        self.all_kde_curves.append(
            {
                "dist_range": data["dist_range"],
                "kde_values": data["kde_values"],
                "color": data["color"],
                "filename": filename,
            }
        )
        self.all_grain_data.append(
            {
                "areas_mm2": data["areas_mm2"],
                "color": data["color"],
                "filename": filename,
            }
        )

    ############################################ Functions ############################################

    def _apply_kmeans(self, image):
        """
        Finds sorted lables and centers using kmeans
        """
        # Necessary conversions to put the image in the k-means algorithm
        image_reshape = image.reshape(-1, 1)
        image_reshape_float = np.float32(image_reshape)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 4
        attempts = 10
        ret, label, center = cv2.kmeans(
            image_reshape_float,
            k,
            None,
            criteria,
            attempts,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        # Sorting labels and centers with increasing values of brightness
        sorted_indices = np.argsort(np.sum(center, axis=1))
        sorted_centers = center[sorted_indices]
        print(
            "Sorted Centers:", sorted_centers
        )  # Print the sorted centers for inspection

        # Create a mapping from old indices to new indices
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}

        # Update labels using the mapping
        sorted_labels = np.vectorize(mapping.get)(label)
        return sorted_labels, sorted_centers

    def _calc_mineral_percentages(self, labels):
        """
        Takes in sorted labels.
        """
        # Calculating mineral percentages - Neph, Feld, Pyr-Amph, Oxides
        mineral_percentages = sem_quantification(labels)

        # Percentages of minerals
        for mineral, percentage in mineral_percentages.items():
            print(f"{mineral}: {percentage:.2f}%")

        return mineral_percentages

    def _calc_iron_prctage(self, mineral_prctage, sorted_centers):
        """
        Calculates iron percentages from oxides.
        """
        # Percentage of iron
        iron_from_oxides = iron_quantity(mineral_prctage, sorted_centers)
        print(f"iron percentage from oxides: {iron_from_oxides:.2f} %")
        return iron_from_oxides


def sem_quantification(sorted_labels):
    # Mapping each cluster to a mineral
    mineral_mapping = {
        0: "nepheline",
        1: "feldspar",
        2: "pyroxenes_amphibole",
        3: "oxides",
    }

    # Convert numpy array to a list if not already in an appropriate format
    label_list = (
        sorted_labels.flatten().tolist()
    )  # This ensures that sorted_labels is in a hashable, countable form
    label_counts = Counter(label_list)

    # Total amount of pixels to calculate percentages
    total_pixels = len(sorted_labels)

    # Calculating percentages of each mineral
    mineral_percentages = {
        mineral: (label_counts[label] / total_pixels) * 100
        for label, mineral in mineral_mapping.items()
    }

    return mineral_percentages


def iron_quantity(mineral_percentages, sorted_centers):
    last_center_value = sorted_centers[-1, 0]
    print(last_center_value)
    iron_rich_threshold = 210

    magnetite_percentage = mineral_percentages.get(
        "oxides", 0
    )  # Extracting the magnetite percentage which = oxides

    # Atomic weights
    atomic_weight_Fe = 55.845  # Iron
    atomic_weight_O = 15.999  # Oxygen
    molecular_weight_Fe3O4 = (
        3 * atomic_weight_Fe + 4 * atomic_weight_O
    )  # Molecular weight of Fe3O4
    percentage_Fe_in_Fe3O4 = (
        3 * atomic_weight_Fe
    ) / molecular_weight_Fe3O4  # Percentage of iron in Fe3O4
    total_iron_percentage = magnetite_percentage * percentage_Fe_in_Fe3O4

    if last_center_value > iron_rich_threshold:
        print(
            f"Percentage of magnetite (and/or other) in the image: {magnetite_percentage:.2f}%"
        )
    else:
        print(
            "No significant magnetite (and/or other) cluster found based on the threshold."
        )

    iron_percentage_from_magnetite = (
        total_iron_percentage  # Returning the value of Iron quantity
    )

    return iron_percentage_from_magnetite


def sem_contour_on_magnetite(
    original_shape,
    filtered_image,
    sorted_labels,
    sorted_centers,
    filename,
    DESTINATION_FOLDER_SEM_PATCHED,
):
    # Drawing contours on brightest particles
    brightest_cluster_label = len(sorted_centers) - 1

    # Create a mask for the brightest cluster
    brightest_mask = (sorted_labels == brightest_cluster_label).reshape(original_shape)

    # If original image is grayscale, convert it to BGR for contour drawing
    if len(original_shape) == 2:  # Grayscale image
        display_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    else:
        display_image = filtered_image.copy()

    # Find contours
    contours, hierarchy = cv2.findContours(
        brightest_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Draw contours on the original image
    cv2.drawContours(display_image, contours, -1, (0, 255, 0), 2)  # Green contours

    base_name = os.path.splitext(filename)[0]
    new_filename = f"{base_name}_contour.jpg"
    save_path = os.path.join(DESTINATION_FOLDER_SEM_PATCHED, new_filename)
    cv2.imwrite(save_path, display_image)

    return display_image, contours


def distances_of_centroids(contours):
    # Centroids of Iron rich minerals
    centroids = []  # Array for storing centers of contours

    for contour in contours:
        M = cv2.moments(
            contour
        )  # Multiple characteristics of an object (in this case contour)

        # Calculating x,y coordinates of the center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])  # m10/m00 gives x coordinate
            cY = int(M["m01"] / M["m00"])  # m01/m00 gives y coordinate
            area_pixels = M["m00"]

        else:
            cX, cY, area_pixels = 0, 0, 0

        centroids.append((cX, cY))

    # Calculating distances of grains
    distances = []  # Create a list of distances

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            (x1, y1), (x2, y2) = (
                centroids[i],
                centroids[j],
            )  # Getting coordinates of two centroids

            # Calculating distances
            distance_in_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Store the distances
            distances.append(distance_in_pixels)

    return distances


############################################################
#################### Fractions processing
##############################################################


class FractionProcessor(BaseProcessor):

    def __init__(self):
        self.plotter = FractionPlotter()
        self.fractions_liberated = []
        self.fractions_non_liberated = []
        self.iron_lib_grains = []
        self.iron_non_lib_grains = []
        self.k = 7 # Degree of blurring - must be an odd number
        self.d = 11  # Degree of dilation - must be an odd number
        self.thresh = 210 # Value of intensity of pixels considered oxides (from 0 to 255)


    def _pre_process(self):
        self.images = load_images(FOLDER_FRACTION)

    def _process(self):
        """
        Perform Fractions segmentation
        """
        for image, filename in self.images:

            filtered_image = image_fractions_preproc(image, self.k)
            # Store the shape before any flattening
            og_shape = filtered_image.shape

            # Processing the original image and coloring
            sorted_indices, label = self._apply_kmeans(filtered_image)

            # Coloring the image
            segmented_image = self._color_image(
                sorted_indices, label, filtered_image, image
            )

            # Plotting for inspection
            self.plotter.colored_image(segmented_image)

            # Separating oxides
            oxide_segmented_image, ct_blue_dilated, dilated_blue_mask= self._contrours_on_oxides(
                filename, filtered_image, segmented_image
            )

            # Liberated and non liberated grain calculation
            lib_sizes_mm2, non_lib_sizes_mm2, lib_grains, non_lib_grains = (
                self._liberation_degree(segmented_image, ct_blue_dilated)
            )
            
            iron_lib_grain, iron_non_lib_grain = self._estimate_fe_mass_from_areas_2d(lib_sizes_mm2,
                                                                    non_lib_sizes_mm2)


            self.plotter.visualize_blue_mask(oxide_segmented_image, ct_blue_dilated)

            
            plot_textbox = self._variables_for_textbox(
                filtered_image,
                filename,
                label,
                dilated_blue_mask,
                lib_grains,
                non_lib_grains,
                iron_lib_grain,
                iron_non_lib_grain,
            )
            
            print('plot textbox', plot_textbox)

            self.plotter.subplots_of_grain_info(
                lib_sizes_mm2, non_lib_sizes_mm2, plot_textbox, filename
            )

            self._save_data_for_overall_plot(lib_sizes_mm2, non_lib_sizes_mm2, iron_lib_grain, iron_non_lib_grain)

    def _post_process(self):

        # Creating lists to send for the final plot
        lib_post = self._post_process_fractions_plot(
            self.fractions_liberated, 
            self.iron_lib_grains
        )

        non_lib_post = self._post_process_fractions_plot(
            self.fractions_non_liberated,
            self.iron_non_lib_grains,
        )        

        # Plotting the final plot
        self.plotter.plot_cdf(lib_post, non_lib_post)


    #####################################
    #           Functions               #
    #####################################


    def _apply_kmeans(self, filtered_image):

        # Necessary conversions to put the image in the k-means algorithm
        image_reshape = filtered_image.reshape(-1, 1)
        image_reshape_float = np.float32(image_reshape)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 2  # To separate grains from matrix initially
        attempts = 10
        ret, label, center = cv2.kmeans(
            image_reshape_float, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
        )

        # Sorting labels and centers with increasing values of brightness
        sorted_indices = np.argsort(np.sum(center, axis=1))

        return sorted_indices, label

    def _color_image(self, sorted_indices, label, filtered_image, image):

        # Directly use the original labels of the darkest clusters -> Polished section's Matrix (resin)
        darkest_cluster_label = sorted_indices[0]  # Original label of the darkest cluster

        # Reshape labels to match the image's shape for accurate masking
        labels_reshaped = label.reshape(filtered_image.shape)

        # Initialize the segmented image -> Our new image
        segmented_image = image.copy()

        # Define the colors in BGR
        dark_purple_color = (128, 0, 128)  # Dark purple for the matrix (resin)
        green_color = (0, 255, 0)  # Green for all other clusters

        # Apply green color to the entire segmented image first
        for cluster_label in np.unique(label):  # Iterate through all cluster labels
            mask = (labels_reshaped == cluster_label).astype(np.uint8)
            segmented_image[mask == 1] = green_color

        # Apply purple color to the matrix (darkest cluster)
        darkest_mask = (labels_reshaped == darkest_cluster_label).astype(np.uint8)
        segmented_image[darkest_mask == 1] = dark_purple_color

        return segmented_image

    def _contrours_on_oxides(self, filename, filtered_image, segmented_image):

        blue_color = (255, 0, 0)  # Blue for the brightest cluster

        # Targeting magnetite & drawing contours that are filled + create a mask
        ret, threshold_image = cv2.threshold(
            filtered_image, self.thresh, maxval=255, type=cv2.THRESH_BINARY
        )
        contours_blue, hierarchy = cv2.findContours(
            threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )  # Contours
        cv2.drawContours(
            segmented_image, contours_blue, -1, blue_color, thickness=cv2.FILLED
        )
        blue_mask = cv2.inRange(segmented_image, blue_color, blue_color)  # Create a binary mask for the blue-filled areas

        # Dilate Oxides
        # Create a kernel for dilation (the size of the kernel affects the extent of dilation)
        kernel = np.ones((self.d, self.d), np.uint8)  # Example kernel size, adjust as necessary
        dilated_blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)      # Apply dilation to the blue mask for expansion
        segmented_image[dilated_blue_mask == 255] = blue_color    # Wherever the dilated mask is white (255), recolor blue

         # Recompute contours for the dilated blue areas
        ct_blue_dilated, _ = cv2.findContours(dilated_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Saving
        base_filename = os.path.splitext(filename)[0]
        new_filename = f"{base_filename}_colored.png"
        save_path = os.path.join(DESTINATION_FOLDER_FRACTIONS, new_filename)
        cv2.imwrite(save_path, segmented_image)
        print(f"Image saved to {save_path}")

        return segmented_image, ct_blue_dilated, dilated_blue_mask

    def _liberation_degree(self, segmented_image, ct_blue_dilated):
        """
        Determine the liberation degree of grains identified in a segmented image.
        """

        dark_purple_color = (128, 0, 128)  # Dark purple for the matrix (resin)
        green_color = (0, 255, 0)  # Green for all other clusters

        lib_grains = 0  # Coounter of liberated grains
        non_lib_grains = 0  # Coounter of non-liberated grains
        lib_sizes_mm2 = []  # List to store sizes of liberated grains
        non_lib_sizes_mm2 = []  # List to store sizes of non-liberated grains

        for contour in ct_blue_dilated:
            edge_pixels = (
                contour.squeeze()
            )  # Remove redundant dimensions from the contour array
            dark_purple_touch = 0

            # Check each pixel on the contour's edge for contact with the matrix
            for pixel in edge_pixels:

                # Check surrounding pixels for contact with dark purple (matrix) or green (surrounding material)
                dp_count, _ = self._check_pixel_contact(
                    (pixel[1], pixel[0]),
                    segmented_image,
                    dark_purple_color,
                    green_color,
                )

                # If contact with the matrix is detected, increment the counter
                if dp_count > 0:
                    dark_purple_touch += 1

            # Calculate the liberation ratio (fraction of edge touching the matrix)
            liberation_ratio = dark_purple_touch / len(edge_pixels)

            # Conversion factor from pixels to millimeters
            pixel_to_mm = 0.002  # 1 pixel = 0.002 mm
            # Convert contour area from pixels² to µm² using the scale factor
            grain_size_pixels = cv2.contourArea(
                contour
            )  # Area enclosed by the contour in pixels²
            grain_size_mm2 = grain_size_pixels * pixel_to_mm**2

            # Classify grain as liberated or non-liberated based on the liberation ratio
            if liberation_ratio > 0.75:
                lib_grains += 1
                lib_sizes_mm2.append(grain_size_mm2)
            else:
                non_lib_grains += 1
                non_lib_sizes_mm2.append(grain_size_mm2)

        return lib_sizes_mm2, non_lib_sizes_mm2, lib_grains, non_lib_grains

    def _check_pixel_contact(
        self, pixel_position, segmented_image, dark_purple_color, green_color
    ):
        """
        Checks the immediate neighbors of a pixel for contact with specific colors.
        """

        # Define Up, down, left, right
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dark_purple_count = 0
        green_count = 0

        # Iterate through each neighbor's position
        for dx, dy in neighbors:
            nx, ny = (
                pixel_position[0] + dx,
                pixel_position[1] + dy
            )  # Calculate neighbor's absolute position

            # Ensure neighbor's position is within image boundaries to avoid indexing errors
            if 0 <= nx < segmented_image.shape[0] and 0 <= ny < segmented_image.shape[1]:
                pixel_color = segmented_image[nx, ny]  # Retrieve neighbor's color

                # Increment counts based on the color of the neighbor
                if np.all(pixel_color == dark_purple_color):
                    dark_purple_count += 1
                elif np.all(pixel_color == green_color):
                    green_count += 1

        return dark_purple_count, green_count
    
    def _estimate_fe_mass_from_areas_2d(self, lib_sizes_mm2, non_lib_sizes_mm2):
        """
        Estimate the mass of Fe in magnetite grains for both liberated and non-liberated grains based on their 2D areas.
        """

        # Calculate Fe mass for liberated and non-liberated grains
        iron_lib = self._calc_fe_mass(sum(lib_sizes_mm2))
        iron_non_lib = self._calc_fe_mass(sum(non_lib_sizes_mm2))

        return iron_lib, iron_non_lib
    
    # Helper function to calculate Fe mass from total area, assuming a single layer of grains

    def _calc_fe_mass(self, total_area_mm2, density_magnetite_g_cm3=5.17):

        # Molar masses
        molar_mass_Fe = 55.85  # g/mol for Fe
        molar_mass_Fe3O4 = 231.54  # g/mol for Fe3O4
        # Assume a standard thickness for a monolayer of magnetite grains (e.g., 1 µm = 0.001 mm)
        layer_thickness_mm = 0.001  # This is a chosen value for the calculation
        volume_magnetite_cm3 = total_area_mm2 * layer_thickness_mm / 1000  # Convert mm³ to cm³
        mass_magnetite_g = volume_magnetite_cm3 * density_magnetite_g_cm3
        mass_Fe_g = mass_magnetite_g * (3 * molar_mass_Fe / molar_mass_Fe3O4)
            
        return mass_Fe_g * 1e6  # Convert grams to micrograms


    def _variables_for_textbox(
        self,
        filtered_image,
        filename,
        label,
        dilated_blue_mask,
        lib_grains,
        non_lib_grains,
        iron_lib_grain,
        iron_non_lib_grain,
    ):

        # Reshape labels to match the image's shape for accurate masking
        labels_reshaped = label.reshape(filtered_image.shape)

        # Variables and calculations for the textbox
        grains_pixels = np.sum(labels_reshaped == 1)
        total_pixels = labels_reshaped.size
        grain_percent = grains_pixels / total_pixels * 100
        oxide_pixels = np.count_nonzero(dilated_blue_mask)
        oxide_percentage = oxide_pixels / grains_pixels * 100

        # Adding the textbox with iron content and other metrics
        textstr = (
            f"Filename: {filename}\n"
            f"Total Grain: {grain_percent:.2f}%\n"
            f"Oxide grains: {oxide_percentage:.2f}%\n"
            f"Liberated Grains of oxides: {lib_grains}\n"
            f"Non-Liberated Grains of oxides: {non_lib_grains}\n"
            f"Total Fe Content (Liberated): {iron_lib_grain:.2f}%\n"
            f"Total Fe Content (Non-Liberated): {iron_non_lib_grain:.2f}%"
        )

        return textstr


    def _save_data_for_overall_plot(self, lib_sizes_mm2, non_lib_sizes_mm2, iron_lib, iron_non_lib):
        self.fractions_liberated.append(lib_sizes_mm2)
        self.fractions_non_liberated.append(non_lib_sizes_mm2)
        self.iron_lib_grains.append(iron_lib)
        self.iron_non_lib_grains.append(iron_non_lib)


    def _post_process_fractions_plot(self, fractions, iron):
        mean_iron = 0
        if iron:    
            mean_iron = np.mean(iron)
        areas_combined = [area for sublist in fractions for area in sublist]
        areas_sorted = np.sort(areas_combined)
        areas_cdf = np.arange(1, len(areas_sorted)+1) / len(areas_sorted)
        return {
            "sorted": areas_sorted,
            "cdf": areas_cdf,
            "mean_iron": mean_iron
        }
