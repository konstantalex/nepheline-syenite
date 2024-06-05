import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from scipy.stats import gaussian_kde
import cv2

from .config import (
    DESTINATION_FOLDER_FRACTIONS,
    DESTINATION_FOLDER_SEM_PATCHED,
    parameters,
)


class BasePlotter:
    pass


class StitchedSemPlotter(BasePlotter):

    def __init__(self):
        self.fontsize_label = 14
        self.fontsize_title = 16

    def plot_kde(self, kde_curves):
        if not kde_curves:
            print("No data available to plot")
            return
        plt.figure(figsize=(10, 6))
        for data in kde_curves:
            plt.plot(
                data["dist_range"],
                data["kde_values"],
                color=data["color"],
                label=data["filename"],
            )

        plt.xlabel("Distances (mm)", fontsize=self.fontsize_label)
        plt.ylabel("Frequency", fontsize=self.fontsize_label)
        plt.title("Comparison of KDE Curves", fontsize=self.fontsize_title)
        # plt.legend()
        save_path_kde = os.path.join(
            DESTINATION_FOLDER_SEM_PATCHED, "kde_comparison_plot.png"
        )
        plt.savefig(save_path_kde)
        print(f"Plot saved to {save_path_kde}")

    def plot_grain_data(self, grain_data):
        if not grain_data:
            print("No data available to plot")
            return
        plt.figure(figsize=(10, 6))
        for data in grain_data:
            # Sort areas to ensure cumulative calculation is correct
            sorted_areas = np.sort(data["areas_mm2"])
            cumulative_areas = np.cumsum(sorted_areas)
            cumulative_percentage = (cumulative_areas / cumulative_areas[-1]) * 100

            plt.plot(
                sorted_areas,
                cumulative_percentage,
                drawstyle="steps-post",
                color=data["color"],
                label=data["filename"],
                alpha=0.5,
            )

        plt.xlabel("Grain Area (mm²)", fontsize=self.fontsize_label)
        plt.ylabel("Cumulative Area Percentage (%)", fontsize=self.fontsize_label)
        plt.title("Comparison of Oxide Grain Areas", fontsize=self.fontsize_title)
        # plt.legend()

        # Assuming destination_folder_sem_patched is defined
        save_path_grain_size = os.path.join(
            DESTINATION_FOLDER_SEM_PATCHED, "grain-area_comparison.png"
        )
        plt.savefig(save_path_grain_size)
        print(f"Plot saved to {save_path_grain_size}")

    def visualize_contours(self, image):
        resized_display_image = cv2.resize(image, (1200, 800))
        if parameters["show_images"]:
            cv2.imshow("Oxides segmentation:", resized_display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _prep_sem_image(self, sorted_labels, sorted_centers, orig_shape):
        # Reconstruct the image using sorted labels
        sorted_image = sorted_centers[sorted_labels].reshape(orig_shape).astype(np.uint8)
        # Resize the image to 1600x1200 before displaying
        resized_img = cv2.resize(sorted_image, (1200, 800))
        return resized_img

    def display_orig_and_sorted_images(self, image, labels, centers, orig_shape):
        resized_img = self._prep_sem_image(labels, centers, orig_shape)
        resized_orig_image = cv2.resize(image, (1200, 800))
        # Display the original and sorted images
        if parameters["show_images"]:
            cv2.imshow("Original Image", resized_orig_image)
            cv2.imshow("Sorted Clusters Image", resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def precompute_sem_plot_data(self, contours, distances_pixels, filename):
        color = self._extract_color(
            filename
        )  # Use your existing function to extract the color
        if color == "pink":  # Check if the color for this dataset is supposed to be pink
            color = "#C2185B"  # A darker shade of pink
        distances_mm = self._convert_pixels_to_mm(distances_pixels)

        dist_range = np.linspace(min(distances_mm), max(distances_mm), 300)
        kde = gaussian_kde(distances_mm)
        kde_values = kde(dist_range)

        # Adjustments for Plot 2 to show cumulative distribution of grain diameters
        # Applying this conversion to your area calculations
        areas_pixels = np.array([cv2.contourArea(contour) for contour in contours])
        areas_mm2 = self._convert_pixels_to_mm2(areas_pixels)

        total_area_mm2 = np.sum(areas_mm2)

        # Compute cumulative area percentage
        sorted_areas = np.sort(areas_mm2)
        cumulative_areas = np.cumsum(sorted_areas)
        cumulative_percentage = (cumulative_areas / total_area_mm2) * 100

        return {
            "distances_mm": distances_mm,
            "color": color,
            "sorted_areas": sorted_areas,
            "cumulative_percentage": cumulative_percentage,
            "dist_range": dist_range,
            "kde_values": kde_values,
            "areas_mm2": areas_mm2,
        }

    def plot_data(
        self, mineral_percentages, iron_percentage_from_oxides, filename, **kwargs
    ):

        distances_mm = kwargs["distances_mm"]
        color = kwargs["color"]
        sorted_areas = kwargs["sorted_areas"]
        cumulative_percentage = kwargs["cumulative_percentage"]
        dist_range = kwargs["dist_range"]
        kde_values = kwargs["kde_values"]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Adjust the figure size here
        # Convert distances and oxide grain sizes from pixels to mm

        # Plot 1: Histogram of distances in mm
        axs[0, 0].hist(distances_mm, bins=20, color=color)
        axs[0, 0].set_xlabel("Distances (mm)")
        axs[0, 0].set_ylabel("Frequency")

        # Adjusted Plot 2: Cumulative area percentage of oxide grains
        axs[0, 1].plot(
            sorted_areas, cumulative_percentage, drawstyle="steps-post", color=color
        )
        axs[0, 1].set_xlabel("Grain Area (mm²)")
        axs[0, 1].set_ylabel("Cumulative Area (%)")
        axs[0, 1].set_xscale("linear")
        axs[0, 1].set_yscale("linear")

        # KDE Plot for distances in mm
        axs[1, 0].plot(dist_range, kde_values, color=color, label="KDE for Distances")
        axs[1, 0].set_xlabel("Distance (mm)")
        axs[1, 0].set_ylabel("Frequency")
        axs[1, 0].legend()

        # Hide the fourth subplot
        axs[1, 1].axis("off")

        # Add a text box with mineral percentages and iron quantity
        textstr = "\n".join(
            (
                f"Filename: {filename}",
                r"Mineral Percentages:",
                rf'Nepheline: {mineral_percentages.get("nepheline", 0):.2f}%',
                rf'Feldspar: {mineral_percentages.get("feldspar", 0):.2f}%',
                rf'Pyroxene-Amphibole: {mineral_percentages.get("pyroxenes_amphibole", 0):.2f}%',
                rf'Oxides: {mineral_percentages.get("oxides", 0):.2f}%',
                "",
                rf"Iron content: {iron_percentage_from_oxides:.2f} %",
            )
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        axs[1, 1].text(
            0.95,
            0.05,
            textstr,
            transform=axs[1, 1].transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        plt.tight_layout()

        save_path_fig = os.path.join(
            DESTINATION_FOLDER_SEM_PATCHED, f"{filename}_subplots.png"
        )
        fig.savefig(save_path_fig)
        # return fig

    def _extract_color(self, filename):
        # Extract the color information from the filename
        parts = filename.split("_")
        color = parts[-1].split(".")[
            0
        ]  # Extract the last part of the filename without extension
        known_colors = [
            "orange",
            "blue",
            "green",
            "pink",
            "black",
        ]  # List of known colors

        if color.lower() in known_colors:
            return color.lower()  # If it's a known color, return it
        else:
            return "black"  # If it's not a known color, return 'black'

    def _convert_pixels_to_mm(self, pixels):
        """Converts pixels to millimeters (mm) with 1 μm = 250 pixels."""
        pixels_per_mm = 250 * 1000  # 250 pixels per μm, and 1000 μm in 1 mm
        return [pixel / pixels_per_mm for pixel in pixels]

    def _convert_pixels_to_mm2(self, pixels_area):
        """Converts pixel areas to millimeters squared (mm²), considering the square of the conversion factor."""
        pixels_per_mm2 = (250 * 1000) ** 2  # Square the conversion factor for area
        return pixels_area / pixels_per_mm2


########################################################################
# Fraction Analysis
########################################################################


class FractionPlotter(BasePlotter):

    def __init__(self):
        self.fontsize_label = 14
        self.fontsize_title = 16

    def plot_cdf(self, lib_post, non_lib_post):
        # Plotting the CDF
        plt.figure(figsize=(12, 6))
        plt.plot(
            lib_post["sorted"],
            lib_post["cdf"],
            label="Liberated Grains",
            color="blue",
        )
        plt.plot(
            non_lib_post["sorted"],
            non_lib_post["cdf"],
            label="Non-Liberated Grains",
            color="red",
        )
        plt.xlabel("Grain Area (mm²)", fontsize=self.fontsize_label)
        plt.ylabel("CDF (Cumulative Distribution Function)", fontsize=self.fontsize_label)
        plt.title("Cumulative Distribution of Iron Oxide Grain Sizes", fontsize=self.fontsize_title)
        plt.legend(loc = 'upper left')
        plt.grid(True)

        # Define the text string with the mean iron quantities
        textstr = "\n".join(
            (
                f'Mean Iron Quantity (Liberated): {lib_post["mean_iron"]:.2f}%',
                f'Mean Iron Quantity (Non-Liberated): {non_lib_post["mean_iron"]:.2f}%',
            )
        )

        # Position the text box in bottom right in axes coords
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        plt.text(
            0.95,
            0.05,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="right",
            bbox=props,
        )

        # Save the plot before displaying
        save_path = os.path.join(
            DESTINATION_FOLDER_FRACTIONS, "grain_size_cumulative_distribution.png"
        )
        plt.savefig(save_path)

        print(f"Plot saved to {save_path}")

    def visualize_blue_mask(self, segmented_image, ct_blue_dilated):

        blue_color = (255, 0, 0)  # Blue for the brightest cluster

        # Drawing contours
        for contour in ct_blue_dilated:
            cv2.drawContours(
                segmented_image, [contour], -1, blue_color, thickness=cv2.FILLED
            )

        # Display the result
        if parameters["show_images"]:
            resized_segmented_image = cv2.resize(segmented_image, (1200, 1200))
            cv2.imshow(
                "Segmented Image with Highlighted Clusters", resized_segmented_image
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def colored_image(self,segmented_image):

        # Display the result
        if parameters["show_images"]:
            resized_segmented_image = cv2.resize(segmented_image, (1200, 1200))
            cv2.imshow(
                "Segmented Image with Highlighted Clusters", resized_segmented_image
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def subplots_of_grain_info(self, lib_sizes_mm2, non_lib_sizes_mm2, plot_textbox, filename):

        # Calculate equivalent diameters for the liberated and non-liberated grain sizes
        lib_diameters_mm = [2 * np.sqrt(size / np.pi) for size in lib_sizes_mm2]
        non_lib_diameters_mm = [2 * np.sqrt(size / np.pi) for size in non_lib_sizes_mm2]

        if parameters["show_images"]:
            plt.figure(figsize=(20, 8))  # Adjust figure size for subplots

            # First subplot for diameters
            plt.subplot(1, 2, 1)

            plt.hist(
                lib_diameters_mm,
                alpha=0.5,
                label="Liberated Grains",
                color="blue",
                bins=20,
                edgecolor="black",
            )
            plt.hist(
                non_lib_diameters_mm,
                alpha=0.5,
                label="Non-Liberated Grains",
                color="red",
                bins=20,
                edgecolor="black",
            )
            plt.xlabel("Grain Diameter (mm)", fontsize=self.fontsize_label)
            plt.ylabel("Frequency", fontsize=self.fontsize_label)
            plt.title("Frequency over Iron Oxide Grain Diameter", fontsize=self.fontsize_title)
            plt.legend(loc="upper right")

            # Second subplot: Cumulative Distribution of Grain Areas - Separate lines for liberated and non-liberated
            plt.subplot(1, 2, 2)

            # Calculate and plot for liberated grains
            sorted_liberated_areas = np.sort(lib_sizes_mm2)
            cumulative_percent_liberated = (
                np.cumsum(sorted_liberated_areas) / sum(lib_sizes_mm2) * 100
            )
            plt.plot(
                sorted_liberated_areas,
                cumulative_percent_liberated,
                label="Liberated Grains",
                color="blue",
                drawstyle="steps-post",
            )

            # Calculate and plot for non-liberated grains
            sorted_non_liberated_areas = np.sort(non_lib_sizes_mm2)
            cumulative_percent_non_liberated = (
                np.cumsum(sorted_non_liberated_areas) / sum(non_lib_sizes_mm2) * 100
            )
            plt.plot(
                sorted_non_liberated_areas,
                cumulative_percent_non_liberated,
                label="Non-Liberated Grains",
                color="red",
                drawstyle="steps-post",
            )

            plt.xlabel("Grain Area (mm²)", fontsize=self.fontsize_label)
            plt.ylabel("Cumulative Percentage", fontsize=self.fontsize_label)
            plt.title("Cumulative Distribution of Iron Oxide Grain Areas", fontsize=self.fontsize_title)
            plt.legend(loc="upper left")

            anchored_text = AnchoredText(plot_textbox, loc="lower right", frameon=True)
            anchored_text.patch.set_boxstyle("round,pad=0.5,rounding_size=0.2")
            plt.gca().add_artist(anchored_text)

            plt.tight_layout()

            # Saving the figure
            base_filename = os.path.splitext(filename)[0]
            new_graph_filename = f"{base_filename}_graphs.png"
            save_path_graph = os.path.join(
                DESTINATION_FOLDER_FRACTIONS, new_graph_filename
            )
            plt.savefig(save_path_graph, dpi=300)
            print(f"Image saved to {save_path_graph}")
            plt.show

            return
