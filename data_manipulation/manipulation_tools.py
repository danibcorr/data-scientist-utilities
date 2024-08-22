# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import ipywidgets as widgets

# --------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# --------------------------------------------------------------------------------------------

def balance_dataset(X: np.ndarray, y: np.ndarray) -> tuple:

    """
    Balances the dataset by undersampling it to match the class with the minimum number of samples.

    Args:
    	X (numpy.ndarray): Input data, where each row is a sample and each column is a feature.
    	y (numpy.ndarray): Labels corresponding to the input data.

    Returns:
    	X_balanced (numpy.ndarray): Balanced input data.
    	y_balanced (numpy.ndarray): Labels corresponding to the balanced input data.
    """

    # Compute the number of samples in each class
    classes, counts = np.unique(y, return_counts = True)

    # Identify the minimum number of samples across all classes
    min_samples = np.min(counts)

    # For each class, randomly select samples to match the minimum number of samples
    sample_indexes = np.concatenate([np.random.choice(np.where(y == label)[0], min_samples, replace = False) for label in classes])

    # Return the balanced dataset
    return X[sample_indexes].astype('float32'), y[sample_indexes]

def manual_relabeling_images(X: np.ndarray, y: np.ndarray, start_index: int = 0, num_rows: int = 5, num_columns: int = 5) -> None:

    """
    This function allows for manual relabeling of images.

    Args:
    	X (numpy.ndarray): The images to be relabeled.
    	y (numpy.ndarray): The current labels of the images.
    """

    # Replace these with your actual images
    images = X  

    # Replace these with your actual labels
    labels = y  

    # List of unique labels in the dataset
    unique_labels = np.unique(labels)

    def display_images(start_index = 0, num_rows = 5, num_columns = 5):

        """
        This function displays a grid of images and allows for their labels to be changed.

        Args:
        	start_index (int): The index to start displaying the images from.
        	num_rows (int): The number of rows in the grid.
        	num_columns (int): The number of columns in the grid.
        """

        for i in range(num_rows):

            fig, axs = plt.subplots(1, num_columns, figsize = (20, 20))
            widgets_list = []

            for j in range(num_columns):

                index = start_index + i * num_columns + j

                if index < len(images):

                    img = axs[j].imshow(images[index], vmin = 0, vmax = 1)
                    axs[j].axis('off')

                    # Display the index of the image
                    axs[j].set_title(f'Index: {index}', fontsize = 10, pad = 2) 

                    # Add interactive dropdown to change the label
                    rect = widgets.Dropdown(
                        options=unique_labels,
                        value=labels[index],
                        description=f'Index {index}:',
                        disabled=False,
                    )

                    def on_rect_change(change, index = index):

                        labels[index] = change['new']

                    rect.observe(on_rect_change, names = 'value')
                    
                    # Add the dropdown to the list
                    widgets_list.append(rect)

            # Adjust the spacing between the subplots
            plt.tight_layout()  
            plt.show()

            # Display the label selection dropdowns next to each other
            display(widgets.HBox(widgets_list))

        # Display the label selection dropdowns next to the images
        display_widgets(start_index, num_rows, num_columns)


    def on_next_button_clicked(b, num_rows: int = 5, num_columns: int = 5) -> None:

        """
        This function is called when the 'Next' button is clicked. It updates the start index and refreshes the image display.

        Args:
        	b (Button): The button that was clicked.
        	num_rows (int): The number of rows in the grid.
        	num_columns (int): The number of columns in the grid.
        """

        nonlocal start_index
        start_index += (num_rows * num_columns)
        clear_output(wait = True)
        display_images(start_index)
        

    def on_previous_button_clicked(b, num_rows: int = 5, num_columns: int = 5) -> None:

        """
        This function is called when the 'Previous' button is clicked. It updates the start index and refreshes the image display.

        Args:
        	b (Button): The button that was clicked.
        	num_rows (int): The number of rows in the grid.
        	num_columns (int): The number of columns in the grid.
        """

        nonlocal start_index
        start_index -= (num_rows * num_columns)

        if start_index < 0:

            start_index = 0

        clear_output(wait = True)
        display_images(start_index)

    def on_apply_button_clicked(b):

        """
        This function is called when the 'Apply Changes' button is clicked. It refreshes the image display.

        Args:
        	b (Button): The button that was clicked.
        """

        # Refresh the image display
        clear_output(wait = True)
        display_images(start_index)

    def display_widgets(start_index: int, num_rows: int, num_columns: int) -> int:

        """
        This function displays the navigation and apply changes buttons.

        Args:
        	start_index (int): The index to start displaying the images from.
        	num_rows (int): The number of rows in the grid.
        	num_columns (int): The number of columns in the grid.
        """

        previous_button = widgets.Button(description = "Previous")
        previous_button.on_click(lambda b: on_previous_button_clicked(b, num_rows, num_columns))

        next_button = widgets.Button(description = "Next")
        next_button.on_click(lambda b: on_next_button_clicked(b, num_rows, num_columns))

        apply_button = widgets.Button(description = "Apply Changes")
        apply_button.on_click(on_apply_button_clicked)

        display(previous_button)
        display(next_button)
        display(apply_button)

    # Display the images
    display_images(start_index, start_index, num_rows, num_columns)