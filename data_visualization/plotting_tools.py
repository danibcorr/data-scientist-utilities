# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import os
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# --------------------------------------------------------------------------------------------


def show_class_frequency(data, x_label = "Type of pattern", y_label = "Frequency", title_name = None, save_fig = False, path = './Images', file_name = "freq.png"):

    """
    This function displays a bar chart of the frequency of classes in the given data.

    Parameters:
    - data (pandas.Series or pandas.DataFrame): Input data to visualize. Expected to be a series or single-column dataframe.
    - x_label (str, optional): Label for the x-axis. Defaults to "Type of pattern".
    - y_label (str, optional): Label for the y-axis. Defaults to "Frequency".
    - title_name (str, optional): Title for the plot. If None, no title is set. Defaults to None.
    - save_fig (bool, optional): If True, the figure is saved to the specified path. Defaults to False.
    - path (str, optional): Path where the figure is saved if save_fig is True. Defaults to './Images'.
    - file_name (str, optional): Name of the file to save the figure as if save_fig is True. Defaults to "freq.png".
    """

    # Check if the specified path exists, if not, create it
    if not os.path.exists(path):

        os.makedirs(path)

    # Create a new figure with specified size
    plt.figure(figsize=(10, 6))

    # Create a bar plot of the data
    ax = data.plot(kind='bar')

    # Set the title of the plot if a title name is provided
    if title_name is not None:

        plt.title(title_name)

    # Set the labels of the x and y axes
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Annotate the height of each bar in the plot
    for p in ax.patches:

        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), rotation=45)

    # Adjust the layout for better visualization
    plt.tight_layout()

    # Save the figure to the specified path if save_fig is True
    if save_fig:

        plt.savefig(os.path.join(path, file_name))

    # Display the plot
    plt.show()


def show_history(history, model_name, plot_accuracy = False, save_fig = False, path = './Images', file_name = "history.png"):

    """
    This function plots the training history of a model. It displays the loss history and optionally the accuracy history.

    Parameters:
    - history (History): Training history of the model. This is typically the output of the `fit` method of a Keras model.
    - model_name (str): Name of the model. This is used in the title of the plots and in the names of the saved plot images.
    - plot_accuracy (bool, optional): Whether to plot accuracy as well. If True, a second plot showing the accuracy history is displayed. Defaults to False.
    - save_fig (bool, optional): If True, the figure is saved to the specified path. Defaults to False.
    - path (str, optional): Path where the plot images are saved. The directory is created if it does not exist. Defaults to './Images'.
    - file_name (str, optional): Name of the file to save the figure as if save_fig is True. Defaults to "history.png".
    """

    # Check if the specified path exists, if not, create it
    if not os.path.exists(path):

        os.makedirs(path)

    # Create a plot of the loss history
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(True)
    plt.title(f'Loss of the model {model_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper right')

    # If save_fig is True, save the figure to the specified path
    if save_fig:

        plt.savefig(os.path.join(path, f'loss_{file_name}'))

    plt.show()

    # If plot_accuracy is True, create a plot of the accuracy history
    if plot_accuracy:

        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.grid(True)
        plt.title(f'Accuracy of the model {model_name}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper right')

        # If save_fig is True, save the figure to the specified path
        if save_fig:
            
            plt.savefig(os.path.join(path, f'accuracy_{file_name}'))

        plt.show()
