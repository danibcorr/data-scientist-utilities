# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import pandas as pd

# --------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# --------------------------------------------------------------------------------------------

def show_class_frequency(data: pd.Series or pd.DataFrame, x_label: str = "Type of pattern", y_label: str = "Frequency", title_name: str = None,
                         save_fig: bool = False, path: str = './Images', file_name: str = "freq.png")-> None:

    """
    This function displays a bar chart of the frequency of classes in the given data.

    Args:
    	data (pandas.Series or pandas.DataFrame): Input data to visualize. Expected to be a series or single-column dataframe.
    	x_label (str, optional): Label for the x-axis. Defaults to "Type of pattern".
    	y_label (str, optional): Label for the y-axis. Defaults to "Frequency".
    	title_name (str, optional): Title for the plot. If None, no title is set. Defaults to None.
    	save_fig (bool, optional): If True, the figure is saved to the specified path. Defaults to False.
    	path (str, optional): Path where the figure is saved if save_fig is True. Defaults to './Images'.
    	file_name (str, optional): Name of the file to save the figure as if save_fig is True. Defaults to "freq.png".
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

def show_history_model(history, model_name: str, plot_accuracy: bool = False, save_fig: bool = False, path: str = './Images', file_name: str = "history.png"):

    """
    This function plots the training history of a model. It displays the loss history and optionally the accuracy history.

    Args:
    	history (History): Training history of the model. This is typically the output of the `fit` method of a Keras model.
    	model_name (str): Name of the model. This is used in the title of the plots and in the names of the saved plot images.
    	plot_accuracy (bool, optional): Whether to plot accuracy as well. If True, a second plot showing the accuracy history is displayed. Defaults to False.
    	save_fig (bool, optional): If True, the figure is saved to the specified path. Defaults to False.
    	path (str, optional): Path where the plot images are saved. The directory is created if it does not exist. Defaults to './Images'.
    	file_name (str, optional): Name of the file to save the figure as if save_fig is True. Defaults to "history.png".
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

def show_tsne_2d(data: np.ndarray, labels: np.ndarray, tsne_perplexity: int = 40, tsne_niter: int = 1000, tsne_random_state: int = 42,
                 save_fig: bool = False, path: str = './Images', file_name: str = "tsne.png"):

    """
    This function performs t-SNE (t-Distributed Stochastic Neighbor Embedding) on the input data and visualizes the results in a 2D scatter plot.

    Args:
    	data (numpy.ndarray): The input data to perform t-SNE on.
    	labels (numpy.ndarray): The labels corresponding to the input data.
    	tsne_perplexity (int, optional): The perplexity parameter for the t-SNE algorithm. Defaults to 40.
    	tsne_niter (int, optional): The number of iterations for the t-SNE algorithm. Defaults to 1000.
    	tsne_random_state (int, optional): The random state for the t-SNE algorithm. Defaults to 42.

    Returns:
    	numpy.ndarray: The 2D representation of the data after t-SNE.
    """

    # Perform t-SNE on the data
    tsne = TSNE(n_components=2, verbose=1, perplexity=tsne_perplexity, n_iter=tsne_niter, random_state=tsne_random_state)
    tsne_results_2D = tsne.fit_transform(data)

    # Create a DataFrame to hold the t-SNE results and labels
    df_show = pd.DataFrame()
    df_show['T-SNE 2D First Component'] = tsne_results_2D[:, 0]
    df_show['T-SNE 2D Second Component'] = tsne_results_2D[:, 1]
    df_show['labels'] = labels

    # Create a scatter plot of the t-SNE results
    plt.figure(figsize=(16, 10))
    ax = sns.scatterplot(
        x="T-SNE 2D First Component", y="T-SNE 2D Second Component",
        hue="labels",
        style=df_show['labels'],
        palette=sns.color_palette("hls", len(df_show['labels'].factorize()[1])),
        data=df_show,
        legend="full",
        alpha=1
    )

    # Set the title and labels of the plot
    plt.title("2D T-SNE Representation", fontweight='bold')
    plt.xlabel('T-SNE 2D First Component', fontweight='bold')
    plt.ylabel('T-SNE 2D Second Component', fontweight='bold')
    plt.grid(True)

    if save_fig:
            
        plt.savefig(os.path.join(path, file_name))

    # Return the t-SNE results
    return tsne_results_2

def show_confusion_matrix(y_test: pd.Series, y_pred: pd.Series, figsize: tuple = (10, 8)) -> None:

    """
    Displays a confusion matrix heatmap for the given true and predicted labels.
    
    Args:
    	y_test (pd.Series): Actual labels of the test data.
    	y_pred (pd.Series): Predicted labels by the model.
    	figsize (tuple, optional): Figure size of the plot. Default is (10, 8).
    
    Returns:
    	None: The function displays the plot and does not return any value.
    """

    plt.figure(figsize=figsize)
    
    # Generate the confusion matrix and display it as a heatmap
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', fmt='g')
    
    # Set plot title and labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    
    # Display the plot
    plt.show()

def show_roc_curve(y_test: pd.Series, y_pred: pd.Series, figsize: tuple = (10, 8)) -> None:

    """
    Displays the ROC curve for the given true and predicted labels.
    
    Args:
    	y_test (pd.Series): Actual labels of the test data.
    	y_pred (pd.Series): Predicted probabilities or scores by the model.
    	figsize (tuple, optional): Figure size of the plot. Default is (10, 8).
    
    Returns:
    	None: The function displays the plot and does not return any value.
    """

    plt.figure(figsize=figsize)
    
    # Compute the false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    
    # Plot the diagonal line representing a random classifier
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    
    # Plot the ROC curve for the classifier
    plt.plot(fpr, tpr, label='ROC Curve', color='blue')
    
    # Set plot title and labels
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # Add a legend to the plot
    plt.legend(loc='lower right')
    
    # Display the plot
    plt.show()