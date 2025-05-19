import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import logging
import os

from movie_recommender.utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)


class RecommendationVisualizer:
    """
    Class for visualizing recommendation system results and evaluations.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save visualization outputs. If None, figures will not be saved.
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    
    def _save_figure(self, fig, filename):
        """
        Save a figure to the output directory.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filename : str
            Filename for the saved figure.
        """
        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
    
    @log_execution_time
    def plot_rating_distribution(self, ratings_df, title="Rating Distribution", save_as=None):
        """
        Plot the distribution of ratings.
        
        Parameters
        ----------
        ratings_df : pandas.DataFrame
            DataFrame containing ratings data with a 'rating' column.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting rating distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot rating distribution
        sns.countplot(x='rating', data=ratings_df, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        
        # Add count labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_prediction_vs_actual(self, y_true, y_pred, title="Predicted vs Actual Ratings", save_as=None):
        """
        Plot predicted ratings against actual ratings.
        
        Parameters
        ----------
        y_true : array-like
            Actual ratings.
        y_pred : array-like
            Predicted ratings.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting predicted vs actual ratings")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(title)
        ax.set_xlabel('Actual Rating')
        ax.set_ylabel('Predicted Rating')
        ax.grid(True)
        
        # Add correlation coefficient
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_error_distribution(self, y_true, y_pred, title="Error Distribution", save_as=None):
        """
        Plot the distribution of prediction errors.
        
        Parameters
        ----------
        y_true : array-like
            Actual ratings.
        y_pred : array-like
            Predicted ratings.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting error distribution")
        
        # Calculate errors
        errors = np.array(y_pred) - np.array(y_true)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot error distribution
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Prediction Error (Predicted - Actual)')
        ax.set_ylabel('Count')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--')
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax.text(0.05, 0.95, f'Mean Error: {mean_error:.3f}\nStd Dev: {std_error:.3f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_rating_heatmap(self, user_item_matrix, title="User-Item Rating Heatmap", save_as=None):
        """
        Plot a heatmap of user-item ratings.
        
        Parameters
        ----------
        user_item_matrix : pandas.DataFrame
            User-item rating matrix.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting user-item rating heatmap")
        
        # Limit the size of the heatmap to avoid memory issues
        max_users = 50
        max_items = 50
        
        if user_item_matrix.shape[0] > max_users or user_item_matrix.shape[1] > max_items:
            logger.info(f"Limiting heatmap to {max_users} users and {max_items} items")
            matrix_subset = user_item_matrix.iloc[:max_users, :max_items]
        else:
            matrix_subset = user_item_matrix
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(matrix_subset, cmap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Item ID')
        ax.set_ylabel('User ID')
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_recommendation_metrics(self, metrics_dict, title="Recommendation Metrics", save_as=None):
        """
        Plot recommendation system evaluation metrics.
        
        Parameters
        ----------
        metrics_dict : dict
            Dictionary containing metric names and values.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting recommendation metrics")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract metric names and values
        metric_names = list(metrics_dict.keys())
        metric_values = list(metrics_dict.values())
        
        # Plot bar chart
        bars = ax.bar(metric_names, metric_values)
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.set_ylim(0, max(metric_values) * 1.2)  # Add some space above the highest bar
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_precision_recall_curve(self, y_true, y_pred_proba, threshold=3.5, title="Precision-Recall Curve", save_as=None):
        """
        Plot precision-recall curve for recommendation system.
        
        Parameters
        ----------
        y_true : array-like
            Actual ratings.
        y_pred_proba : array-like
            Predicted ratings or probabilities.
        threshold : float, optional
            Threshold to convert ratings to binary (liked/not liked). Default is 3.5.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting precision-recall curve")
        
        # Convert ratings to binary (liked/not liked)
        y_true_binary = (np.array(y_true) >= threshold).astype(int)
        
        # Calculate precision and recall at different thresholds
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_proba)
        
        # Calculate area under the precision-recall curve
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot precision-recall curve
        ax.plot(recall, precision, lw=2)
        ax.set_title(title)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.grid(True)
        
        # Add AUC value
        ax.text(0.05, 0.05, f'AUC: {pr_auc:.3f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_roc_curve(self, y_true, y_pred_proba, threshold=3.5, title="ROC Curve", save_as=None):
        """
        Plot ROC curve for recommendation system.
        
        Parameters
        ----------
        y_true : array-like
            Actual ratings.
        y_pred_proba : array-like
            Predicted ratings or probabilities.
        threshold : float, optional
            Threshold to convert ratings to binary (liked/not liked). Default is 3.5.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting ROC curve")
        
        # Convert ratings to binary (liked/not liked)
        y_true_binary = (np.array(y_true) >= threshold).astype(int)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba)
        
        # Calculate area under the ROC curve
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], 'k--', lw=2)  # Random guess line
        ax.set_title(title)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True)
        
        # Add AUC value
        ax.text(0.05, 0.95, f'AUC: {roc_auc:.3f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_top_recommendations(self, recommendations_df, title="Top Recommendations", max_items=10, save_as=None):
        """
        Plot top recommended items for a user.
        
        Parameters
        ----------
        recommendations_df : pandas.DataFrame
            DataFrame containing recommended items with predicted ratings.
        title : str, optional
            Title for the plot.
        max_items : int, optional
            Maximum number of items to display. Default is 10.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting top recommendations")
        
        # Limit the number of items to display
        if len(recommendations_df) > max_items:
            plot_df = recommendations_df.head(max_items)
        else:
            plot_df = recommendations_df
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        bars = ax.barh(plot_df['title'], plot_df['predicted_rating'])
        ax.set_title(title)
        ax.set_xlabel('Predicted Rating')
        ax.set_xlim(0, 5.5)  # Rating scale from 0 to 5
        
        # Add value labels to the right of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}',
                    ha='left', va='center')
        
        # Reverse y-axis to show highest rated items at the top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_genre_distribution(self, movies_df, title="Genre Distribution", save_as=None):
        """
        Plot the distribution of movie genres.
        
        Parameters
        ----------
        movies_df : pandas.DataFrame
            DataFrame containing movie information with a 'genres' column.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting genre distribution")
        
        # Extract all genres
        all_genres = []
        for genres in movies_df['genres']:
            if isinstance(genres, str):
                all_genres.extend(genres.split('|'))
        
        # Count genre occurrences
        genre_counts = pd.Series(all_genres).value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot bar chart for top genres
        top_genres = genre_counts.head(15)
        bars = ax.bar(top_genres.index, top_genres.values)
        ax.set_title(title)
        ax.set_xlabel('Genre')
        ax.set_ylabel('Count')
        ax.set_xticklabels(top_genres.index, rotation=45, ha='right')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_user_activity(self, ratings_df, title="User Activity Distribution", save_as=None):
        """
        Plot the distribution of user activity (number of ratings per user).
        
        Parameters
        ----------
        ratings_df : pandas.DataFrame
            DataFrame containing ratings data with a 'userId' column.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting user activity distribution")
        
        # Count ratings per user
        user_activity = ratings_df['userId'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of user activity
        sns.histplot(user_activity, bins=50, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Number of Ratings per User')
        ax.set_ylabel('Count')
        ax.set_xscale('log')  # Use log scale for better visualization
        
        # Add statistics
        mean_activity = user_activity.mean()
        median_activity = user_activity.median()
        ax.text(0.05, 0.95, f'Mean: {mean_activity:.1f}\nMedian: {median_activity:.1f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_confusion_matrix(self, y_true, y_pred, threshold=3.5, title="Confusion Matrix", save_as=None):
        """
        Plot confusion matrix for recommendation system.
        
        Parameters
        ----------
        y_true : array-like
            Actual ratings.
        y_pred : array-like
            Predicted ratings.
        threshold : float, optional
            Threshold to convert ratings to binary (liked/not liked). Default is 3.5.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting confusion matrix")
        
        # Convert ratings to binary (liked/not liked)
        y_true_binary = (np.array(y_true) >= threshold).astype(int)
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['Not Liked', 'Liked'])
        ax.set_yticklabels(['Not Liked', 'Liked'])
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    @log_execution_time
    def plot_learning_curve(self, train_sizes, train_scores, val_scores, title="Learning Curve", save_as=None):
        """
        Plot learning curve showing model performance as a function of training set size.
        
        Parameters
        ----------
        train_sizes : array-like
            Training set sizes.
        train_scores : array-like
            Training scores for each training size.
        val_scores : array-like
            Validation scores for each training size.
        title : str, optional
            Title for the plot.
        save_as : str, optional
            Filename to save the figure. If None, the figure will not be saved.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        logger.info("Plotting learning curve")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and std for train and validation scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        # Plot learning curve
        ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='r')
        
        ax.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation score')
        ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color='g')
        
        ax.set_title(title)
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig