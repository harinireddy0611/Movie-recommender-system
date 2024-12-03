
# Movie Recommender System

The Movie Recommender System is a machine learning model that recommends movies based on the similarity of the movie title entered by the user. It uses **cosine similarity** to calculate the similarity between movie names and recommend similar movies from a dataset. The dataset used for this project is the **IMDb Dataset of Top 1000 Movies and TV Shows** from Kaggle.

## Dataset

The recommender system uses the **IMDb Dataset of Top 1000 Movies and TV Shows** available on Kaggle. You can access the dataset [here](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows).

## Key Features

- **Cosine Similarity-based Recommendations**: Recommends movies similar to the one entered by the user, based on the similarity of movie titles.
- **Data Preprocessing**: The dataset is cleaned and preprocessed before applying the machine learning model.
- **Machine Learning Model**: Uses a cosine similarity approach to compare movie titles and recommend the most similar ones.

## Tech Stack

- **Machine Learning**: Python (Jupyter Notebook, scikit-learn, pandas, numpy)
- **Dataset**: IMDb Dataset of Top 1000 Movies and TV Shows from Kaggle

## Repository

You can access the project's repository [here](https://github.com/yourusername/movie-recommender-system).

## File Structure

The project contains the following files:

- `moviere.ipynb`: The Jupyter Notebook where the machine learning model is implemented. This file contains all the code for data preprocessing, model training, and generating movie recommendations based on the cosine similarity.
- `movies.pkl`: A serialized version of the movie recommendation model. It stores the trained cosine similarity model that can be loaded for predictions.
- `movies_dict.pkl`: A pickled dictionary that contains the movie data used for recommendations.
- `tmdb_5000_credits (1).csv`: A CSV file containing the credits information of movies, including cast and crew data.
- `tmdb_5000_movies.csv`: A CSV file containing movie metadata such as titles, genres, release dates, etc.
- `.gitignore`: A file that ensures certain files (like temporary or large data files) are not tracked by Git.

## Installation Guide

To run the Movie Recommender System and train the model locally, follow these steps:

### Prerequisites

- **Python 3.x** installed on your system.
- **Jupyter Notebook** or **Jupyter Lab** installed for running the notebook.
- **Pip** for installing Python libraries.

### Step-by-Step Installation

1. **Clone the Repository**

   First, clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/yourusername/movie-recommender-system.git
   ```

2. **Navigate to the Project Directory**

   Go to the project directory:

   ```bash
   cd movie-recommender-system
   ```

3. **Install Dependencies**

   Install the necessary Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` file, you can manually install the following libraries:

   ```bash
   pip install pandas scikit-learn numpy
   ```

4. **Open the Jupyter Notebook**

   Open the **moviere.ipynb** Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   This will open Jupyter Notebook in your browser.

5. **Run the Notebook**

   Run the notebook to execute the file and start the model to obtain recommendations.
6. **Using the Model**

   After training the model, you can use it to recommend movies based on user input (movie name) in the notebook.

### File Structure Overview

- `moviere.ipynb`: Jupyter Notebook that includes data preprocessing, model training, and recommendation logic.
- `movies.pkl`: The saved cosine similarity model, which can be loaded to make predictions.
- `movies_dict.pkl`: Pickled dictionary containing movie data used for recommendations.
- `tmdb_5000_credits (1).csv`: A dataset containing cast and crew information.
- `tmdb_5000_movies.csv`: A dataset containing metadata for the top 5000 movies.
- `.gitignore`: Ensures files like `.DS_Store` or temporary files are ignored by Git.

## Contributing

If you'd like to contribute to the Movie Recommender System, feel free to fork the repository and submit pull requests. Please ensure that your code adheres to the existing coding standards and includes tests where appropriate.

