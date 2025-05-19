# Movie Recommendation System Backend

This is the .NET Web API backend for the Movie Recommendation System that integrates with a neuro-fuzzy model for generating personalized movie recommendations.

## Architecture

The backend is built using ASP.NET Core Web API and integrates with a Python-based neuro-fuzzy recommendation model. The architecture consists of:

- **Controllers**: Handle HTTP requests and responses
- **Services**: Implement business logic and model integration
- **Models**: Define data structures
- **Python Integration**: Connect to the Python-based neuro-fuzzy model

## Setup

### Prerequisites

- .NET 8.0 SDK
- Python 3.9 with TensorFlow, NumPy, Pandas, and scikit-fuzzy installed
- Set the `PYTHONHOME` environment variable to your Python installation directory

### Configuration

Update the `appsettings.Development.json` file with your Python installation path:

```json
"PythonSettings": {
  "PythonHome": "C:\Python39"
}
```

## API Endpoints

### Get All Movies

```
GET /api/MovieRecommendation/movies
```

Returns a list of all movies in the dataset.

### Get Movie by ID

```
GET /api/MovieRecommendation/movies/{id}
```

Returns details for a specific movie.

### Get Recommendations

```
POST /api/MovieRecommendation/recommend
```

Request body:

```json
{
  "userId": 1,
  "count": 10
}
```

Returns personalized movie recommendations for the specified user.

### Add Rating

```
POST /api/MovieRecommendation/rating
```

Request body:

```json
{
  "userId": 1,
  "movieId": 123,
  "rating": 4.5
}
```

Adds a new rating for a movie.

## Integration with Neuro-Fuzzy Model

The backend integrates with the Python-based neuro-fuzzy model using Python.NET. The integration flow is:

1. The `MovieRecommendationController` receives API requests
2. The `MovieRecommendationService` processes these requests
3. For recommendations, it calls the `PythonModelService`
4. The `PythonModelService` uses Python.NET to execute the Python prediction script
5. The Python script loads the trained neuro-fuzzy model and generates recommendations

## Running the Application

```
dotnet run
```

The API will be available at:

- https://localhost:7xxx/swagger (where xxx is the port assigned by ASP.NET)
- http://localhost:5xxx/swagger (where xxx is the port assigned by ASP.NET)
