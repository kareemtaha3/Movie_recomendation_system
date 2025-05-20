from setuptools import setup, find_packages

setup(
    name="movie_recommender",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "psutil",
    ],
    python_requires=">=3.8",
) 