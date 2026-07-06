from setuptools import setup, find_packages


def get_requirements(file_path):
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="multi_series_forecasting",
    version="1.0.0",
    author="Aman Sain",
    author_email="your_email@example.com",  # Optional: apna email likh sakte ho
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)