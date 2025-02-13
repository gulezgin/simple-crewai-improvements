from setuptools import setup, find_packages

setup(
    name="agriculture-source-selector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "crewai>=0.13.0",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "pysbd>=0.3.4",
    ],
)