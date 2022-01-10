import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="omnihub",
    version="0.0.1",
    author="Adam Gibson (agibsonccc)",
    author_email="adam@konduit.ai",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eclipse/deeplearning4j/contrib/modelhub",
    project_urls={
        "Bug Tracker": "https://github.com/eclipse/deeplearning4j/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"
)