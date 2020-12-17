from setuptools import setup, find_packages
import pkg_resources

# The below line is used by GH Actions and should not be modified or moved above or below
version = '0.0.0'

readme_path = pkg_resources.resource_filename(__name__, "README.md")
with open(readme_path, encoding="utf8") as f:
    long_description = f.read()


setup(
    name="tfrec",
    version=version,
    description="A recommender library built on top of Tensorflow and \
    Keras with implementations of SVD and SVD++ algorithms.",
    packages=find_packages(),
    url="https://github.com/Praful932/tf-rec",
    author="Praful Mohanan",
    author_email="praful.mohanan@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=[
        "tensorflow>=2.0",
        "numpy>=1.18.5",
        "pandas>=1.1.5",
        "scikit-learn>=0.22.2",
        "requests",
    ],
)
