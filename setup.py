import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wkskel",
    version="0.0.2",
    author="Florian Drawitsch",
    author_email="florian.drawitsch@brain.mpg.de",
    description="A library for scientific analysis and manipulation of webKnossos skeleton tracings",
    long_description_content_type="text/markdown",
    url="https://gitlab.mpcdf.mpg.de/connectomics/wkskel",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.1',
        'pandas>=0.25',
        'matplotlib>=3.1',
        'networkx>=2.3',
        'wknml>=0.0.9'
    ]
)
