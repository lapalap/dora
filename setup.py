import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="dora",
    version="0.0.0",
    author="Mayukh Deb, Kiril Bykov",
    author_email="mayukhmainak2000@gmail.com, kirill079@gmail.com",
    description="Data-agnOstic Representation Analysis of Deep Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lapalap/dora",
    packages=setuptools.find_packages(),
    install_requires=required,
    python_requires=">=3.6",
    include_package_data=True,
    keywords=["machine learning", "neural networks", "representation analysis"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
    ],
)
