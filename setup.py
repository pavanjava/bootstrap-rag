from setuptools import setup, find_packages
import pathlib

doc_path = pathlib.Path(__file__).parent.resolve()
long_description = (doc_path / "README.md").read_text(encoding="utf-8")

setup(
    name='bootstrap-rag',
    version='0.0.9',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
    ],
    entry_points={
        'console_scripts': [
            'bootstraprag=bootstraprag.cli:cli',
        ],
    },
    package_data={
        'bootstraprag': ['templates/*'],
    },
)