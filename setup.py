from setuptools import setup, find_packages

setup(
    name="ScratchNet",
    version="0.1dev",
    license="MIT License",
    author="Angus S. Hilts",
    author_email="grimeyjr@gmail.com",
    long_description=open('README.md').read(),
    packages=find_packages(include=["scratchnet"]),
    tests_require=['pytest', 'pytest-cov'],
    test_suite="tests",
    install_requires=[
        'numpy',
    ]  # ,
    # entry_points={
    #    'console_scripts': ['snktx=snaketax.snaketax:main',],
    # }
)
