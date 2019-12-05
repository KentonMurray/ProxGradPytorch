from distutils.core import setup

setup(
    name='proximal_gradient',
    version='0.1.0',
    author='Kenton Murray',
    author_email='kmurray4@nd.edu',
    packages=['proximal_gradient'],
    scripts=[],
    url='',#'http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Proximal Gradient Methods for Pytorch',
    long_description=open('README.md').read(),
    install_requires=[
        "torch >= 0.4.0",
    ],
)
