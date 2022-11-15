from setuptools import setup

setup(
    name='community_layout',
    version='0.1.0',
    description='A Python package for community-separated layout of large networkx graphs.',
    url='https://github.com/shuds13/pyexample',
    author='Alex Davies',
    author_email='alexander.davies@bristol.ac.uk',
    license='GNU GPL3',
    packages=['community_layout'],
    install_requires=['numpy',
                      'matplotlib',
                      'networkx',
                      'pandas',
                      'datashader',
                      'tqdm',
                      'scikit-image'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU GPL3 License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3'
    ],
)
