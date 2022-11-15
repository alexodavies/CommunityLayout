from setuptools import setup

setup(
    name='community_layout',
    version='0.1.0',
    description='A Python package for community-separated layout of large networkx graphs.',
    url='https://github.com/neutralpronoun/CommunityLayout.git',
    author='Alex Davies',
    author_email='alexander.davies@bristol.ac.uk',
    license='GPLv3',
    packages=['community_layout'],
    install_requires=['networkx'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3'
    ],
)
