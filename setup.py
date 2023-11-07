from setuptools import setup

setup(
    name = 'BDfinder',
    version = '0.1.0',
    description = 'A ML-based algorithm that uses WISE and YJHK magnitudes to find brown dwarfs.',
    py_modules = ["classifier"],
    package_dir = {'':'src'},
    author = 'Ankit Biswas',
    author_email = 'ankit.biswas.31908@gmail.com',
    long_description = open('README.md').read() + '\n\n' + open('CHANGELOG.md').read(),
    long_description_content_type = "text/markdown",
    url='https://github.com/extremelylongname/BD-finder',
    include_package_data=True,
    
    classifiers  = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Text Processing',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
    ],
    
    
    install_requires = [

        'pandas >= 1.2.4',
        'numpy >= 1.23.0',
        'astropy >= 5.2.0',
        'torch >= 1.13.0',
        'scikit-learn >= 1.2.0'
    ],
    
    keywords = ['Data Science', 'Astronomy', 'Brown Dwarf', 'Astropy'],
    
)
