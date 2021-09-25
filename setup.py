import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
   README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='EEG_Tensorflow_models',
    version='0.1',
    packages=['EEG_Tensorflow_models'],

    author='D.G Garcia-Murillo, J.C Caicedo-Acosta',
    author_email='dggarciam@unal.edu.co, jccaicedoac@unal.edu.co',
    maintainer='D.G Garcia-Murillo, J.C Caicedo-Acosta',
    maintainer_email='dggarciam@unal.edu.co, jccaicedoac@unal.edu.co',

    download_url='',

    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'tensorflow',
                      'tensorflow-addons',
                      'mne',
                      'moabb',
                      'skorch',
                      'braindecode',
                      'braindecode @ git+https://github.com/braindecode/braindecode',
                      ],
#     extras_require={
#         'utils': ['sphinx',
#                   'colorama',
#                   'jinja2',
#                   'ipython',
#                   ],
#     },

#     scripts=[
#        "cmd/gcpds_distutils",
#     ],


    include_package_data=True,
    license='Simplified BSD License',
    description="",
    zip_safe=False,

    long_description=README,
    long_description_content_type='text/markdown',

    python_requires='>=3.6',

    classifiers=[
       'Development Status :: 4 - Beta',
       'Intended Audience :: Developers',
       'Intended Audience :: Education',
       'Intended Audience :: Healthcare Industry',
       'Intended Audience :: Science/Research',
       'License :: OSI Approved :: BSD License',
       'Programming Language :: Python :: 3.7',
       'Programming Language :: Python :: 3.8',
       'Topic :: Scientific/Engineering',
       'Topic :: Scientific/Engineering :: Artificial Intelligence',
       'Topic :: Scientific/Engineering :: Human Machine Interfaces',
       'Topic :: Scientific/Engineering :: Medical Science Apps.',
       'Topic :: Software Development :: Embedded Systems',
       'Topic :: Software Development :: Libraries',
       'Topic :: Software Development :: Libraries :: Application Frameworks',
       'Topic :: Software Development :: Libraries :: Python Modules',
       'Topic :: System :: Hardware :: Hardware Drivers',
    ],

)
