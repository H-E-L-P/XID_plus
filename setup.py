from distutils.core import setup
import sys

sys.path.append('XID_plus')
import 


setup(name='XID_plus',
      version='0.1',
      author='Peter Hurley',
      author_email='p.d.hurley@sussex.ac.uk',
      url='http://pdh21.github.io/XID_plus/',
      download_url='https://github.com/pdh21/XID_plus',
      description='XID+ is the next generation deblender tool for Herschel SPIRE maps. Its uses a probabilistic framework which allows the use prior information about the sources.',
      package_dir={'': 'XID_plus'},
      py_modules=['XIDp_mode_beta'],
      requires=['astropy','pystan','pickle','dill', 'numpy','scipy'],
      keywords='',
      license='Lesser Affero General Public License v3',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Developers',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2',
                   'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
                   'License :: OSI Approved :: GNU Affero General Public License v3',
                   'Topic :: Internet',
                   'Topic :: Internet :: WWW/HTTP',
                   'Topic :: Scientific/Engineering :: GIS',
                  ],
     )
