from setuptools import setup

setup(name='XID_plus',
      version='0.1',
      author='Peter Hurley',
      author_email='p.d.hurley@sussex.ac.uk',
      url='http://pdh21.github.io/XID_plus/',
      download_url='https://github.com/pdh21/XID_plus',
      description='XID+ is the next generation deblender tool for Herschel SPIRE maps. Its uses a probabilistic framework which allows the use prior information about the sources.',
      package_dir={'': 'xidplus'},
      py_modules=['xidplus'],
      requires=['astropy','pystan','pickle','dill', 'numpy','scipy'],
      keywords='',
      license='Lesser Affero General Public License v3',

     )
