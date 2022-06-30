from setuptools import setup

setup(name='XID_plus',
      version='3.0',
      author='Peter Hurley',
      author_email='p.d.hurley@sussex.ac.uk',
      url='http://pdh21.github.io/XID_plus/',
      download_url='https://github.com/pdh21/XID_plus',
      description='XID+ is the next generation deblender tool for Herschel SPIRE maps. Its uses a probabilistic framework which allows the use prior information about the sources.',
      # py_modules=['xidplus'],
      packages=["xidplus", "./"],
      install_requires=['astropy','dill', 'numpy','scipy','seaborn','healpy','pymoc','daft','aplpy','numpyro','jax','arviz'],
      keywords='',
      zip_safe = False,
      license='MIT',

     )
