from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='oceanmodes',
      version='0.1',
      description='QG baroclinic mode analysis for ocean data',
      url='https://bitbucket.org/ryanaberanthey/mitgcmdata',
      author='Ryan Abernathey',
      author_email='rpa@ldeo.columbia.edu',
      license='MIT',
      packages=['oceanmodes'],
      install_requires=[
          'numpy','scipy'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'coverage'],
      zip_safe=False)
