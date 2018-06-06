# -*- coding: utf-8 -*-
from distutils.core import setup
from os import chdir
from os.path import dirname, abspath

curdir = dirname(abspath(__file__))
chdir(curdir)
setup(name='convnetskeras2',
      version='0.2',
      description='Pre-trained AlexNet in Keras2',
      author=['Leonard Blier', 'Filippo Santarelli'],
      author_email=['leonard.blier@ens.fr', 'filippo2.santarelli@gmail.com'],
      packages=['convnetskeras2'],
      package_dir={'convnetskeras2': 'src'},
      package_data={'convnetskeras2': ['data/*']},
      long_description=open('README.md').read(),
      )
