from pathlib import Path

from setuptools import setup

setup(name='autoclean',
      version='1.0.0',
      description='A library for cleaning text data',
      long_description=(Path(__file__).parent.resolve() / 'README.md').read_text(encoding='utf8'),
      long_description_content_type='text/markdown',
      url='https://github.com/JoseLlarena/autoclean',
      author='Jose Llarena',
      author_email='jose.llarena@gmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Other Environment',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Utilities',
          'Topic :: Software Development :: Libraries'
      ],
      keywords='cleaning, text, unsupervised, segmentation',
      zip_safe=False,
      package_data={'autoclean': ['lmplz', 'build_binary', 'ngram-count']},
      include_package_data=True,
      packages=['autoclean', 'autoclean.filtering', 'autoclean.segmentation'],
      python_requires='>=3.8',
      dependency_links=['https://download.pytorch.org/whl/cu113/torch_stable.html'],
      install_requires=['pypey',
                        'click',
                        'torch @ https://download.pytorch.org/whl/cu113/torch-1.10.0%2Bcu113-cp38-cp38-linux_x86_64.whl',
                        'torchaudio @ https://download.pytorch.org/whl/cu113/torchaudio-0.10.0%2Bcu113-cp38-cp38-linux_x86_64.whl',
                        'torchvision @ https://download.pytorch.org/whl/cu113/torchvision-0.11.1%2Bcu113-cp38-cp38-linux_x86_64.whl',
                        'kenlm @ https://github.com/kpu/kenlm/archive/master.zip'],
      extras_require={'test': ['pytest']})
