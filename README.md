# pyqrack-examples
Useful and "cool" example scripts for PyQrack

## Introduction
[PyQrack](https://github.com/unitaryfund/pyqrack) is the official Python wrapper for the (C++) [Qrack](https://github.com/unitaryfund/qrack) quantum computer simulator and framework. [Examples already exist as a Jupyter Notebook collection.](https://github.vom/vm6502q/pyqrack-jupyter).

Most people with interest in quantum computing would probably be happy to consume PyQrack via Jupyter Notebook and Jupyter Lab. However, while Qrack recognizes and supports that popular ("data science") workflow, relying on so many development environment and runtime dependencies is, basically, antithetical to the original point and design of the Qrack project. Qrack has _no_ required dependencies whatsoever, except for the C ABI, `libc6`, as a dependency of any C or C++ "pure-language" code project. Qrack optionally provides GPU acceleration via choice of OpenCL or CUDA, but even this is an _optional_ dependency that can be omitted. 128-bit floating point math can optionally be supplied by the [Boost libraries](https://www.boost.org/), but even Qrack's ("big integers") "arbitrary precision integers," when demanded according to user build settings, is _"pure language standard."_

You can understand, Qrack prides itself on a comparatively tiny, secure, self-sufficient "supply chain." While we certainly don't _object_ to "more typical" workflow "supply chains," with large, cascading dependency trees across the Python ecosystem, we _should_ supply examples and applications that work in a minimalist development environment. Hence, besides C++ Qrack examples in the library repository itself, we provide these Python script examples as well, with minimal dependencies.

## Getting started
Clone this repository:
```sh
 $ git clone htttps://github.com/vm6502q/pyqrack-examples.git
```
If this doesn't work, you might need to install `git`, first.

Install PyQrack. On Ubuntu, you can use the official system package:
```sh
 $ sudo add-apt-repository ppa:wrathfulspatula/vm6502q
 $ sudo apt update
 $ sudo apt install pyqrack
```
Otherwise, if the Ubuntu package won't work with your system, you can use `pip`:
```sh
 $ python3 -m pip3 install pyqrack
```
or, depending on your environment, this is the basic idea:
```sh
 $ pip install pyqrack
```

That's it! To run any of the scripts in this repository, just invoke the Python interpreter! For example, this script times a random circuit sampling benchmark:
```sh
 $ python3 rcs.py
```
