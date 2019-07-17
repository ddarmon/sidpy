sidpy: Python Tools for Specific Information Dynamics
=====================================================

What is sidpy?
---------------

<b>sidpy</b> is a Python package for estimating quantities from <b>s</b>pecific <b>i</b>nformation <b>d</b>ynamics such as [specific entropy rate](http://www.mdpi.com/1099-4300/18/5/190) and [specific transfer entropy](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.022121) from data, as well as local information dynamics via the [Java Information Dynamics Toolbox (JIDT)](https://github.com/jlizier/jidt). <b>sidpy</b> also incorporates model selection routines to determine the appropriate model order for scalar and input-output time series.

Installation
------------

To use sidpy, you need to install these 4 Python packages, which you can most easily install via the [Anaconda](https://www.anaconda.com/distribution/) distribution and pip:

> conda install -c conda-forge nlopt

> conda install -c conda-forge pyflann

> conda install -c conda-forge jpype1

> pip install sdeint

Using sidpy
------------

See the Jupyter notebooks in <tt>example-notebooks</tt> for a demonstration of the functionality of <b>sidpy</b>.

If you use **sidpy** in a scientific publication, please cite this GitHub repository.