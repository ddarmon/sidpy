sidpy: Python Tools for Specific Information Dynamics
=====================================================

What is sidpy?
---------------

<b>sidpy</b> is a Python package for estimating quantities from <b>s</b>pecific <b>i</b>nformation <b>d</b>ynamics such as [specific entropy rate](http://www.mdpi.com/1099-4300/18/5/190) and [specific transfer entropy](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.022121) from data, as well as local information dynamics via the [Java Information Dynamics Toolbox (JIDT)](https://github.com/jlizier/jidt). <b>sidpy</b> also incorporates model selection routines to determine the appropriate model order for scalar and input-output time series.

Installation
------------

```
pip install git+https://github.com/ddarmon/sidpy
```

## Installation on ARM-based Macs

If you run into an error message like the following after running `import sidpy`:

```
File /path/site-packages/pyflann/bindings/flann_ctypes.py:173
    171 flannlib = load_flann_library()
    172 if flannlib == None:
--> 173     raise ImportError('Cannot load dynamic library. Did you compile FLANN?')
    176 class FlannLib:
    177     pass

ImportError: Cannot load dynamic library. Did you compile FLANN?
```

this is likely because the dynamic library `libflann.dylib` is not in the `pyflann/bindings` folder.

To fix this:

1. **Install `flann`**
   ```
   brew install flann
   ```

2. **Find Installation Location**
   Find where the `flann` library is installed by running:
   ```
   brew info flann
   ```
   Go to the location provided by `brew`. On my current installation and version of `flann`, it is:
   ```
   cd /opt/homebrew/Cellar/flann/1.9.2_1
   ```

3. **Copy the `libflann.v.v.v.dylib` File**
   Finally, copy the `libflann.v.v.v.dylib` file to `site-packages/pyflann` as `libflann.dylib` using:
   ```
   mkdir -p /path/site-packages/pyflann/bindings/lib/darwin/
   cp /opt/homebrew/Cellar/flann/1.9.2_1/lib/libflann.1.9.2.dylib /path/site-packages/pyflann/bindings/lib/darwin/libflann.dylib
   ```

Make sure to replace `/path/` with the actual path where your `site-packages/pyflann` is located. Also, the version number in the `libflann.v.v.v.dylib` should match the installed version of `flann`. In this example, it's `1.9.2`.

Using sidpy
------------

See the Jupyter notebooks in <tt>example-notebooks</tt> for a demonstration of the functionality of <b>sidpy</b>.

If you use **sidpy** in a scientific publication, please cite this GitHub repository.
