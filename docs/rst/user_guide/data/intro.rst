.. _sec-user_guide-data-intro:

Introduction
============

Eradiate ships, processes and produces data. This guide presents:

* the rationale underlying data models used in Eradiate;
* the components used to manipulate data shipped with Eradiate.

Formats
-------

Most data sets used and produced by Eradiate are stored in the
`NetCDF format <https://www.unidata.ucar.edu/software/netcdf/>`_. Eradiate
interacts with these data using the `xarray <https://xarray.pydata.org/>`_
library, whose data model is based on NetCDF. Xarray provides a comprehensive,
robust and convenient interface to read, write, manipulate and visualise NetCDF
data.

Metadata conventions
--------------------
Data sets include metadata to describe what the data represents.
As much as possible, we try to follow the 
`NetCDF Climate and Forecast (CF) Metadata Conventions, version 1.8 
<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html>`_.
These metadata includes variable names and units but also information about
where and how the data set was produced as well as links to published 
or web-based references describing the data set of the method used to produce it.


.. _sec-user_guide-manual_download:

Adding manually downloaded data
-------------------------------

Due to the impracticality of storing large data sets with the code base,
Eradiate does not ship all data required to run simulations.
Certain large data sets are hosted on a `FTP server <https://eradiate.eu/data>`_
and must be downloaded manually.

The data are served as compressed archives, including their containing folders.
To install the downloaded data, first decompress the archive into a temporary
location. The decompressed folder can then be placed directly in the
``resources/data`` folder and typical file managers will be able to merge the
two file trees, placing the data files in the correct location in the local file
tree.

Accessing shipped data
----------------------

Eradiate ships with a series of data sets located in its ``resources/data``
directory. Some of the larger data sets consist of aggregates of many NetCDF
files, while others are stand-alone NetCDF files. In order to provide a simple
and unified interface, Eradiate references data sets in a registry which can be
queried using the :mod:`eradiate.data` module.

Data sets are grouped by category, *e.g.* solar irradiance spectrum, absorption
spectrum, etc. The complete list of registered categories can be found in the
reference documentation for the :mod:`eradiate.data` module. Each data set is
then referenced by an identifier unique in its category. The pair
(category, identifier) therefore identifies a data set completely.

To load a specific data set, the :func:`eradiate.data.open` function should be
used:

.. code-block:: python

   import eradiate
   ds = eradiate.data.open("solar_irradiance_spectrum", "thuillier_2003")

The :func:`~eradiate.data.open` function can also be used to load user-defined
data at a known location:

.. code-block:: python

   ds = eradiate.data.open("path/to/my/data.nc")

.. note::

   :func:`~eradiate.data.open` resolves paths using Eradiate's
   :class:`.PathResolver`.

.. _sec-user_guide-data_guide-working_angular_data:

Working with angular data
-------------------------

Eradiate notably manipulates and produces what we refer to as *angular data*,
which represent variables dependent on one or more directional parameters.
Typical examples are BRDFs
(:math:`f_\mathrm{r} (\theta_\mathrm{i}, \varphi_\mathrm{i}, \theta_\mathrm{o}, \varphi_\mathrm{o})`)
or top-of-atmosphere BRFs
(:math:`\mathit{BRF}_\mathrm{TOA} (\theta_\mathrm{sun}, \varphi_\mathrm{sun}, \theta_\mathrm{view}, \varphi_\mathrm{view})`):
a xarray data array representing them has at least one angular dimension (and
corresponding coordinates). Eradiate has specific functionality to deal more
easily with this sort of data.

Angular dependencies and coordinate variable names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Angular variable naming in Earth observation and radiative transfer modelling
may sometimes clash or be confusing. Eradiate clearly distinguishes between two
types of angular dependencies for its variables:

* Physical properties such as BRDFs and phase functions have intrinsic
  bidirectional dependencies which are referred to as *incoming* and *outgoing*
  directions. Data sets representing such quantities use  coordinate variables
  ``phi_i``, ``theta_i`` for the incoming direction's azimuth and zenith angles,
  and ``phi_o``, ``theta_o`` for their outgoing counterparts.

* Observations are usually parametrised by *illumination* (or *solar*) and
  *viewing* (or *sensor*) directions. For data sets representing such results,
  Eradiate uses coordinate variables ``sza``, ``saa`` for
  *solar zenith/azimuth angle* and ``vza``, ``vaa`` for
  *viewing zenith/azimuth angle*. A typical example of such variable is
  the top-of-atmosphere bidirectional reflectance factor (TOA BRF).

Under specific circumstances, one can directly convert an observation dataset to
a physical property dataset. This, for instance, applies to top-of-atmosphere
BRF data, but also any BRF computed or measured in a vacuum. In such cases,
incoming/outgoing directions can be directly converted to
illumination/viewing directions. **But in general, this does not work.**

Angular data set types
^^^^^^^^^^^^^^^^^^^^^^

While one should clearly distinguish intrinsic and observation angular
dependencies for correct physical interpretation of radiative data, both share
an asymmetry between 'incoming' and 'outgoing' directions. Eradiate uses
similar semantics to handle both angular data types, and the table below clarifies
the nomenclature for the two types:

.. list-table::
   :header-rows: 1

   * - Type
     - Incoming
     - Outgoing
   * - Intrinsic
     - :math:`\varphi_\mathrm{i}`, :math:`\theta_\mathrm{i}`
     - :math:`\varphi_\mathrm{o}`, :math:`\theta_\mathrm{o}`
   * - Observation
     - :math:`\varphi_\mathrm{s}`, :math:`\theta_\mathrm{s}`
     - :math:`\varphi_\mathrm{v}`, :math:`\theta_\mathrm{v}`

Eradiate's xarray containers do not explicitly keep track of the angular data
set type. However, when relevant, coordinate naming is used to determine whether
an angular data set is of intrinsic or observation type.

Angular data sets with a pair of angular dimensions :math:`(\theta, \varphi)`
are called *hemispherical*. If they have two pairs of angular dimensions
(incoming and outgoing), they are then called *bi-hemispherical*.

Measure data formats
--------------------

Most measures in Earth observation radiative transfer modelling have angular
dependencies. However, Eradiate uses storage data structures inherited from
computer graphics technology and measure results are usually mapped against
*film coordinates* :math:`(x, y) \in [0, 1]^2`. When those data represent
hemispherical quantities, a mapping transformation associate angles to film
coordinates. For convenience, Eradiate ships helpers to convert data from film
coordinates to angular coordinates. See
:ref:`sphx_glr_examples_generated_tutorials_data_01_polar_plot.py` for a
concrete introduction to those features, as well as angular data visualisation
in polar coordinates.
