mestra and other formats
========================

Which file to reach for, and how mestra sits beside the formats you
already use. The short version: VTK, CGNS and Exodus describe one
simulation well; mestra describes a study -- many runs over one
support, with the inputs, roles, units, splits and models that make
them a dataset -- and borrows VTK's cell types and netCDF-4's
container, so that moving between them is a short script.
`design-notes.md` has the reasoning behind each choice below. What is
said here about other projects is as of October 2026, from their own
documentation, linked below; a correction is welcome as an issue.


When something else is the better file
--------------------------------------

- One simulation, with its zones, boundary conditions and solver
  state, for a solver or a post-processor: CGNS, Exodus II or VTK.
  A study made of such runs can still be gathered into one mestra
  file when it is time to compare the runs or fit a model to them.
- Gridded earth-system data on longitude, latitude and time: netCDF
  with the CF conventions, whose tools understand those axes.
- A machine-learning training set whose samples have different
  meshes, several zones each, or a mesh that changes during a run,
  served from the Hugging Face Hub: PLAID was built for that case.
  mestra holds samples with different meshes as separate supports,
  but its strength is the aligned case, where every row shares one.
- A handful of scalars that only a spreadsheet will ever read: CSV is
  simpler. A mestra file of keys and scalars is valid and small, and
  it is worth it once units, failed runs or a split need to travel
  with the numbers.
- Very large datasets read in pieces over a network: mestra is one
  HDF5 file per dataset, read through the HDF5 library. A chunked
  object-store layout such as Zarr serves that case better today.


Side by side
------------

|                                | mestra | VTK, VTKHDF | CGNS | Exodus II | netCDF-CF, UGRID | PLAID | The Well, PDEBench |
|---|---|---|---|---|---|---|---|
| What one file or folder holds  | a study: many rows | one mesh, or a time series | one simulation | one simulation | one dataset of variables | a dataset folder: YAML, and a CGNS tree per sample | one collection, in a layout of its own |
| Unstructured mesh              | VTK cell codes, mixed | yes | yes, multi-zone | yes, in blocks | UGRID | yes, through CGNS | uniform grids |
| Inputs of each run, by role    | keys: design, condition, time, group, split, status | -- | -- | -- | a `realization` axis for ensemble members | inputs and outputs listed in a problem definition | per collection |
| Units on every quantity        | required (E11) | -- | optional | -- | `units` | not required | per collection |
| Splits                         | a split key, checked against the declared group (W01) | -- | -- | -- | -- | index lists in a problem definition | separate files, or per collection |
| Run outcome                    | a status key (W02) | -- | -- | -- | -- | -- | -- |
| Uncertainty                    | draws, bands, quantiles | -- | -- | -- | ancillary variables | -- | -- |
| Fitted models                  | callables, in the same schema | -- | -- | -- | -- | -- | -- |
| Read by                        | Python, C++, MATLAB, Julia; any HDF5 or netCDF-4 tool | many | many | several | many | Python | Python, through their own loaders |

A dash means the format has no place for the thing, not that a
producer cannot put it somewhere by convention; a convention that only
its own reader knows is the problem mestra is for.

Sources: [VTKHDF](https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html),
[CGNS](https://cgns.org),
[Exodus II](https://github.com/sandialabs/seacas),
[CF conventions](https://cfconventions.org),
[UGRID](https://ugrid-conventions.github.io/ugrid-conventions/),
[PLAID](https://github.com/PLAID-lib/plaid) (its disk format,
sample and problem-definition pages, and the 1.0 upgrade guide),
[The Well](https://github.com/PolymathicAI/the_well/blob/master/docs/data_format.md),
[PDEBench](https://github.com/pdebench/PDEBench).


Moving data in and out
----------------------

- **VTK.** Cell types are VTK's codes, in VTK's node order (SPEC
  section 20), so a support is a VTK unstructured grid and each row's
  arrays are its point and cell data. Reading a folder of VTU files
  with meshio and writing them as rows of one support is a short
  script.
- **xarray and netCDF-4 tools.** A mestra file is a valid netCDF-4
  file: `xarray.open_datatree(path)` opens every group with its named
  dimensions, without this package, and `Dataset.to_xarray()` gives one
  flattened `xarray.Dataset`.
- **Exodus II.** The model is Exodus generalised: the time index
  becomes role-tagged keys, and element blocks become label arrays.
- **PLAID.** A sample maps to a row and a constant-connectivity
  collection to one aligned support (the VKI-LS59 mapping in
  `mappings.md`); a problem definition's inputs become keys and its
  splits a split key. A PLAID dataset does not declare roles, units
  or groups, so a conversion supplies them.
- **Dataset catalogues.** A mestra file is one HDF5 file, so a
  Croissant description or a dataset card can point at it like any
  other; a table of its keys and scalars as Parquet beside it is what
  the Hugging Face viewer can show.
- **Models.** A callable names its type and carries that type's
  dictionary; FMI and ONNX are formats a tool could carry as a
  callable type of its own, the way `affine` is carried here.


What mestra does not do
-----------------------

It does not run models other than the reference `affine`: a callable
names the model that serves a slot and carries its dictionary, and
executing any other type needs the software that owns it
(`compatibility.md`). It does not keep lineage or validation records;
those belong to the tools that produce the file. And it is
specification version 0: the meaning of an existing file does not
change within it, and what may be added is in `../CONTRIBUTING.md`.
