"""GeoTIFF stacks -> HDF5 tensors of shape (n_windows, H, W, 14).

TODO: implement. Reads fg_<aoi>_<season>_<YYYYMMDD>.tif from S2_EXPORT_DIR,
stacks by date, writes one HDF5 per (aoi, season) into HDF5_DIR with a
`dates` vector. See CLAUDE.md section 5 for the data contract.
"""
