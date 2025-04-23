FROM josafatburmeister/pointtree:latest

RUN python -m pip install \
    dacite \
    fire \
    geojson \
    geopandas \
    rasterio \
    seaborn \
    tifffile \
    git+https://github.com/weecology/DeepForest.git \
    git+https://github.com/ai4trees/pointtree.git \
    lightning==2.5.1
