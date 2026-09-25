#!/bin/bash
# Download MVTec AD into data/mvtec_ad (about 5 GB).

mkdir -p data/mvtec_ad
cd data/mvtec_ad
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz
rm mvtec_anomaly_detection.tar.xz
cd ../..
