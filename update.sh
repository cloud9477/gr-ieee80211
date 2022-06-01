#!/bin/bash

echo "update start"

cd ./build/
sudo make uninstall
make clean
make -j8
sudo make install
sudo ldconfig

echo "update finish"

