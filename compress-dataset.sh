#!/bin/bash
for folder in $1/*/; do
    echo "Compressing $folder"

    # Remove the trailing slash
    zfile=${folder%*/}
    # 7z a -t7z -m0=lzma2 -mx=9 -mfb=128 -md=64m -ms=on $zfile.7z $folder
    # 7z a -t7z -m0=lzma2 -mx=0 -mfb=128 -md=64m -ms=on $zfile.17z $folder

    env GZIP=-9 tar -zcvf $zfile.tar.gz $folder
done