#!/bin/bash
  
CMD=${1:-/bin/bash}

docker run -it --rm \
  -v $PWD:/workspace/ \
  --workdir=/workspace/ \
  jetson-onnxruntime-object-detection $CMD