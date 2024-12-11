#!/usr/bin/env bash

LIBJEMALLOC="/home/agibsonccc/jemalloc-5.3.0/lib/libjemalloc.so.2"
MALLOC_CONF="prof:true,lg_prof_interval:31,lg_prof_sample:17,prof_prefix:jeprof.out"

mvn clean package

LD_PRELOAD="$LIBJEMALLOC" \
MALLOC_CONF="$MALLOC_CONF" \
java \
  -Dorg.bytedeco.javacpp.openblas.load=mkl_rt \
  -jar target/benchmarks.jar \
  -f2 \
  -i10 \
  -wi 2 \
  org.nd4j.*