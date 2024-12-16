#!/bin/bash
LIBJEMALLOC="/home/agibsonccc/jemalloc-5.3.0/lib/libjemalloc.so.2"
MALLOC_CONF="prof:true,lg_prof_interval:31,lg_prof_sample:17,prof_prefix:jeprof.out"
mvn  package

LD_PRELOAD="$LIBJEMALLOC" \
MALLOC_CONF="$MALLOC_CONF" \
java \
  -cp target/benchmarks.jar \
   org.nd4j.memorypressure.MemoryPressureTest \
  -f2 \
  -i10 \
  -wi 2 \

jeprof --show_bytes --gif $(which java) jeprof.*.heap > app-profiling.gif
