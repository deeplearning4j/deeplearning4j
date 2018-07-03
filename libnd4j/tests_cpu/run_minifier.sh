#!/bin/bash
# script for running and manual testing of the minifier
# by GS <sgazeos@gmail.com
#
# only for special use
#
#
CXX=/usr/bin/g++
#CXX_PATH=`$CXX --print-search-dirs | awk '/install/{print $2;}'`
#export CXX_PATH
export CXX

make -j4 && layers_tests/minifier -l -o nd4j_minilib.h 
#
#echo "TESTING MINIFIER with all resources"
#echo "Testing adam_sum.fb"
#layers_tests/minifier -l -o nd4j_adam.h ./resources/adam_sum.fb
#echo "Done"
#
#echo "Testing ae_00.fb"
#layers_tests/minifier -l -o nd4j_ae.h ./resources/ae_00.fb
#echo "Done"
#
#layers_tests/minifier -l -o nd4j_conv.h ./resources/conv_0.fb
#layers_tests/minifier -l -o nd4j_expand_dim.h ./resources/expand_dim.fb
#layers_tests/minifier -l -o nd4j_inception.h ./resources/inception.fb
#layers_tests/minifier -l -o nd4j_nested_while.h ./resources/nested_while.fb
#layers_tests/minifier -l -o nd4j_partition_stitch_misc.h ./resources/partition_stitch_misc.fb
#layers_tests/minifier -l -o nd4j_reduce_dim_false.h ./resources/reduce_dim_false.fb
#layers_tests/minifier -l -o nd4j_reduce_dim.h ./resources/reduce_dim.fb
#layers_tests/minifier -l -o nd4j_reduce_dim_true.h ./resources/reduce_dim_true.fb
##layers_tests/minifier -l -o nd4j_simpleif01.h ./resources/simpleif_0_1.fb
##layers_tests/minifier -l -o nd4j_simpleif0.h ./resources/simpleif_0.fb
##layers_tests/minifier -l -o nd4j_simpleif_java.h ./resources/simpleif_0_java.fb
#layers_tests/minifier -l -o nd4j_simplewhile03.h ./resources/simplewhile_0_3.fb
#layers_tests/minifier -l -o nd4j_simplewhile04.h ./resources/simplewhile_0_4.fb
#layers_tests/minifier -l -o nd4j_simplewhile0.h ./resources/simplewhile_0.fb
#layers_tests/minifier -l -o nd4j_simplewhile1.h ./resources/simplewhile_1.fb
#layers_tests/minifier -l -o nd4j_simple_while.h ./resources/simple_while.fb
#layers_tests/minifier -l -o nd4j_simplewhile_nested.h ./resources/simplewhile_nested.fb
#layers_tests/minifier -l -o nd4j_tensor_array.h ./resources/tensor_array.fb
#layers_tests/minifier -l -o nd4j_tensor_array_loop.h ./resources/tensor_array_loop.fb
#layers_tests/minifier -l -o nd4j_tensor_dot_misc.h ./resources/tensor_dot_misc.fb
#layers_tests/minifier -l -o nd4j_tensor_slice.h ./resources/tensor_slice.fb
#layers_tests/minifier -l -o nd4j_three_args_while.h ./resources/three_args_while.fb
#layers_tests/minifier -l -o nd4j_transpose.h ./resources/transpose.fb
#
#echo "All Done (for g++)!!!"
#
#CXX=/usr/bin/g++-5
#CXX_PATH=`$CXX --print-search-dirs | awk '/install/{print $2;}'`
#export CXX_PATH
#export CXX
#
#make -j4 && layers_tests/minifier -l -o nd4j_minilib.h 
#
##echo "TESTING MINIFIER with all resources"
##echo "Testing adam_sum.fb"
##layers_tests/minifier -l -o nd4j_adam.h ./resources/adam_sum.fb
##echo "Done"
##
##echo "Testing ae_00.fb"
##layers_tests/minifier -l -o nd4j_ae.h ./resources/ae_00.fb
##echo "Done"
##
##layers_tests/minifier -l -o nd4j_conv.h ./resources/conv_0.fb
##layers_tests/minifier -l -o nd4j_expand_dim.h ./resources/expand_dim.fb
#layers_tests/minifier -l -o nd4j_inception.h ./resources/inception.fb
#layers_tests/minifier -l -o nd4j_nested_while.h ./resources/nested_while.fb
##layers_tests/minifier -l -o nd4j_partition_stitch_misc.h ./resources/partition_stitch_misc.fb
#layers_tests/minifier -l -o nd4j_reduce_dim_false.h ./resources/reduce_dim_false.fb
#layers_tests/minifier -l -o nd4j_reduce_dim.h ./resources/reduce_dim.fb
#layers_tests/minifier -l -o nd4j_reduce_dim_true.h ./resources/reduce_dim_true.fb
##layers_tests/minifier -l -o nd4j_simpleif01.h ./resources/simpleif_0_1.fb
##layers_tests/minifier -l -o nd4j_simpleif0.h ./resources/simpleif_0.fb
##layers_tests/minifier -l -o nd4j_simpleif_java.h ./resources/simpleif_0_java.fb
#layers_tests/minifier -l -o nd4j_simplewhile03.h ./resources/simplewhile_0_3.fb
#layers_tests/minifier -l -o nd4j_simplewhile04.h ./resources/simplewhile_0_4.fb
#layers_tests/minifier -l -o nd4j_simplewhile0.h ./resources/simplewhile_0.fb
##layers_tests/minifier -l -o nd4j_simplewhile1.h ./resources/simplewhile_1.fb
##layers_tests/minifier -l -o nd4j_simple_while.h ./resources/simple_while.fb
##layers_tests/minifier -l -o nd4j_simplewhile_nested.h ./resources/simplewhile_nested.fb
##layers_tests/minifier -l -o nd4j_tensor_array.h ./resources/tensor_array.fb
#layers_tests/minifier -l -o nd4j_tensor_array_loop.h ./resources/tensor_array_loop.fb
#layers_tests/minifier -l -o nd4j_tensor_dot_misc.h ./resources/tensor_dot_misc.fb
#layers_tests/minifier -l -o nd4j_tensor_slice.h ./resources/tensor_slice.fb
#layers_tests/minifier -l -o nd4j_three_args_while.h ./resources/three_args_while.fb
#layers_tests/minifier -l -o nd4j_transpose.h ./resources/transpose.fb
#
#echo "All Done!!!"
#
#CXX=/usr/bin/g++-7
#CXX_PATH=`$CXX --print-search-dirs | awk '/install/{print $2;}'`
#export CXX_PATH
#export CXX
#
#make -j4 && layers_tests/minifier -l -o nd4j_minilib.h 
#
#echo "TESTING MINIFIER with all resources"
#echo "Testing adam_sum.fb"
#layers_tests/minifier -l -o nd4j_adam.h ./resources/adam_sum.fb
#echo "Done"
#
#echo "Testing ae_00.fb"
#layers_tests/minifier -l -o nd4j_ae.h ./resources/ae_00.fb
#echo "Done"
#
##layers_tests/minifier -l -o nd4j_conv.h ./resources/conv_0.fb
#layers_tests/minifier -l -o nd4j_expand_dim.h ./resources/expand_dim.fb
#layers_tests/minifier -l -o nd4j_inception.h ./resources/inception.fb
#layers_tests/minifier -l -o nd4j_nested_while.h ./resources/nested_while.fb
#layers_tests/minifier -l -o nd4j_partition_stitch_misc.h ./resources/partition_stitch_misc.fb
#layers_tests/minifier -l -o nd4j_reduce_dim_false.h ./resources/reduce_dim_false.fb
#layers_tests/minifier -l -o nd4j_reduce_dim.h ./resources/reduce_dim.fb
#layers_tests/minifier -l -o nd4j_reduce_dim_true.h ./resources/reduce_dim_true.fb
#layers_tests/minifier -l -o nd4j_simpleif01.h ./resources/simpleif_0_1.fb
#layers_tests/minifier -l -o nd4j_simpleif0.h ./resources/simpleif_0.fb
#layers_tests/minifier -l -o nd4j_simpleif_java.h ./resources/simpleif_0_java.fb
#layers_tests/minifier -l -o nd4j_simplewhile03.h ./resources/simplewhile_0_3.fb
#layers_tests/minifier -l -o nd4j_simplewhile04.h ./resources/simplewhile_0_4.fb
#layers_tests/minifier -l -o nd4j_simplewhile0.h ./resources/simplewhile_0.fb
#layers_tests/minifier -l -o nd4j_simplewhile1.h ./resources/simplewhile_1.fb
#layers_tests/minifier -l -o nd4j_simple_while.h ./resources/simple_while.fb
#layers_tests/minifier -l -o nd4j_simplewhile_nested.h ./resources/simplewhile_nested.fb
#layers_tests/minifier -l -o nd4j_tensor_array.h ./resources/tensor_array.fb
#layers_tests/minifier -l -o nd4j_tensor_array_loop.h ./resources/tensor_array_loop.fb
#layers_tests/minifier -l -o nd4j_tensor_dot_misc.h ./resources/tensor_dot_misc.fb
#layers_tests/minifier -l -o nd4j_tensor_slice.h ./resources/tensor_slice.fb
#layers_tests/minifier -l -o nd4j_three_args_while.h ./resources/three_args_while.fb
#layers_tests/minifier -l -o nd4j_transpose.h ./resources/transpose.fb
#
echo "All Done!!!"
