/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_PARITY_H
#define LIBND4J_HEADERS_PARITY_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation returns index of max element in a given NDArray (optionally: along given dimension(s))
         * Expected input:
         * 0: N-dimensional array
         * 1: optional axis vector
         *
         * Int args:
         * 0: optional axis
         */
        #if NOT_EXCLUDED(OP_argmax)
        DECLARE_CUSTOM_OP(argmax, 1, 1, false, 0, -2);
        #endif

        /**
         * This operation returns index of min element in a given NDArray (optionally: along given dimension(s))
         * Expected input:
         * 0: N-dimensional array
         * 1: optional axis vector
         *
         * Int args:
         * 0: optional axis
         */
        #if NOT_EXCLUDED(OP_argmin)
        DECLARE_CUSTOM_OP(argmin, 1, 1, false, 0, -2);
        #endif

        /**
         * This operation provides various normalization modes:
         * 0: frobenius
         * 1: euclidean (norm2)
         * 2: norm1
         * 3: norm2
         * 4: inf-norm
         * 5: p-norm
         *
         * Expected arguments:
         * input: N-dimensional array
         *
         *
         * Int args:
         * 0...: axis
         *
         * T args:
         * 0: norm mode
         * 1: p for p-norm
         */
        #if NOT_EXCLUDED(OP_norm)
        DECLARE_REDUCTION_OP(norm, 1, 1, false, 1, -2);
        #endif

        /**
        * Inserts elements provided by diagonal array into the main diagonal of innermost matrices of input array
        *
        * Input arrays:
        *  0: input array, considered as batch of matrices
        *  1: diagonal array containing elements to be inserted into input array,
        *     following rank condition should be satisfied: diagonal_rank = input_rank - 1,
        *     the shapes of diagonal and input arrays must be equal except last dimension of input array,
        *     for example if input_shape = [A,B,C,D] then diagonal_shape = [A,B,C],
        *     also last dimension of diagonal array should be equal to smaller of last and last but one input dimensions
        *     that is: diagonal_shape[-1] = min(input_shape[-1], input_shape[-2])
        *
        * Output array:
        *  0: has the same shape as input, corresponding diagonal elements are substituted
        */
        #if NOT_EXCLUDED(OP_matrix_set_diag)
        DECLARE_CONFIGURABLE_OP(matrix_set_diag, 2, 1, false, 0, 0);
        #endif

        /**
        * Inserts elements provided by diagonal array into the main diagonal of innermost matrices of output array,
        * rest output elements are set to zeros
        *
        * Input array:
        *    diagonal: array containing elements to be inserted into output array,
        *              following rank condition is present: diagonal_rank = ouput_rank - 1
        *
        * Output array:
        *   0: is considered as batch of matrices, if for example diagonal array has shape [A,B,C] then output array has shape [A,B,C,C]
        */
        DECLARE_CUSTOM_OP(matrix_diag, 1, 1, false, 0, 0);

        /**
        * This op calculates regularized incomplete beta integral Ix(a, b).
        * Implementation is based on two algorithms depending on input values of a and b:
        * - when a and b are both >  maxValue (3000.), then Gauss-Legendre quadrature method is applied
        * - when a and b are both <= maxValue (3000.), then modified Lentz’s algorithm for continued fractions is applied
        *
        * Input arrays:
        *    a: defines power t^{a-1}, must be > 0, type float.
        *    b: defines power (1-t)^{b-1}, must be > 0, type float.
        *    x: defines upper limit of integration, must be within (0 <= x <= 1) range, type float.
        *
        * Output array:
        *    0: values of  regularized incomplete beta integral that corresponds to variable upper limit x, type float
        *
        * Three input and one output arrays must have the same shape
        */
        #if NOT_EXCLUDED(OP_betainc)
        DECLARE_CONFIGURABLE_OP(betainc, 3, 1, false, 0, 0);
        #endif

        /**
         * This operation is added for compatibility purposes mostly.
         * PLEASE NOTE: Please consider using Add instead
         * Expected arguments:
         * 0: N-dimensional input
         * 1: bias vector
         */
        #if NOT_EXCLUDED(OP_biasadd)
        DECLARE_CUSTOM_OP(biasadd, 2, 1, true, 0, 0);
        DECLARE_CUSTOM_OP(biasadd_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * Returns a diagonal tensor with a given diagonal values. Given a diagonal, this operation returns a tensor with the diagonal and everything else padded with zeros.
         */
        #if NOT_EXCLUDED(OP_diag)
        DECLARE_CUSTOM_OP(diag, 1, 1, false, 0, 0);
        #endif

        /**
         * Returns a diagonal tensor with a given diagonal values. Given a diagonal, this operation returns a tensor with the diagonal and everything else padded with zeros.
         */
        #if NOT_EXCLUDED(OP_diag_part)
        DECLARE_CUSTOM_OP(diag_part, 1, 1, false, 0, 0);
        #endif

        /**
         * Returns a diagonal vector for any submatricies with in a given tensor.
         * It is an op inverse to matrix_set_giag.
         * Using input tensor as batched 2D diagonals flat them to vector (1D) with diagonal values.
         *
         * Input : batched tensor with rank >=2
         * Output: tensor with rank lesser by 1 from input
         */
        DECLARE_CUSTOM_OP(matrix_diag_part, 1, 1, false, 0, 0);


        /**
         * This operation takes 2 arrays: original values, and values to be excluded. And returns 2 arrays: values left after exclusion, and indices in original array for surivals.
         * Expected arguments:
         * 0: vector with original values
         * 1: vector with values to exclude
         */
        #if NOT_EXCLUDED(OP_listdiff)
        DECLARE_CUSTOM_OP(listdiff, 2, 2, false, 0, 0);
        #endif

        /**
         * This operation applies Add operation to specific inputs wrt indices
         * Expected arguments:
         * input: array to be updated
         * indices: array containing indexes for first dimension of input
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_add)
        DECLARE_OP(scatter_add, 3, 1, true);
        #endif

        /**
         * This operation applies Subtract operation to specific inputs wrt indices
         * Expected arguments:
         * input: array to be updated
         * indices: array containing indexes for first dimension of input
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_sub)
        DECLARE_OP(scatter_sub, 3, 1, true);
        #endif

        /**
         * This operation applies Multiply operation to specific inputs wrt indices
         * Expected arguments:
         * input: array to be updated
         * indices: array containing indexes for first dimension of input
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_mul)
        DECLARE_OP(scatter_mul, 3, 1, true);
        #endif

        /**
         * This operation applies Divide operation to specific inputs wrt indices
         * Expected arguments:
         * input: array to be updated
         * indices: array containing indexes for first dimension of input
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_div)
        DECLARE_OP(scatter_div, 3, 1, true);
        #endif

        /**
         * This operation applies Assign operation to specific inputs wrt indices
         * Expected arguments:
         * input: array to be updated
         * indices: array containing indexes for first dimension of input
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_upd)
        DECLARE_OP(scatter_upd, 3, 1, true);
        #endif

        /**
         * This operation applies Max operation to specific inputs through given indices
         * Expected arguments:
         * input: array to be updated
         * indices: array containing indexes for first dimension of input
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_max)
        DECLARE_OP(scatter_max, 3, 1, true);
        #endif

        /**
         * This operation applies Min operation to specific inputs through given indices
         * Expected arguments:
         * input: array to be updated
         * indices: array containing indexes for first dimension of input
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_min)
        DECLARE_OP(scatter_min, 3, 1, true);
        #endif

        /**
         * This operation scatter "updates" elements into new output array according to given "indices"
         * Expected arguments:
         * indices: array containing elements/slices indexes of output array to put "updates" elements into, the rest output elements will be zeros
         * updates: array containing elements to be inserted into output array
         * shape: contains shape of output array
         */
        #if NOT_EXCLUDED(OP_scatter_nd)
        DECLARE_CUSTOM_OP(scatter_nd, 3, 1, false, 0, 0);
        #endif

        /**
         * This operation scatter "updates" elements into input array along given "indices"
         * Expected arguments:
         * input: array to be updated
         * indices: array containing elements/slices indexes of input array to put "updates" elements into
         * updates: array containing elements to be inserted into input array
         */
        #if NOT_EXCLUDED(OP_scatter_nd_update)
        DECLARE_OP(scatter_nd_update, 3, 1, true);
        #endif

        /**
         * This operation adds "updates" elements to input array along given "indices"
         * Expected arguments:
         * input: array to be updated
         * indices: array containing elements/slices indexes of input array to add "updates" elements to
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_add)
        DECLARE_OP(scatter_nd_add, 3, 1, true);
        #endif

        /**
         * This operation subtract "updates" elements from input array along given "indices"
         * Expected arguments:
         * input: array to be updated
         * indices: array containing elements/slices indexes of input array to subtract "updates" elements from
         * updates: array containing elements to be interfered with input
         */
        #if NOT_EXCLUDED(OP_scatter_sub)
        DECLARE_OP(scatter_nd_sub, 3, 1, true);
        #endif

        /**
         * This operation takes input's shape, and returns new NDArray filled with specified value
         * Expected arguments:
         * input: N-dimensional array
         *
         * T args:
         * 0: scalar value, used to fill NDArray
         */
        #if NOT_EXCLUDED(OP_fill_as)
        DECLARE_CONFIGURABLE_OP(fill_as, 1, 1, true, 1, 0);
        #endif

        /**
         * This operation applies element-wise rint (round to integral value) operation
         */
        #if NOT_EXCLUDED(OP_rint)
        DECLARE_OP(rint, 1, 1, true);
        #endif

        /**
         * This operation returns unique elements from input array as vector, and their original indices in input array
         * Expected input:
         * input: N-dimensional array
         */
        #if NOT_EXCLUDED(OP_unique)
        DECLARE_CUSTOM_OP(unique, 1, 2, false, 0, 0);
        #endif

        /**
         * This operation returns 3 1D arrays for given 1D array with unique element count and indexes
         * input:
         *     0 - 1D array
         *
         * output:
         *     0 - 1D array with unique values
         *     1 - 1D array with ids for values in array above
         *     2 - 1D array with counts for values in array above
         */
        #if NOT_EXCLUDED(OP_unique_with_counts)
        DECLARE_CUSTOM_OP(unique_with_counts, 1, 3, false, 0, 0);
        #endif

        /**
         * This operation splits input NDArray into multiple TADs along given dimensions
         * Expected arguments:
         * input: N-dimensional array
         *
         * Int args:
         * 0..: TAD axis
         */
        #if NOT_EXCLUDED(OP_tear)
        DECLARE_CUSTOM_OP(tear, 1, -1, false, 0, -1);
        #endif

        /**
         * This op does the same as tear, just uses different input format:
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_unstack)
        DECLARE_CUSTOM_OP(unstack, 1, -1, false, 0, 1);
        #endif

        /**
         * This operation extracts a strided (optionally) slice from a tensor,
         */
        #if NOT_EXCLUDED(OP_strided_slice)
        DECLARE_CUSTOM_OP(strided_slice, 1, 1, false, 0, 5); // TODO: new op type needed. that returns VIEW
        DECLARE_CUSTOM_OP(strided_slice_bp, 2, 1, false, 0, 5);
        #endif

        /**
         * This operation extracts a slice from a tensor.
         *
         */
        #if NOT_EXCLUDED(OP_slice)
        DECLARE_CUSTOM_OP(slice, 1, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(slice_bp, 2, 1, false, 0, -2);
        #endif

        /**
         * This operation generate sequences. Basically from......to, with step used as increment.
         * Expected arguments:
         * start: optional scalar with starting value
         * stop: optional scalar with end value
         * step: optional scalar witn step value
         *
         * Int args: (optional)
         * 0: optional scalar with starting value
         * 1: optional scalar with end value
         * 1: optional scalar witn step value
         *
         * T args: (optional)
         * 0: optional scalar with starting value
         * 1: optional scalar with end value
         * 1: optional scalar witn step value
         */
        #if NOT_EXCLUDED(OP_range)
        DECLARE_CUSTOM_OP(range, -2, 1, false, -2, -2);
        #endif

        /**
         * This operation return one-hot encoded n-dimensional array
         * Expected arguments:
         * input: N-dimensional array
         *
         * T args:
         * 0: 'on' value
         * 1: 'off' value
         *
         * Int args:
         * 0: depth
         * 1: axis
         */
        #if NOT_EXCLUDED(OP_onehot)
        DECLARE_CUSTOM_OP(onehot, 1, 1, false, -2, -2);
        #endif


        /**
         * This operation calculate the confusion matrix for a
         * pair of prediction and label 1-D arrays.
         * Expected arguments:
         * Input arrays:
         *   0 - predictions: 1-D array
         *   1 - labels: 1-D array
         *   2 - weights : optional
         * Int args:
         *   0 - num_classes: optional
         *
         */
        #if NOT_EXCLUDED(OP_confusion_matrix)
        DECLARE_CUSTOM_OP(confusion_matrix, 2, 1, false, 0, -2);
        #endif

        /**
		 * This operation stacks a list of rank tensors into one rank-(R+1) tensor.
		 * Expected arguments:
		 * 0...: N-Dimensional arrays to stack
		 *
		 */
        #if NOT_EXCLUDED(OP_stack)
        DECLARE_CUSTOM_OP(stack, -1, 1, false, 0, 0);
        #endif

        /**
         * This operation returns length of input array
         * Expected arguments:
         * input: N-dimensional array
         *
         * TODO: make this operation reduction, to allow TAD -> size
         */
        #if NOT_EXCLUDED(OP_size)
        DECLARE_CUSTOM_OP(size, 1, 1, false, 0, 0); // add DeclarableScalarOp?
        #endif


        /**
         * This operation returns rank of input array as scalar value.
         */
        #if NOT_EXCLUDED(OP_rank)
        DECLARE_CUSTOM_OP(rank, 1, 1, false, 0, 0); // ^
        #endif


        #if NOT_EXCLUDED(OP_broadcastgradientargs)
        DECLARE_OP(broadcastgradientargs, 2, 2, true);
        #endif

        /**
         * This operation takes input's shape, and returns new NDArray filled with zeros
         * Expected arguments:
         * input: N-dimensional array
         *
         */
        #if NOT_EXCLUDED(OP_zeros_as)
        DECLARE_OP(zeros_as, 1, 1, false);
        #endif

        /**
         * This operation takes input's shape, and returns new NDArray filled with ones
         * Expected arguments:
         * input: N-dimensional array
         *
         */
        #if NOT_EXCLUDED(OP_ones_as)
        DECLARE_OP(ones_as, 1, 1, false);
        #endif

        /**
         * This operation applies element-wise pow(x, 2) to the given input
         * Expected arguments:
         * input: N-Dimensional array
         */
        #if NOT_EXCLUDED(OP_square)
        DECLARE_OP(square, 1, 1, true);
        #endif

        /**
        * This op calculates Hurwitz zeta function zeta(x, q) = sum_{n=0}^{inf} (q + n)^{-x}
        * Implementation is based on Euler-Maclaurin summation formula
        *
        *   Input arrays:
        *   x: define power {-x}, must be > 1, type float.
        *   q: define summand in denominator, must be > 0, type float.
        *
        * Output array:
        *    0: corresponding values of Hurwitz zeta function
        *
        * Two input and one output arrays must have the same shape
        */
        #if NOT_EXCLUDED(OP_zeta)
        DECLARE_CONFIGURABLE_OP(zeta, 2, 1, false, 0, 0);
        #endif

        /**
        * This op calculates polygamma function psi^(n)(x). Implementation is based on serial representation written in
        * terms of the Hurwitz zeta function: polygamma = (-1)^{n+1} * n! * zeta(n+1, x).
        *
        * Input arrays:
        *    0: n - define derivative order (n+1), type integer (however currently is implemented as float casted to integer)
        *    1: x - abscissa points where to evaluate the polygamma function, type float
        *
        * Output array:
        *    0: values of polygamma function at corresponding x, type float
        *
        * Two input and one output arrays have the same shape
        */
        #if NOT_EXCLUDED(OP_polygamma)
        DECLARE_CONFIGURABLE_OP(polygamma, 2, 1, false, 0, 0);
        #endif

        /**
        * This op calculates digamma function psi(x) = derivative of log(Gamma(x))
        *
        * Input arrays:
        *    0: x - abscissa points where to evaluate the digamma function, type float
        *
        * Output array:
        *    0: values of digamma function at corresponding x, type float
        *
        */
        #if NOT_EXCLUDED(OP_digamma)
        DECLARE_CONFIGURABLE_OP(digamma, 1, 1, false, 0, 0);
        #endif

        /**
         * This operation takes shape as first argument, and returns new NDArray filled with specific scalar value.
         * Input arrays:
         * 0 - shape vector
         * 1 - optional scalar NDArray
         *
         * T arguments:
         * 0 - optional scalar value
         *
         */
        #if NOT_EXCLUDED(OP_fill)
        DECLARE_CUSTOM_OP(fill, 1, 1, false, -2, 0);
        #endif

        /**
         * This operation splits given NDArray into chunks of specific size, along given dimension
         * Input arrays:
         * 0 - input array
         * 1 - array of sizes
         * 2 - optional axis
         *
         * Integer arguments:
         * 0 - optional axis
         *
         */
        #if NOT_EXCLUDED(OP_split_v)
        DECLARE_CUSTOM_OP(split_v, 2, -1, false, 0, -2);
        #endif

        /**
         * This operation splits given NDArray into chunks of specific size, along given dimension
         * 0 - input array
         * 1 - optional axis
         *
         * Integer arguments:
         * 0 - number of splits
         * 1 - optional axis
         */
        #if NOT_EXCLUDED(OP_split)
        DECLARE_CUSTOM_OP(split, 1, -1, false, 0, 1);
        #endif


        /**
         * This operation adjusts image hue by delta
         * Input arrays:
         * 0 - input array with rank >= 3, must have at least one dimension equal 3, that is dimension containing channels.
         * 1 - optional argument, input scalar-array containing delta
         *
         * T arguments:
         * 0 - optional argument, delta value
         *
         * Int arguments:
         * 0 - optional argument, corresponds to dimension with 3 channels
         */
        #if NOT_EXCLUDED(OP_adjust_hue)
        DECLARE_CONFIGURABLE_OP(adjust_hue, 1, 1, true, 0, 0);
        #endif

        /**
         * This operation adjusts image saturation by delta
         * Input arrays:
         * 0 - input array with rank >= 3, must have at least one dimension equal 3, that is dimension containing channels.
         * 1 - optional argument, input scalar-array containing saturation factor
         *
         * T arguments:
         * 0 - optional argument, saturation factor
         *
         * Int arguments:
         * 0 - optional argument, corresponds to dimension with 3 channels
         */
        #if NOT_EXCLUDED(OP_adjust_saturation)
        DECLARE_CONFIGURABLE_OP(adjust_saturation, 1, 1, true, 0, 0);
        #endif

        /**
         * This operation adjusts image contrast by given factor ( z = (x - mean) * factor + mean )
         * Input arrays:
         * 0 - input array with rank >= 3, must have last one dimension equal 3, that is dimension containing channels.
         * 1 - optional argument, input scalar-array containing saturation contrast factor
         *
         * T arguments:
         * 0 - optional argument, contrast factor
         *
         */
        #if NOT_EXCLUDED(OP_adjust_contrast)
        DECLARE_CONFIGURABLE_OP(adjust_contrast, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(adjust_contrast_v2, 1, 1, true, 0, 0);
        #endif




        /**
         * This operation rearranges data from depth into blocks of spatial data. This is the reverse transformation
         * of space_to_depth op. This op output is a copy of the input tensor where values from the depth dimension
         * are moved in spatial blocks to the height and width dimensions. Int attr 0 indicates the input
         * block size and how the data is moved.
         * Input:
         *     0 - 4D tensor on given type
         * Output:
         *     0 - 4D tensor of given type and proper shape
         *
         * Int arguments:
         *     0 - block size
         *     1 - output data format: 0 ("NHWC"): shape{ batch, height, width, channels }
         *                             1 ("NCHW"): shape{ batch, channels, height, width }
         *                             2 ("NCHW_VECT_C"): int8 shape{ batch, channels / 4, height, width, 4 }
         *                             optional (default 0)
         */
        #if NOT_EXCLUDED(OP_depth_to_space)
        DECLARE_CUSTOM_OP(depth_to_space, 1, 1, false, 0, -1);
        #endif

        /**
         * This operation rearranges blocks of spatial data, into depth.This op output is a copy of the input tensor
         * where values from the height and width dimensions are moved to the depth dimension. Int attr 0 indicates
         * the input block size.
         *
         * Input:
         *     - 4D tensor of given type
         * Output:
         *     - 4D tensor
         *
         * Int arguments:
         *     0 - block size
         *     1 - output data format: 0 ("NHWC"): shape{ batch, height, width, channels }
         *                             1 ("NCHW"): shape{ batch, channels, height, width }
         *                             2 ("NCHW_VECT_C"): int8 shape{ batch, channels / 4, height, width, 4 }
         *                             optional (default 0)
         *
         */
        #if NOT_EXCLUDED(OP_space_to_depth)
        DECLARE_CUSTOM_OP(space_to_depth, 1, 1, false, 0, -1);
        #endif

        /**
         * This op calculates cross-product between input arguments
         * Input arguments
         * 0 - vector or tensor A
         * 1 - vector or tensor B
         */
        #if NOT_EXCLUDED(OP_cross)
        DECLARE_OP(cross, 2, 1, false);
        #endif

        /**
         * Zero-pads and then rearranges (permutes) blocks of spatial data into batch. More specifically, this op
         * outputs a copy of the input tensor where values from the height and width dimensions are moved to the
         * batch dimension. After the zero-padding, both height and width of the input must be divisible by the block
         * size.
         *
         * Inputs:
         *  0 - input tensor
         *  1 - 2D paddings tensor (shape {M, 2})
         *
         *  Output:
         *    - result tensor
         *
         *  Int args:
         *      0 - block size (M)
         *
         */
        #if NOT_EXCLUDED(OP_space_to_batch)
        DECLARE_CUSTOM_OP(space_to_batch, 2, 1, false, 0, 1);
        #endif

        /*
         * This operation divides "spatial" dimensions [1, ..., M] of the input into a grid of blocks of shape
         * block_shape, and interleaves these blocks with the "batch" dimension (0) such that in the output,
         * the spatial dimensions [1, ..., M] correspond to the position within the grid, and the batch dimension
         * combines both the position within a spatial block and the original batch position. Prior to division into
         * blocks, the spatial dimensions of the input are optionally zero padded according to paddings.
         *
         * Inputs:
         *      0 - input (N-D tensor)
         *      1 - block_shape - int 1D tensor with M length
         *      2 - paddings - int 2D tensor with shape {M, 2}
         *
         * Output:
         *      - N-D tensor with the same type as input 0.
         *
         * */
        #if NOT_EXCLUDED(OP_space_to_batch_nd)
        DECLARE_CUSTOM_OP(space_to_batch_nd, 3, 1, false, 0, 0);
        #endif

        /**
         *
         *
         */
        #if NOT_EXCLUDED(OP_batch_to_space)
        DECLARE_CUSTOM_OP(batch_to_space, 2, 1, false, 0, 1);
        #endif
        #if NOT_EXCLUDED(OP_batch_to_space_nd)
        DECLARE_CUSTOM_OP(batch_to_space_nd, 3, 1, false, 0, 0);
        #endif

        /**
         * top_k operation returns a vector of k top values for
         *  given NDArray as tensor with default boolean (true)
         *  as sort for result index array
         *  will be sorted by the values in descending order.
         *  The first parameter is a NDArray for working.
         *  The second is k (default 1) - optional
         *  The third is boolean value(default is true) (0 - as is, 1 - sorted by value) optional
         */
        #if NOT_EXCLUDED(OP_top_k)
        DECLARE_CUSTOM_OP(top_k, 1, 2, false, 0, -1);
        #endif

        /**
         * in_top_k operation returns a vector of k boolean values for
         *  given NDArray as 2D matrix of predicted in the NDArray k top values
         *  The first parameter is a NDArray of predicted values (2d array).
         *  The second is NDArray as vector of indeces k top values will be search.
         *  The third is k
         */
        #if NOT_EXCLUDED(OP_in_top_k)
        DECLARE_CUSTOM_OP(in_top_k, 2, 1, true, 1, 1);
        #endif

        /**
         * moments operation calculate a mean and variation for given NDArray
         * with reduce a result according to axis array given.
         * For full axis the result is both mean and variance of all members in array.
         * Otherwise there are two NDArrays with means and variances for
         * Axes can be put as the second NDArray or as int vector.
         *
         * the optional flag "keep_dims" can be set as T param
         */
        #if NOT_EXCLUDED(OP_moments)
        DECLARE_CUSTOM_OP(moments, 1, 2, false, 0, -2);
        #endif

        /**
         * embedding_lookup - search for submatrices in given matrix and retunts them
         * accordingly to index array given.
         */
        #if NOT_EXCLUDED(OP_embedding_lookup)
        DECLARE_CUSTOM_OP(embedding_lookup, 2, 1, false, 0, 1);
        #endif

        /**
         * dynamic_partition - partition a input tensor onto num_partitions
         * accordingly to index array given.
         *
         * the first param - NDArray to be partitioned.
         * the second param - index array
         * the third param (integer param) - num or partitions.
         *
         * returns a num of NDArrays as output
         */
        #if NOT_EXCLUDED(OP_dynamic_partition)
        DECLARE_CUSTOM_OP(dynamic_partition, 2, 1, false, 0, 1);
        #endif

        #if NOT_EXCLUDED(OP_dynamic_partition_bp)
        DECLARE_CUSTOM_OP(dynamic_partition_bp, 3, 2, false, 0, 1);
        #endif

        /**
         * dynamic_stitch - merge partitions from the second param a input tensor
         * into a single tensor accordingly to index array given.
         *
         * the first param - index array
         * the second params - tensors to be merged
         *
         * returns a num of NDArrays as output
         *
         * the operation is inversion od dynamic_partition
         */
        #if NOT_EXCLUDED(OP_dynamic_stitch)
        DECLARE_CUSTOM_OP(dynamic_stitch, 2, 1, false, 0, 0);
        #endif

        /**
         * zero_fraction op.
         * compute a fraction of zeros in given array
         *
         * input param - an array (tensor)
         * output value - a real number with given type (e.g. float or double)
         */
        #if NOT_EXCLUDED(OP_zero_fraction)
        DECLARE_CUSTOM_OP(zero_fraction, 1, 1, false, 0, 0);
        #endif

        /**
         * xw_plus_b op.
         * multiply two first matrices and add third vector to each row of result
         *
         * input params:
         *   - 2D matrix NxM
         *   - 2D matrix MxN
         *   - 1D vector with N elements
         * output value - 2D matrix NxN as multiply of matrixes and add vector
         */
        #if NOT_EXCLUDED(OP_xw_plus_b)
        DECLARE_CUSTOM_OP(xw_plus_b, 3, 1, false, 0, 0);
        #endif

        /**
         * This operation is missed due it simplicy.
         * Input and output params are the same after operation.
         * Input - NDArray, output - NDArray with the same shape.
         */
        #if NOT_EXCLUDED(OP_stop_gradient)
        DECLARE_OP(stop_gradient, 1, 1, true);
        #endif

        #if NOT_EXCLUDED(OP_parallel_stack)
        DECLARE_CUSTOM_OP(parallel_stack, -1, 1, false, 0, 0);
        #endif

        /**
         * normalize_moments operation normalize already calculated mean and variation
         * accordingly to shift and count.
         * input params:
         *  - count of data
         *  - tensor with mean
         *  - tensor with variance (the same shape as before)
         *
         *  - optional floating point param shift.
         *
         *  returns a normalized pair mean and variance with the same shapes as input
         */
        #if NOT_EXCLUDED(OP_normalize_moments)
        DECLARE_CUSTOM_OP(normalize_moments, 3, 2, false, 1, 0);
        #endif

        /**
         * sufficient_statistics operation return calculated mean and variation with data count.
         * this operation is invert for moments
         * accordingly to shift and count.
         * input params:
         *  - input tensor
         *  - axes vector
         *
         *
         *  - optional floating point param shift.
         *  - optional int (as bool) keep_dimension
         *
         *  returns four tensors:
         *     - scalar tensor (data count)
         *     - sum elements of input (accross axises)
         *     - sum of squares of input (accross axises)
         *     - shift (if was given by input floating param)
         */
        #if NOT_EXCLUDED(OP_sufficient_statistics)
        DECLARE_CUSTOM_OP(sufficient_statistics, 2, 3, false, 0, 0);
        #endif

        /**
         * This op calculates weighted logarithmic loss of input
         * Input arguments
         *  0 - target
         *  1 - input
         *  2 - weights (scalar or vector with same as last dimension)
         *
         *  return value - a tensor with the same shape as target or input
         */
        #if NOT_EXCLUDED(OP_weighted_cross_entropy_with_logits)
        DECLARE_OP(weighted_cross_entropy_with_logits, 3, 1, true);
        #endif

        /**
         * This op calculates dropout of input
         * Input arguments
         *  0 - input tensor
         *  1 - noise_shape - (vector with shape to reduce) - optional
         *
         *  int parameter - seed for random numbers
         *  T parameter - probability (should be between 0 and 1)
         *  return value - a tensor with the same shape as target or input
         */
        #if NOT_EXCLUDED(OP_dropout)
        DECLARE_CONFIGURABLE_OP(dropout, 1, 1, true, 1, 1);
        #endif
        #if NOT_EXCLUDED(OP_dropout_bp)
        DECLARE_CONFIGURABLE_OP(dropout_bp, 2, 1, false, 1, 1);
        #endif

        /*  Calculates alpha weighted dropout
            T params:
                0 - drop probability
                1 - alpha value
                2 - alpha' value
                3 - beta value
         */
        #if NOT_EXCLUDED(OP_alpha_dropout_bp)
        DECLARE_CONFIGURABLE_OP(alpha_dropout_bp, 2, 1, false, 4, 1);
        #endif


        /**
         * bincount operation return a vector with element counted.
         *
         * input params:
         *  - input tensor - only int part are accepted
         *  - weights - the same shape tensor with integer weights for element (optional)
         *  default weight - 1,1,1..,1 for all values in the tensor
         *
         *  optional ints:
         *  - min_length - zero or greater
         *  - max_length - between min_length and max(input) + 1
         *
         *  returns four tensors:
         *     - vector tensor with length to min(max_len, max(input) + 1) with count
         *  of values in indexed place
         *
         */
        #if NOT_EXCLUDED(OP_bincount)
        DECLARE_CUSTOM_OP(bincount, 1, 1, false, 0, 0);
        #endif

        /**
         * broadcast_dynamic_shape op.
         *
         * input params:
         *    0 - the first shape (vector with shape)
         *    1 - the second shape (vector with shape)
         *
         * return value:
         *    vector with broadcasted shape
         */
        #if NOT_EXCLUDED(OP_broadcast_dynamic_shape)
        DECLARE_CUSTOM_OP(broadcast_dynamic_shape, 2, 1, false, 0, 0);
        #endif

        /**
         * matrix_determinant op.
         *
         * input params:
         *    0 - the tensor with dimension (x * y * z * ::: * M * M)
         *
         * return value:
         *    tensor with dimension (x * y * z * ::: *) with determinant for all
         * M x M matricies
         */
        #if NOT_EXCLUDED(OP_matrix_determinant)
        DECLARE_CUSTOM_OP(matrix_determinant, 1, 1, false, 0, 0);
        #endif

        /**
         * log_matrix_determinant op.
         *
         * input params:
         *    0 - the tensor with dimension (x * y * z * ::: * M * M)
         *
         * return value:
         *    tensor with dimension (x * y * z * ::: *) with log determinant for all
         * M x M matricies
         */

        #if NOT_EXCLUDED(OP_log_matrix_determinant)
        DECLARE_CUSTOM_OP(log_matrix_determinant, 1, 1, false, 0, 0);
        #endif

        /**
         * logdet op. Logarithm of the determinant of hermitian positive matricies.
         *
         * input params:
         *    0 - the tensor with dimension (x * y * z * ::: * M * M)
         *
         * return value:
         *    tensor with dimension (x * y * z * ::: *) with log determinant for all
         * M x M matricies
         */

        #if NOT_EXCLUDED(OP_logdet)
        DECLARE_CUSTOM_OP(logdet, 1, 1, false, 0, 0);
        #endif

        /**
         * matrix_inverse op. - make inverse for all 2D square matricies found in the input tensor
         *
         * input params:
         *    0 - the tensor with dimension (x * y * z * ::: * M * M)
         *
         * return value:
         *    tensor with dimension (x * y * z * ::: * M * M) with inverse M x M matricies in it
         */
        #if NOT_EXCLUDED(OP_matrix_inverse)
        DECLARE_OP(matrix_inverse, 1, 1, true);
        #endif

        /**
         * sequence_mask op. - make mask for given tensor filled by (j > x[i_1, i_2,...,i_n]) -> z[i_1, i_2,...,i_n,j]
         *
         * input params:
         *    0 - the ND-tensor filled by integer-like values
         *
         * optional int param - maxlength (maxlength >= max(x)). By default maxlength = max(x).
         * return value:
         *    (N+1)D tensor filled by 0 and 1 accordingly the mask
         */
        #if NOT_EXCLUDED(OP_sequence_mask)
        DECLARE_CUSTOM_OP(sequence_mask, 1, 1, false, 0, 0);
        #endif
        /**
         * segment_max op. - make a tensor filled by max values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * return value:
         *    tensor with max values according to indices sets.
         */

        #if NOT_EXCLUDED(OP_segment_max)
        DECLARE_CUSTOM_OP(segment_max, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_segment_max_bp)
        DECLARE_CUSTOM_OP(segment_max_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * segment_min op. - make a tensor filled by min values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * return value:
         *    tensor with min values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_segment_min)
        DECLARE_CUSTOM_OP(segment_min, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_segment_min_bp)
        DECLARE_CUSTOM_OP(segment_min_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * segment_sum op. - make a tensor filled by sum of values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * return value:
         *    tensor with sum of values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_segment_sum)
        DECLARE_CUSTOM_OP(segment_sum, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_segment_sum_bp)
        DECLARE_CUSTOM_OP(segment_sum_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * segment_prod op. - make a tensor filled by product of values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * return value:
         *    tensor with product of values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_segment_prod)
        DECLARE_CUSTOM_OP(segment_prod, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_segment_prod_bp)
        DECLARE_CUSTOM_OP(segment_prod_bp, 3, 2, false, 0, 0);
        #endif
        /**
         * segment_mean op. - make a tensor filled by average of values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * return value:
         *    tensor with average of values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_segment_mean)
        DECLARE_CUSTOM_OP(segment_mean, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_segment_mean_bp)
        DECLARE_CUSTOM_OP(segment_mean_bp, 3, 2, false, 0, 0);
        #endif

        /**
         * unsorted_segment_max op. - make a tensor filled by max values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * return value:
         *    tensor with max values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_unsorted_segment_max)
        DECLARE_CUSTOM_OP(unsorted_segment_max, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_unsorted_segment_max_bp)
        DECLARE_CUSTOM_OP(unsorted_segment_max_bp, 3, 2, false, 0, 1);
        #endif

        /**
         * unsorted_segment_min op. - make a tensor filled by min values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * integer param:
         *    0 - num of segments
         *
         * return value:
         *    tensor with min values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_unsorted_segment_min_bp)
        DECLARE_CUSTOM_OP(unsorted_segment_min, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_unsorted_segment_min_bp)
        DECLARE_CUSTOM_OP(unsorted_segment_min_bp, 3, 2, false, 0, 1);
        #endif

        /**
         * unsorted_segment_sum op. - make a tensor filled by sum of values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * integer param:
         *    0 - num of segments
         *
         * return value:
         *    tensor with sum of values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_unsorted_segment_sum)
        DECLARE_CUSTOM_OP(unsorted_segment_sum, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_unsorted_segment_sum_bp)
        DECLARE_CUSTOM_OP(unsorted_segment_sum_bp, 3, 2, false, 0, 1);
        #endif

        /**
         * unsorted_segment_prod op. - make a tensor filled by product of values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * integer param:
         *    0 - num of segments
         *
         * return value:
         *    tensor with product of values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_unsorted_segment_prod)
        DECLARE_CUSTOM_OP(unsorted_segment_prod, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_unsorted_segment_prod_bp)
        DECLARE_CUSTOM_OP(unsorted_segment_prod_bp, 3, 2, false, 0, 1);
        #endif

        /**
         * unsorted_segment_mean op. - make a tensor filled by average of values according to index tensor given.
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * integer param:
         *    0 - num of segments
         *
         * return value:
         *    tensor with average of values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_unsorted_segment_mean)
        DECLARE_CUSTOM_OP(unsorted_segment_mean, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_unsorted_segment_mean_bp)
        DECLARE_CUSTOM_OP(unsorted_segment_mean_bp, 3, 2, false, 0, 1);
        #endif

        /**
         * unsorted_segment_sqrt_n op. - computes the sum along segments of a tensor divided by the sqrt(N).
         *
         * input params:
         *    0 - the tensor with data;
         *    1 - the tensor with indices.
         *
         * integer param:
         *    0 - num of segments
         *
         * return value:
         *    tensor with average of values according to indices sets.
         */
        #if NOT_EXCLUDED(OP_unsorted_segment_sqrt)
        DECLARE_CUSTOM_OP(unsorted_segment_sqrt_n, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_unsorted_segment_sqrt_n_bp)
        DECLARE_CUSTOM_OP(unsorted_segment_sqrt_n_bp, 3, 2, false, 0, 1);
        #endif

        /**
         * extract_image_patches op - Extract patches from images and put them in the "depth" output dimension.
         *
         * input params:
         *    0 - images tensor (4D)
         *
         * int params:
         *    0 - ksize_rows
         *    1 - ksize_cols
         *    2 - strides_rows
         *    3 - strides_cols
         *    4 - rates_rows
         *    5 - rates_cols
         *    6 - padding_type - 0 - equiv 'VALID', 1 - 'SAME'
         */
        #if NOT_EXCLUDED(OP_extract_image_patches)
        DECLARE_CUSTOM_OP(extract_image_patches, 1, 1, false, 0, 7);
        #endif

        /**
         * draw_bounding_boxes op - modified input image with given colors exept given boxes.
         *
         * input params:
         *    0 - images tensor (4D) with shape {batch, width, height, channels}, where channes is 1 (BW image),
         * 3 (RGB) or 4 (RGBA)
         *    1 - boxes tensor (3D) with shape {batch, number_of_boxes, 4} where last dimension encoded as
         * (y_min, x_min, y_max, x_max), all values in between 0. and 1.
         *    2 - colours tensor (2D) with shape {number_of_boxes, channels} -- bordering color set (palette)
         *
         * output:
         *    0 - 4D tensor with same shape as images (input 0)
         */
        #if NOT_EXCLUDED(OP_draw_bounding_boxes)
        DECLARE_OP(draw_bounding_boxes, 3, 1, true);
        #endif

        /**
         * roll - op porting from numpy (https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.roll.html)
         *
         * input params:
         *    0 - NDArray
         *
         * int params:
         *    0 - shift
         *    1 - axe 1
         *    2 - axe 2
         *    ...
         *    N - axe N
         *
         *    All axes are optional and should be between 0 and input->rankOf(). Of course, all axes can be repeated.
         *
         * output:
         *    0 - NDArray with the same shape as input.
         */
        #if NOT_EXCLUDED(OP_roll)
        DECLARE_CONFIGURABLE_OP(roll, 1, 1, true, 0, 1);
        #endif

        /**
         * lin_space - op porting from TF (https://www.tensorflow.org/api_docs/python/tf/lin_space)
         *
         * input params:
         *    0 - startVal - NDArray scalar (float point)
         *    1 - finishVal - NDArray scalar (float point)
         *    2 - numOfElements - NDArray scalar (integer)
         *
         * output:
         *    0 - 1D NDArray with the same type as input and length as given with numOfElements param.
         */
        #if NOT_EXCLUDED(OP_lin_space)
        DECLARE_CUSTOM_OP(lin_space, 3, 1, false, 0, 0);
        #endif

        /**
         * reduction_sum - tf.reduction_sum operation
         *
         * input params:
         *    0 - NDArray
         *
         * T_ARG param (optional):
         * 0 - keep_dims != 0.
         *
         * int params (optional):
         *    0 - axe 1
         *    1 - axe 2
         *    ...
         *    N-1 axe N
         *
         *    All axes are optional and should be between 0 and input->rankOf() - 1
         *
         * output:
         *    0 - NDArray with reduces shape accordingly to axes (the scalar in default case).
         */
        #if NOT_EXCLUDED(OP_reduce_sum)
        DECLARE_CUSTOM_OP(reduce_sum, 1, 1, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_reduce_sum_bp)
        DECLARE_CUSTOM_OP(reduce_sum_bp, 2, 1, false, 0, 0);
        #endif

        /**
         * reduction_prod - tf.reduction_prod operation
         *
         * input params:
         *    0 - NDArray
         *
         * T_ARG param (optional):
         * 0 - keep_dims != 0.
         *
         * int params (optional):
         *    0 - axe 1
         *    1 - axe 2
         *    ...
         *    N-1 axe N
         *
         *    All axes are optional and should be between 0 and input->rankOf() - 1
         *
         * output:
         *    0 - NDArray with reduces shape accordingly to axes (the scalar in default case).
         */
        #if NOT_EXCLUDED(OP_reduce_prod)
        DECLARE_CUSTOM_OP(reduce_prod, 1, 1, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_reduce_prod_bp)
        DECLARE_CUSTOM_OP(reduce_prod_bp, 2, 1, false, 0, 0);
        #endif

       /**
        * This op calculates min of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate mins for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate min along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated mins
        */
        #if NOT_EXCLUDED(OP_reduce_min)
        DECLARE_CUSTOM_OP(reduce_min, 1, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_reduce_min_bp)
        DECLARE_CUSTOM_OP(reduce_min_bp, 2, 1, false, 0, 0);
        #endif

       /**
        * This op calculates max of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate maxes for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate max along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated maxes
        */
        #if NOT_EXCLUDED(OP_reduce_max)
        DECLARE_CUSTOM_OP(reduce_max, 1, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_reduce_max_bp)
        DECLARE_CUSTOM_OP(reduce_max_bp, 2, 1, false, 0, 0);
        #endif

       /**
        * This op calculates norm1 of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate norm1 for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate norm1 along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated norm1
        */
        #if NOT_EXCLUDED(OP_reduce_norm1)
        DECLARE_CUSTOM_OP(reduce_norm1, 1, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_reduce_norm1_bp)
        DECLARE_CUSTOM_OP(reduce_norm1_bp, 2, 1, false, 0, 0);
        #endif

       /**
        * This op calculates norm2 of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate norm2 for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate norm2 along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated norm2
        */
        #if NOT_EXCLUDED(OP_reduce_norm2)
        DECLARE_CUSTOM_OP(reduce_norm2, 1, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_reduce_norm2_bp)
        DECLARE_CUSTOM_OP(reduce_norm2_bp, 2, 1, false, 0, 0);
        #endif


       /**
        * This op calculates squared norm of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate squared norm for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate squared norm along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated norm
        */
        #if NOT_EXCLUDED(OP_reduce_sqnorm)
        DECLARE_CUSTOM_OP(reduce_sqnorm, 1, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_reduce_sqnorm_bp)
        DECLARE_CUSTOM_OP(reduce_sqnorm_bp, 2, 1, false, 0, 0);
        #endif

       /**
        * This op calculates norm max of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate norm max for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate norm max along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated norm
        */
        #if NOT_EXCLUDED(OP_reduce_norm_max)
        DECLARE_CUSTOM_OP(reduce_norm_max, 1, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_reduce_norm_max_bp)
        DECLARE_CUSTOM_OP(reduce_norm_max_bp, 2, 1, false, 0, 0);
        #endif

        /**
        * This op calculates mean of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate mean for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate mean along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated means
        */
        #if NOT_EXCLUDED(OP_reduce_mean)
        DECLARE_CUSTOM_OP(reduce_mean, 1, 1, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_reduce_mean_bp)
        DECLARE_CUSTOM_OP(reduce_mean_bp, 2, 1, false, 0, 0)
        #endif
        /**
        * This op calculates sample variance of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate mean for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *   biasCorrected -  if non zero, then bias correction will be applied, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate mean along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated means
        */
        DECLARE_CUSTOM_OP(reduce_variance, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(reduce_variance_bp, 2, 1, false, 0, 0)

        /**
        * This op calculates sample standard deviation of elements along given dimensions
        *
        * input array:
        *    x: tensor to calculate mean for
        *
        * float arguments:
        *   keepDims: if non zero, then keep reduced dimensions with length = 1, default value is zero
        *   biasCorrected - if non zero, then bias correction will be applied, default value is zero
        *
        * int arguments:
        *    list of integers - dimensions to calculate mean along, default corresponds to empty list in which case calculation is performed for all dimensions and scalar is returned
        *
        * output array:
        *    reduced tensor with calculated means
        */
        DECLARE_CUSTOM_OP(reduce_stdev, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(reduce_stdev_bp, 2, 1, false, 0, 0)

        /**
        * This op calculates backprop dot for two tensors along given dimensions
        *
        * input array:
        *    x: tensor to calculate dot for
        *    y: tensor to calculate dot for
        *    z: tensor with gradient output of the FF dot for x and y
        *
        * int arguments:
        *   list of integers - dimensions to calculate dot along,
        *   default corresponds to empty list in which case calculation
        *   is performed for all dimensions and scalar is returned.
        *
        * output array:
        *   the tensor with calculated backproped dots
        *
        */

        #if NOT_EXCLUDED(OP_reduce_dot_bp)
        DECLARE_CUSTOM_OP(reduce_dot_bp, 3, 2, false, 0, 0);
        #endif
        /**
         * reduce_logsumexp - tf.reduce_logsumexe operation
         *
         * input params:
         *    0 - NDArray (input)
         *    1 - 1D NDArray (axis) (optional) - integer array
         *
         * T_ARG param (optional):
         * 0 - keep_dims != 0.
         *
         * int params (optional):
         *    0 - axe 1
         *    1 - axe 2
         *    ...
         *    N-1 axe N
         *
         *  CAUTION: All axes are optional and should be between 0 and input->rankOf() - 1
         *  and put either with second param or as integers but not both
         *
         * output:
         *    0 - NDArray with reduces shape accordingly to axes (the scalar in default case).
         */
        #if NOT_EXCLUDED(OP_reduce_logsumexp)
        DECLARE_CUSTOM_OP(reduce_logsumexp, 1, 1, false, 0, 0);
        #endif

       /**
        * This op make bilinear or nearest neighbor interpolated resize for given tensor
        *
        * input array:
        *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels) numeric type
        *    1 - 2D-Tensor with shape (num_boxes, 4) float type
        *    2 - 1D-Tensor with shape (num_boxes) int type
        *    3 - 1D-Tensor with 2 values (newWidth, newHeight) (optional) int type
        *
        * float arguments (optional)
        *   0 - exprapolation_value (optional) default 0.f
        *
        * int arguments: (optional)
        *   0 - mode (default 0 - bilinear interpolation)
        *
        * output array:
        *   the 4D-Tensor with resized to crop_size images given - float type
        */
        #if NOT_EXCLUDED(OP_crop_and_resize)
        DECLARE_CUSTOM_OP(crop_and_resize, 4, 1, false, -1, -1);
        #endif

       /**
        * This op make bilinear interpolated resize for given tensor
        *
        * input array:
        *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
        *    1 - 1D-Tensor with 2 values (newWidth, newHeight) (optional)
        *
        * int arguments: (optional)
        *   0 - new width
        *   1 - new height
        *
        * output array:
        *   the 4D-Tensor with calculated backproped dots
        *
        * CAUTION: either size tensor or a pair of int params should be provided.
        */

        #if NOT_EXCLUDED(OP_resize_bilinear)
        DECLARE_CUSTOM_OP(resize_bilinear, 1, 1, false, 0, -2);
        #endif

       /**
        * This op make nearest neighbor interpolated resize for given tensor
        *
        * input array:
        *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
        *    1 - 1D-Tensor with 2 values (newWidth, newHeight) (optional)
        *
        * int arguments: (optional)
        *   0 - new width
        *   1 - new height
        *
        * output array:
        *   the 4D-Tensor with resized image (shape is {batch, newWidth, newHeight, channels})
        *
        * CAUTION: either size tensor or a pair of int params should be provided.
        */

        #if NOT_EXCLUDED(OP_resize_nearest_neighbor)
        DECLARE_CUSTOM_OP(resize_nearest_neighbor, 1, 1, false, 0, -2);
        #endif

       /**
        * This op make bicubic interpolated resize for given tensor
        *
        * input array:
        *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
        *    1 - 1D-Tensor with 2 values (newWidth, newHeight)
        *
        * output array:
        *   the 4D-Tensor with resized image (shape is {batch, newWidth, newHeight, channels})
        *
        */
        #if NOT_EXCLUDED(OP_resize_bicubic)
        DECLARE_CUSTOM_OP(resize_bicubic, 1, 1, false, 0, -2);
        #endif

       /**
        * This op make interpolated resize for given tensor with given algorithm.
        * Supported algorithms are bilinear, bicubic, nearest_neighbor.
        * Need to implement to full compatibility with TF: lanczos5, gaussian, area and mitchellcubic
        *
        * input array:
        *    0 - 4D-Tensor with shape (batch, sizeX, sizeY, channels)
        *    1 - 1D-Tensor with 2 values (newWidth, newHeight)
        *
        * optional int args:
        *    0 - algorithm - bilinear by default
        * optional bool args:
        *    0 - preserve_aspect_ratio - default False
        *    1 - antialias - default False
        *
        * output array:
        *   the 4D-Tensor with resized by given algorithm image (shape is {batch, newWidth, newHeight, channels})
        *
        */

        #if NOT_EXCLUDED(OP_image_resize)
        DECLARE_CUSTOM_OP(image_resize, 2, 1, false, 0, 0);
        #endif

       /**
        * Copy a tensor setting everything outside a central band in each innermost matrix
        *
        * input array:
        *    x: given tensor with shape {..., M, N} - as vector (matrix) of matricies MxN
        *
        * int arguments:
        *   lower band
        *   upper band
        *
        * output array:
        *   matrix with given bands between lower and upper diagonals
        *
        */

        #if NOT_EXCLUDED(OP_matrix_band_part)
        DECLARE_CONFIGURABLE_OP(matrix_band_part, 1, 1, true, 0, 2);
        #endif


        #if NOT_EXCLUDED(OP_Assert)
        DECLARE_OP(Assert, 1, 1, false);
        #endif

        /**
         * image.non_max_suppression ops.
         * input:
         *     0 - boxes - 2D-tensor with shape (num_boxes, 4) by float type
         *     1 - scales - 1D-tensor with shape (num_boxes) by float type
         *     2 - output_size - 0D-tensor by int type (optional)
         * float args:
         *     0 - overlap_threshold - threshold value for overlap checks (optional, by default 0.5)
         *     1 - score_threshold - the threshold for deciding when to remove boxes based on score (optional, by default -inf)
         * int args:
         *     0 - output_size - as arg 2 used for same target. Eigher this or arg 2 should be provided.
         *
         * output:
         *     - vector with size M, where M <= output_size by int type
         *
         * */
        #if NOT_EXCLUDED(OP_image_non_max_suppression)
        DECLARE_CUSTOM_OP(non_max_suppression, 2, 1, false, 0, 0);
        #endif
        #if NOT_EXCLUDED(OP_image_non_max_suppression_v3)
                DECLARE_CUSTOM_OP(non_max_suppression_v3, 2, 1, false, 0, 0);
        #endif

        /*
         * image.non_max_suppression_overlaps op.
         * input:
         *     0 - boxes - 2D-tensor with shape (num_boxes, 4) by float type
         *     1 - scales - 1D-tensor with shape (num_boxes) by float type
         *     2 - output_size - 0D-tensor by int type (optional)
         * float args:
         *     0 - overlap_threshold - threshold value for overlap checks (optional, by default 0.5)
         *     1 - score_threshold - the threshold for deciding when to remove boxes based on score (optional, by default -inf)
         * int args:
         *     0 - output_size - as arg 2 used for same target. Eigher this or arg 2 should be provided.
         *
         * output:
         *     0 - 1D integer tensor with shape [M], epresenting the selected indices from the overlaps tensor, where M <= max_output_size
         * */
        #if NOT_EXCLUDED(OP_image_non_max_suppression_overlaps)
        DECLARE_CUSTOM_OP(non_max_suppression_overlaps, 2, 1, false, 0, 0);
        #endif

        /*
         * cholesky op - decomposite positive square symetric matrix (or matricies when rank > 2).
         * input:
         *     0 - matricies - tensor with shape (..., N, N) by float type
         *
         * output - lower triangular matrix (matricies when rank > 2) with the same shape as input.
         * */
        #if NOT_EXCLUDED(OP_cholesky)
        DECLARE_OP(cholesky, 1, 1, true);
        #endif
        /*
         * nth_element - apply nth_element for last dimension of input tensor
         * input array:
         *     0 - input array
         *     1 - scalar tensor with n for operation. n should be less than last dimension
         *
         * output:
         *    0 - NDArray with the same shape as input
         */
        #if NOT_EXCLUDED(OP_nth_element)
        DECLARE_CUSTOM_OP(nth_element, 2, 1, false, 0, 0);
        #endif

        /**
         * This op checks for Inf/NaN values within input array, and throws exception if there's at least one
         */
        #if NOT_EXCLUDED(OP_check_numerics)
        DECLARE_CUSTOM_OP(check_numerics, 2, 1, true, 0, 0);
        #endif
/**
         * fake_quant_with_min_max_vals - tf.quantization.fake_quant_with_min_max_vars
         *
         * input params:
         *    0 - NDArray (input)
         *    1 - 0D Tensor - min value
         *    2 - 0D Tensor - max value
         *
         * int params (optional):
         *    0 - num_bits (allowed interval [2, 16], default 8)
         *    1 - narrow_range (default False)
         *
         * output:
         *    0 - NDArray with the same shape as input
         */
        #if NOT_EXCLUDED(OP_fake_quant_with_min_max_vars)
        DECLARE_CONFIGURABLE_OP(fake_quant_with_min_max_vars, 3, 1, true, 0, -2);
        #endif

/**
         * fake_quant_with_min_max_vals_per_channel - tf.quantization.fake_quant_with_min_max_vars_per_channel
         *
         * input params:
         *    0 - NDArray (input) - at least 2D.
         *    1 - 1D Tensor - min values (min length equals to last dim of input)
         *    2 - 1D Tensor - max value (length equals to min)
         *
         * int params (optional):
         *    0 - num_bits (allowed interval [2, 16], default 8)
         *    1 - narrow_range (default False)
         *
         * output:
         *    0 - NDArray with the same shape as input
         */
        #if NOT_EXCLUDED(OP_fake_quant_with_min_max_vars_per_channel)
                DECLARE_CONFIGURABLE_OP(fake_quant_with_min_max_vars_per_channel, 3, 1, true, 0, -2);
        #endif

        /**
         * compare_and_bitpack - compare with greater and pack result with uint8
         *
         * input params:
         *    0 - NDArray (input)
         *    1 - 0D Tensor - threshold
         *
         *
         * output:
         *    0 - NDArray with the same shape as input and type uint8
         */
        #if NOT_EXCLUDED(OP_compare_and_bitpack)
        DECLARE_CUSTOM_OP(compare_and_bitpack, 2, 1, false, 0, 0);
        #endif
    }
}

#endif
