/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.shape;

import com.google.common.primitives.Ints;
import org.nd4j.bytebuddy.shape.IndexMapper;
import org.nd4j.bytebuddy.shape.ShapeMapper;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * Encapsulates all shape related logic (vector of 0 dimension is a scalar is equivalent to
 * a vector of length 1...)
 *
 * @author Adam Gibson
 */
public class Shape {

    private static Map<Integer,IndexMapper> indexMappers = Collections.synchronizedMap(new HashMap<Integer, IndexMapper>());
    private static Map<Integer,IndexMapper> indexMappersC = Collections.synchronizedMap(new HashMap<Integer, IndexMapper>());

    static {
        for(int i = 0; i < 10; i++) {
            indexMappersC.put(i,ShapeMapper.getInd2SubInstance('c',i));
            indexMappers.put(i,ShapeMapper.getInd2SubInstance('f',i));

        }
    }


    /**
     * Create a copy of the matrix
     * where the new offset is zero
     * @param arr the array to copy to offset 0
     * @return the same array if offset is zero
     * otherwise a copy of the array with
     * elements set to zero
     */
    public static INDArray toOffsetZero(INDArray arr) {
        if(arr.offset() < 1 && arr.data().length() == arr.length() || arr instanceof  IComplexNDArray && arr.length() * 2 == arr.data().length())
            return arr;

        if(arr.isRowVector()) {
            if(arr instanceof IComplexNDArray) {
                IComplexNDArray ret = Nd4j.createComplex(arr.shape());
                for(int i = 0; i < ret.length(); i++)
                    ret.putScalar(i, ((IComplexNDArray) arr).getComplex(i));
                return ret;
            }
            else {
                INDArray ret = Nd4j.create(arr.shape());
                for(int i = 0; i < ret.length(); i++)
                    ret.putScalar(i,arr.getDouble(i));
                return ret;
            }
        }


        if(arr instanceof IComplexNDArray) {
            IComplexNDArray ret = Nd4j.createComplex(arr.shape());
            for(int i = 0; i < ret.slices(); i++)
                ret.putSlice(i,arr.slice(i));
            return ret;
        }
        else {
            INDArray ret = Nd4j.create(arr.shape());
            for(int i = 0; i < ret.slices(); i++)
                ret.putSlice(i,arr.slice(i));
            return ret;
        }
    }

    /**
     * Create a copy of the matrix
     * where the new offset is zero
     * @param arr the array to copy to offset 0
     * @return the same array if offset is zero
     * otherwise a copy of the array with
     * elements set to zero
     */
    public static INDArray toOffsetZeroCopy(INDArray arr) {
        if(arr.isRowVector()) {
            if(arr instanceof IComplexNDArray) {
                IComplexNDArray ret = Nd4j.createComplex(arr.shape());
                for(int i = 0; i < ret.length(); i++)
                    ret.putScalar(i, ((IComplexNDArray) arr).getComplex(i));
                return ret;
            }
            else {
                INDArray ret = Nd4j.create(arr.shape());
                for(int i = 0; i < ret.length(); i++)
                    ret.putScalar(i,arr.getDouble(i));
                return ret;
            }
        }


        if(arr instanceof IComplexNDArray) {
            IComplexNDArray ret = Nd4j.createComplex(arr.shape());
            for(int i = 0; i < ret.slices(); i++)
                ret.putSlice(i,arr.slice(i));
            return ret;
        }
        else {
            INDArray ret = Nd4j.create(arr.shape());
            for(int i = 0; i < arr.vectorsAlongDimension(0); i++) {
                ret.vectorAlongDimension(i,0).assign(arr.vectorAlongDimension(i,0));
            }


            return ret;
        }
    }


    /**
     * Output an int array for a particular dimension
     * @param axes the axes
     * @param shape the current shape
     * @return
     */
    public static int[] sizeForAxes(int[] axes, int[] shape) {
        int[] ret = new int[shape.length];
        for (int i = 0; i < axes.length; i++) {
            ret[i] = shape[axes[i]];
        }
        return ret;
    }


    /**
     * Returns whether the given shape is a vector
     *
     * @param shape the shape to test
     * @return whether the given shape is a vector
     */
    public static boolean isVector(int[] shape) {
        if (shape.length > 2 || shape.length < 1)
            return false;
        else {
            int len = ArrayUtil.prod(shape);
            return shape[0] == len || shape[1] == len;
        }
    }

    /**
     * Returns whether the passed in shape is a matrix
     *
     * @param shape whether the passed in shape is a matrix
     * @return true if the shape is a matrix false otherwise
     */
    public static boolean isMatrix(int[] shape) {
        if (shape.length != 2)
            return false;
        return !isVector(shape);
    }


    /**
     * Gets rid of any singleton dimensions of the given array
     *
     * @param shape the shape to squeeze
     * @return the array with all of the singleton dimensions removed
     */
    public static int[] squeeze(int[] shape) {
        if(isColumnVectorShape(shape))
            return shape;

        List<Integer> ret = new ArrayList<>();

        //strip all but last dimension
        for (int i = 0; i < shape.length; i++)
            if (shape[i] != 1)
                ret.add(shape[i]);
        return ArrayUtil.toArray(ret);
    }

    /**
     * Keep all the non one dimensions
     * @param dimensions the dimensions to start with
     * @param shape the shapes to inspect
     * @return the non one dimensions of the given input
     */
    public static int[] nonOneDimensions(int[] dimensions,int[] shape) {
        if(dimensions.length != shape.length)
            throw new IllegalArgumentException("Dimensions and shape must be the same length");

        List<Integer> list = new ArrayList<>();
        for(int i = 0; i < dimensions.length; i++) {
            if(shape[i] != 1) {
                list.add(i);
            }
        }

        return Ints.toArray(list);
    }

    /**
     * Get rid ones in the shape when
     * its not a vector
     * @param original the original shape
     *                 to prune
     * @return the pruned array
     */
    public static int[] leadingAndTrailingOnes(int[] original) {
        List<Integer> ints = new ArrayList<>();
        if (!Shape.isVector(original)) {
            for (int i = 0; i < original.length; i++) {
                if(original[i] != 1)
                    ints.add(original[i]);
            }

            return Ints.toArray(ints);
        }
        return original;
    }


    /**
     * Returns whether 2 shapes are equals by checking for dimension semantics
     * as well as array equality
     *
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the shapes are equivalent
     */
    public static boolean shapeEquals(int[] shape1, int[] shape2) {
        if (isColumnVectorShape(shape1)) {
            if (isColumnVectorShape(shape2)) {
                return Arrays.equals(shape1, shape2);
            }

        }

        if (isRowVectorShape(shape1)) {
            if (isRowVectorShape(shape2)) {
                int[] shape1Comp = squeeze(shape1);
                int[] shape2Comp = squeeze(shape2);
                return Arrays.equals(shape1Comp, shape2Comp);
            }
        }

        shape1 = squeeze(shape1);
        shape2 = squeeze(shape2);

        return scalarEquals(shape1, shape2) || Arrays.equals(shape1, shape2);
    }


    /**
     * Returns true if the given shapes are both scalars (0 dimension or shape[0] == 1)
     *
     * @param shape1 the first shape for comparison
     * @param shape2 the second shape for comparison
     * @return whether the 2 shapes are equal based on scalar rules
     */
    public static boolean scalarEquals(int[] shape1, int[] shape2) {
        if (shape1.length == 0) {
            if (shape2.length == 1 && shape2[0] == 1)
                return true;
        } else if (shape2.length == 0) {
            if (shape1.length == 1 && shape1[0] == 1)
                return true;
        }

        return false;
    }

    /**
     * Returns true if the given shape is of length 1
     * or provided the shape length is 2:
     * element 0 is 1
     * @param shape the shape to check
     * @return true if the above conditions hold,false otherwise
     */
    public static boolean isRowVectorShape(int[] shape) {
        return
                (shape.length == 2
                        && shape[0] == 1) ||
                        shape.length == 1;

    }

    /**
     * Returns true if the given shape is length 2 and
     * the size at element 1 is 1
     * @param shape the shape to check
     * @return true if the above listed conditions
     * hold false otherwise
     */
    public static boolean isColumnVectorShape(int[] shape) {
        return
                (shape.length == 2
                        && shape[1] == 1);

    }


    /**
     * A port of numpy's stride resolution algorithm
     * for multiple arrays
     * @param arrays the arrays to get concat strides for
     * @return the resolved strides for concat
     */
    public static int[] createConcatStrides(INDArray...arrays) {
        int rank = arrays[0].rank();
        for(INDArray arr : arrays) {
            if(arr.rank() != rank)
                throw new IllegalArgumentException("All arrays must have same rank");
        }

        int[] ret = new int[rank];

        int i0, i1, ipos, ax_j0, ax_j1, iarrays;

         /* Initialize the strideperm values to the identity. */
        for (i0 = 0; i0 < rank; i0++) {
            ret[i0] = i0;
        }

    /*
     * This is the same as the custom stable insertion sort in
     * the NpyIter object, but sorting in the reverse order as
     * in the iterator. The iterator sorts from smallest stride
     * to biggest stride (Fortran order), whereas here we sort
     * from biggest stride to smallest stride (C order).
     */
        for (i0 = 1; i0 < rank; i0++) {

            ipos = i0;
            ax_j0 = ret[i0];

            for (i1 = i0 - 1; i1 >= 0; i1--) {
                boolean ambig = true, shouldSwap = false;

                ax_j1 = ret[i1];

                for (iarrays = 0; iarrays < arrays.length; ++iarrays) {
                    if (arrays[iarrays].size(ax_j0) != 1 &&
                            arrays[iarrays].size(ax_j1) != 1) {
                        if (Math.abs(arrays[iarrays].stride(ax_j0)) <=
                                Math.abs(arrays[iarrays].size(ax_j1))) {
                        /*
                         * Set swap even if it's not ambiguous already,
                         * because in the case of conflicts between
                         * different operands, C-order wins.
                         */
                            shouldSwap = false;
                        }
                        else {
                        /* Only set swap if it's still ambiguous */
                            if (ambig) {
                                shouldSwap = true;
                            }
                        }

                    /*
                     * A comparison has been done, so it's
                     * no longer ambiguous
                     */
                        ambig = false;
                    }
                }
            /*
             * If the comparison was unambiguous, either shift
             * 'ipos' to 'i1' or stop looking for an insertion point
             */
                if (!ambig) {
                    if (shouldSwap) {
                        ipos = i1;
                    }
                    else {
                        break;
                    }
                }
            }

        /* Insert out_strideperm[i0] into the right place */
            if (ipos != i0) {
                for (i1 = i0; i1 > ipos; i1--) {
                    ret[i1] = ret[i1 - 1];
                }

                ret[ipos] = ax_j0;
            }
        }

        return ret;

    }


    /**
     * A port of numpy's reshaping algorithm that leverages
     * no copy where possible and returns
     * null if the reshape
     * couldn't happen without copying
     * @param arr  the array to reshape
     * @param newShape the new shape
     * @param isFOrder whether the array will be fortran ordered or not
     * @return null if a reshape isn't possible, or a new ndarray
     */
    public static  INDArray newShapeNoCopy(INDArray arr, int[] newShape, boolean isFOrder) {
        int oldnd;
        int[] olddims = ArrayUtil.copy(arr.shape());
        int[] oldstrides = ArrayUtil.copy(arr.stride());
        int np, op, last_stride;
        int oi, oj, ok, ni, nj, nk;
        int[] newStrides = new int[newShape.length];
        oldnd = 0;
    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
        for (oi = 0; oi < arr.rank(); oi++) {
            if (arr.size(oi) != 1) {
                olddims[oldnd] = arr.size(oi);
                oldstrides[oldnd] = arr.stride(oi);
                oldnd++;
            }
        }

        np = 1;
        for (ni = 0; ni < newShape.length; ni++) {
            np *= newShape[ni];
        }
        op = 1;
        for (oi = 0; oi < oldnd; oi++) {
            op *= olddims[oi];
        }
        if (np != op) {
        /* different total sizes; no hope */
            return null;
        }

        if (np == 0) {
        /* the current code does not handle 0-sized arrays, so give up */
            return null;
        }

    /* oi to oj and ni to nj give the axis ranges currently worked with */
        oi = 0;
        oj = 1;
        ni = 0;
        nj = 1;
        while (ni < newShape.length && oi < oldnd) {
            np = newShape[ni];
            op = olddims[oi];

            while (np != op) {
                if (np < op) {
                /* Misses trailing 1s, these are handled later */
                    np *= newShape[nj++];
                } else {
                    op *= olddims[oj++];
                }
            }

        /* Check whether the original axes can be combined */
            for (ok = oi; ok < oj - 1; ok++) {
                if (isFOrder) {
                    if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
                     /* not contiguous enough */
                        return null;
                    }
                }
                else {
                /* C order */
                    if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                        return null;
                    }
                }
            }

             /* Calculate new strides for all axes currently worked with */
            if (isFOrder) {
                newStrides[ni] = oldstrides[oi];
                for (nk = ni + 1; nk < nj; nk++) {
                    newStrides[nk] = newStrides[nk - 1]* newShape[nk - 1];
                }
            }
            else {
            /* C order */
                newStrides[nj - 1] = oldstrides[oj - 1];
                for (nk = nj - 1; nk > ni; nk--) {
                    newStrides[nk - 1] = newStrides[nk]* newShape[nk];
                }
            }
            ni = nj++;
            oi = oj++;
        }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
        if (ni >= 1) {
            last_stride = newStrides[ni - 1];
        }
        else {
            last_stride = arr.elementStride();
        }
        if (isFOrder) {
            if(ni >= 1)
                last_stride *= newShape[ni - 1];
        }
        for (nk = ni; nk < newShape.length; nk++) {
            newStrides[nk] = last_stride;
        }

        if(arr instanceof IComplexNDArray)
            return Nd4j.createComplex(arr.data(),newShape,newStrides,arr.offset());

        return Nd4j.create(arr.data(),newShape,newStrides,arr.offset());
    }


    /**
     * Infer order from
     * @param shape the shape to infer by
     * @param stride the stride to infer by
     * @param elementStride the element stride to start at
     * @return the storage order given shape and element stride
     */
    public static char getOrder(int[] shape,int[] stride,int elementStride) {
        int sd;
        int dim;
        int i;
        boolean cContiguous = true;
        boolean isFortran = true;

        sd = 1;
        for (i = shape.length - 1; i >= 0; --i) {
            dim = shape[i];

            if (stride[i] != sd) {
                cContiguous = false;
                break;
            }
        /* contiguous, if it got this far */
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }


    /* check if fortran contiguous */
        sd = elementStride;
        for (i = 0; i < shape.length; ++i) {
            dim = shape[i];
            if (stride[i] != sd) {
                isFortran = false;
            }
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }

        if(isFortran && cContiguous)
            return 'a';
        else if(isFortran && !cContiguous)
            return 'f';
        else if(!isFortran && !cContiguous)
            return 'c';
        else
            return 'c';

    }

    /**
     * Infer the order for the ndarray based on the
     * array's strides
     * @param arr the array to get the
     *            ordering for
     * @return the ordering for the given array
     */
    public static char getOrder(INDArray arr) {
        return getOrder(arr.shape(),arr.stride(),arr.elementStride());
    }

    /**
     * Convert the given index (such as 1,1)
     * to a linear index
     * @param shape the shape of the indexes to convert
     * @param indices the index to convert
     * @return the linear index given the shape
     * and indices
     */
    public static int sub2Ind(int[] shape,int[] indices) {
        int index = 0;
        int shift = 1;
        for(int i = 0; i < shape.length; i++) {
            index += shift * indices[i];
            shift *= shape[i];
        }
        return index;
    }

    /**
     * Convert a linear index to
     * the equivalent nd index
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @param numIndices the number of total indices (typically prod of shape(
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2sub(int[] shape,int index,int numIndices) {
        IndexMapper mapper = indexMappers.get(shape.length);
        if(mapper == null) {
            mapper = ShapeMapper.getInd2SubInstance('f',shape.length);
            indexMappers.put(index,mapper);
            mapper = ShapeMapper.getInd2SubInstance('c',shape.length);
            indexMappersC.put(index,mapper);
        }



     /*   int denom = numIndices;
        int[] ret = new int[shape.length];
        for(int i = ret.length - 1; i >= 0; i--) {
            denom /= shape[i];
            ret[i] = index / denom;
            index %= denom;

        }*/
        return mapper.ind2sub(shape,index,numIndices,'f');
    }

    /**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2sub(int[] shape,int index) {
        return ind2sub(shape,index,ArrayUtil.prod(shape));
    }

    /**
     * Convert a linear index to
     * the equivalent nd index based on the shape of the specified ndarray.
     * Infers the number of indices from the specified shape.
     *
     * @param arr the array to compute the indexes
     *            based on
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2sub(INDArray arr,int index) {
        return ind2sub(arr.shape(), index, ArrayUtil.prod(arr.shape()));
    }




    /**
     * Convert a linear index to
     * the equivalent nd index
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @param numIndices the number of total indices (typically prod of shape(
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2subC(int[] shape,int index,int numIndices) {
        IndexMapper mapper = indexMappersC.get(shape.length);
        if(mapper == null) {
            mapper = ShapeMapper.getInd2SubInstance('f',shape.length);
            indexMappers.put(index,mapper);
            mapper = ShapeMapper.getInd2SubInstance('c',shape.length);
            indexMappersC.put(index,mapper);
        }

      /*
        int denom = numIndices;
        int[] ret = new int[shape.length];
        for(int i = 0; i < shape.length; i++) {
            denom /= shape[i];
            ret[i] = index / denom;
            index %= denom;

        }
        return ret;*/
        return mapper.ind2sub(shape,index,numIndices,'c');
    }

    /**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2subC(int[] shape,int index) {
        return ind2subC(shape, index, ArrayUtil.prod(shape));
    }

    /**
     * Convert a linear index to
     * the equivalent nd index based on the shape of the specified ndarray.
     * Infers the number of indices from the specified shape.
     *
     * @param arr the array to compute the indexes
     *            based on
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
    public static int[] ind2subC(INDArray arr,int index) {
        return ind2subC(arr.shape(), index, ArrayUtil.prod(arr.shape()));
    }

    /**
     * Compute the offset for the given array
     * given the indices
     * @param arr the array to compute the offset for
     * @param indexes the indexes along each dimension to create the offset for
     * @return the offset for the given array and indexes
     */
    public static int offsetFor(INDArray arr,int[] indexes) {
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(arr);
        resolution.exec(Shape.toIndexes(indexes));
        return resolution.getOffset();
    }

    /**
     * Compute the offset for a given index
     * @param arr the array to compute the offset for
     * @param index the linear index to compute the offset for
     * @return the offset for the given linear index
     */
    public static int offsetFor(INDArray arr,int index) {
        int[] indexes = arr.ordering() == 'c' ? Shape.ind2subC(arr,index) : Shape.ind2sub(arr, index);
        return offsetFor(arr, indexes);
    }


    /**
     * Assert the both shapes are the same length
     * and shape[i] < lessThan[i]
     * @param shape the shape to check
     * @param lessThan the shape to assert against
     */
    public static void assertShapeLessThan(int[] shape,int[] lessThan) {
        if(shape.length != lessThan.length) {
            throw new IllegalArgumentException("Shape length must be == less than length");
        }
        for(int i = 0; i < shape.length; i++) {
            if(shape[i] >= lessThan[i])
                throw new IllegalStateException("Shape[" + i + "] should be less than lessThan[" + i + "]");
        }
    }


    /**
     * Returns a permutation of the dimensions
     * with all 1s moved to end
     * @param shape the shape to permute
     * @return the permuted axes with all moving towards
     * end
     */
    public static int[] moveOnesToEnd(int[] shape) {
        List<Integer> nonOnes = new ArrayList<>();
        List<Integer> ones = new ArrayList<>();
        for(int i = 0; i < shape.length; i++) {
            if(shape[i] == 1)
                ones.add(i);
            else
                nonOnes.add(i);
        }

        return Ints.concat(Ints.toArray(nonOnes), Ints.toArray(ones));
    }

    /**
     * Convert the given int indexes
     * to nd array indexes
     * @param indices the indices to convert
     * @return the converted indexes
     */
    public static INDArrayIndex[] toIndexes(int[] indices) {
        INDArrayIndex[] ret = new INDArrayIndex[indices.length];
        for(int i = 0; i < ret.length; i++)
            ret[i] = new NDArrayIndex(indices[i]);
        return ret;
    }


    public static int[] newStrides(int[] strides,int newLength,INDArrayIndex[] indexes) {
        if(strides.length > newLength) {
            int[] newStrides = new int[strides.length - 1];
            for(int i = 0; i < newStrides.length; i++) {
                newStrides[i] = strides[i + 1];
            }
            strides = newStrides;
        }

        return strides;
    }

    public static int[] newOffsets(int[] offsets,int newLength,INDArrayIndex[] indexes) {
        if(offsets.length > newLength) {
            int[] newOffsets = new int[offsets.length - 1];
            for(int i = 0; i < newOffsets.length; i++) {
                newOffsets[i] = offsets[i + 1];


            }

            offsets = newOffsets;
        }

        return offsets;
    }


    public static int[] squeezeOffsets(int[] shape,int[] offsets) {
        //bump offsets
        List<Integer> squeezeIndices = new ArrayList<>();
        for(int i = 0; i < shape.length; i++)
            if(offsets[i] == 0)
                squeezeIndices.add(i);
        int[] ret = ArrayUtil.removeIndex(offsets, Ints.toArray(squeezeIndices));
        int delta = Math.abs(ret.length - shape.length);
        if(delta == 0)
            return ret;
        else {
            if(ret.length > shape.length)
                throw new IllegalStateException("Unable to squeeze offsets");
            int[] retOffsets = new int[shape.length];
            System.arraycopy(ret,0,retOffsets,0,ret.length);
            return retOffsets;
        }
    }




    /**
     * Returns true for the case where
     * singleton dimensions are being compared
     *
     * @param test1 the first to test
     * @param test2 the second to test
     * @return true if the arrays
     * are equal with the singleton dimension omitted
     */
    public static boolean squeezeEquals(int[] test1, int[] test2) {
        int[] s1 = squeeze(test1);
        int[] s2 = squeeze(test2);
        return scalarEquals(s1, s2) || Arrays.equals(s1, s2);
    }


}
