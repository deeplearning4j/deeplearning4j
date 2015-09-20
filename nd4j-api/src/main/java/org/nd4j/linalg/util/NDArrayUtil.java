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

package org.nd4j.linalg.util;

import com.google.common.primitives.Ints;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Basic INDArray ops
 *
 * @author Adam Gibson
 */
public class NDArrayUtil {





    public static INDArray exp(INDArray toExp) {
        return expi(toExp.dup());
    }

    /**
     * Returns an exponential version of this ndarray
     *
     * @param toExp the INDArray to convert
     * @return the converted ndarray
     */
    public static INDArray expi(INDArray toExp) {
        INDArray flattened = toExp.ravel();
        for (int i = 0; i < flattened.length(); i++)
            flattened.put(i, Nd4j.scalar(Math.exp((double) flattened.getScalar(i).element())));
        return flattened.reshape(toExp.shape());
    }

    /**
     * Center an array
     *
     * @param arr   the arr to center
     * @param shape the shape of the array
     * @return the center portion of the array based on the
     * specified shape
     */
    public static INDArray center(INDArray arr, int[] shape) {
        if (arr.length() < ArrayUtil.prod(shape))
            return arr;
        for (int i = 0; i < shape.length; i++)
            if (shape[i] < 1)
               shape[i] = 1;

        INDArray shapeMatrix = ArrayUtil.toNDArray(shape);
        INDArray currShape = ArrayUtil.toNDArray(arr.shape());

        INDArray startIndex = Transforms.floor(currShape.sub(shapeMatrix).divi(Nd4j.scalar(2)));
        INDArray endIndex = startIndex.add(shapeMatrix);
        INDArrayIndex[] indexes = Indices.createFromStartAndEnd(startIndex, endIndex);

        if (shapeMatrix.length() > 1)
            return arr.get(indexes);


        else {
            INDArray ret = Nd4j.create(new int[]{(int) shapeMatrix.getDouble(0)});
            int start = (int) startIndex.getDouble(0);
            int end = (int) endIndex.getDouble(0);
            int count = 0;
            for (int i = start; i < end; i++) {
                ret.putScalar(count++, arr.getDouble(i));
            }

            return ret;
        }
    }

    /**
     * Truncates an INDArray to the specified shape.
     * If the shape is the same or greater, it just returns
     * the original array
     *
     * @param nd the INDArray to truncate
     * @param n  the number of elements to truncate to
     * @return the truncated ndarray
     */
    public static INDArray truncate(INDArray nd, final int n, int dimension) {

        if (nd.isVector()) {
            INDArray truncated = Nd4j.create(new int[]{n});
            for (int i = 0; i < n; i++)
                truncated.put(i, nd.getScalar(i));
            return truncated;
        }

        if (nd.size(dimension) > n) {
            int[] targetShape = ArrayUtil.copy(nd.shape());
            targetShape[dimension] = n;
            int numRequired = ArrayUtil.prod(targetShape);
            if (nd.isVector()) {
                INDArray ret = Nd4j.create(targetShape);
                int count = 0;
                for (int i = 0; i < nd.length(); i += nd.stride()[dimension]) {
                    ret.put(count++, nd.getScalar(i));

                }
                return ret;
            } else if (nd.isMatrix()) {
                List<Double> list = new ArrayList<>();
                //row
                if (dimension == 0) {
                    for (int i = 0; i < nd.rows(); i++) {
                        INDArray row = nd.getRow(i);
                        for (int j = 0; j < row.length(); j++) {
                            if (list.size() == numRequired)
                                return Nd4j.create(ArrayUtil.toArrayDouble(list), targetShape);

                            list.add((Double) row.getScalar(j).element());
                        }
                    }
                } else if (dimension == 1) {
                    for (int i = 0; i < nd.columns(); i++) {
                        INDArray row = nd.getColumn(i);
                        for (int j = 0; j < row.length(); j++) {
                            if (list.size() == numRequired)
                                return Nd4j.create(ArrayUtil.toArrayDouble(list), targetShape);

                            list.add((Double) row.getScalar(j).element());
                        }
                    }
                } else
                    throw new IllegalArgumentException("Illegal dimension for matrix " + dimension);


                return Nd4j.create(ArrayUtil.toArrayDouble(list), targetShape);

            }


            if (dimension == 0) {
                List<INDArray> slices = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    INDArray slice = nd.slice(i);
                    slices.add(slice);
                }

                return Nd4j.create(slices, targetShape);

            } else {
                List<Double> list = new ArrayList<>();
                int numElementsPerSlice = ArrayUtil.prod(ArrayUtil.removeIndex(targetShape, 0));
                for (int i = 0; i < nd.slices(); i++) {
                    INDArray slice = nd.slice(i).ravel();
                    for (int j = 0; j < numElementsPerSlice; j++)
                        list.add((Double) slice.getScalar(j).element());
                }

                assert list.size() == ArrayUtil.prod(targetShape) : "Illegal shape for length " + list.size();

                return Nd4j.create(ArrayUtil.toArrayDouble(list), targetShape);

            }


        }

        return nd;

    }

    /**
     * Pads an INDArray with zeros
     *
     * @param nd          the INDArray to pad
     * @param targetShape the the new shape
     * @return the padded ndarray
     */
    public static INDArray padWithZeros(INDArray nd, int[] targetShape) {
        if (Arrays.equals(nd.shape(), targetShape))
            return nd;
        //no padding required
        if (ArrayUtil.prod(nd.shape()) >= ArrayUtil.prod(targetShape))
            return nd;

        INDArray ret = Nd4j.create(targetShape);
        System.arraycopy(nd.data(), 0, ret.data(), 0, (int) nd.data().length());
        return ret;

    }

    public static Tensor1DStats get1DTensorStats(INDArray array, int dimension){
        //As per BaseNDArray.tensorAlongDimension
        int[] tensorShape = ArrayUtil.keep(array.shape(), dimension);
        int tensorLength = ArrayUtil.prod(tensorShape);

        int[] remove = ArrayUtil.removeIndex(ArrayUtil.range(0, array.rank()), dimension);
        int[] newPermuteDims = Arrays.copyOf(remove, remove.length + 1);
        newPermuteDims[newPermuteDims.length-1] = dimension;


        INDArray temp0 = array.tensorAlongDimension(0,dimension);
        INDArray temp1 = array.tensorAlongDimension(1, dimension);
        int tensorStartSeparation = temp1.offset() - temp0.offset();

        //As per NDArrayMath.sliceOffsetForTensor
        int firstTensorOffset = array.offset();

        //As per tensorssAlongDimension:
        int numTensors = array.length() / tensorLength;

        int elementStride = temp0.elementWiseStride();  //TODO

        return new Tensor1DStats(firstTensorOffset,tensorStartSeparation,
                numTensors,tensorLength,elementStride);
    }

    @AllArgsConstructor @Data
    public static class Tensor1DStats {
        public final int firstTensorOffset;
        public final int tensorStartSeparation;
        public final int numTensors;
        public final int tensorLength;
        public final int elementWiseStride;
    }

    public static void doElementWiseOp(INDArray first, INDArray second, char op){
        boolean canDoDirectly = false;
        if(canDoDirectly){
            throw new UnsupportedOperationException("Not implemented");
        } else {
            int opAlongDimension = ArrayUtil.argMin(first.shape());
            Tensor1DStats fs = get1DTensorStats(first, opAlongDimension);
            Tensor1DStats ss = get1DTensorStats(second,opAlongDimension);
            doOp(first,second,op,fs,ss);
        }
    }

    private static INDArray doOp(INDArray first, INDArray second, char op, Tensor1DStats fs, Tensor1DStats ss ){
        DataBuffer df = first.data();
        DataBuffer ds = second.data();
        int n = fs.getTensorLength();
        int nTensors = fs.getNumTensors();
        int incrF = fs.getElementWiseStride();
        int incrS = fs.getElementWiseStride();
        switch(op){
            case 'a':
            case 's':
                //first.addi(second) or first.subi(second)
                double a = (op == 'a' ? 1.0 : -1.0);
                for(int i=0; i<nTensors; i++ ) {
                    int offset1 = fs.firstTensorOffset + i*fs.getTensorStartSeparation();
                    int offset2 = ss.firstTensorOffset + i*ss.getTensorStartSeparation();
                    Nd4j.getBlasWrapper().level1().axpy(n, a, ds,offset2,incrS, df,offset1,incrF);
                }
                break;
            case 'm':
                //muli
                throw new UnsupportedOperationException("not yet implemented");
            case 'd':
                //divi
                throw new UnsupportedOperationException("not yet implemented");
            case 'p':   //put / copy
                throw new UnsupportedOperationException("not yet implemented");
        }
        return null;    //TODO
    }


}
