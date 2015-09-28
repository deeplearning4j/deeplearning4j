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

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.Indices;
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

    /** Tensor1DStats, used to efficiently iterate through tensors on a matrix (2d NDArray) for element-wise ops
     */
    public static Tensor1DStats get1DTensorStats(INDArray array, int dimension){
        //As per BaseNDArray.tensorAlongDimension
        int tensorLength = ArrayUtil.prod(ArrayUtil.keep(array.shape(), dimension));

        //As per tensorssAlongDimension:
        int numTensors = array.length() / tensorLength;

        //First tensor always starts with the first element in the NDArray, regardless of dimension
        int firstTensorOffset = array.offset();

        //Next: Need to work out the separation between the start (first element) of each 1d tensor
        int tensorStartSeparation;
        int elementWiseStride;  //Separation in buffer between elements in the tensor
        if(numTensors == 1){
            tensorStartSeparation = -1; //Not applicable
            elementWiseStride = array.elementWiseStride();
        } else {
            INDArray secondTensor = array.tensorAlongDimension(1, dimension);
            tensorStartSeparation = secondTensor.offset() - firstTensorOffset;
            elementWiseStride = secondTensor.elementWiseStride();
        }

        return new Tensor1DStats(firstTensorOffset,tensorStartSeparation,
                numTensors,tensorLength,elementWiseStride);
    }

    @AllArgsConstructor @Data
    public static class Tensor1DStats {
        public final int firstTensorOffset;
        public final int tensorStartSeparation;
        public final int numTensors;
        public final int tensorLength;
        public final int elementWiseStride;
    }

    /** Do element-wise operation on two NDArrays.
     * Ops:
     * 'a': addi    first += second
     * 's': subi    first -= second
     * 'm': muli    first *= second
     * 'd': divi    first /= second
     * 'p': put     first =  second
     */
    public static void doElementWiseOp(INDArray first, INDArray second, char op) {
        if(canDoElementWiseOpDirectly(first, second)){
            doOpDirectly(first,second,op);
        } else {
            //Decide which dimension we want to split on
            int opAlongDimension = chooseElementWiseTensorDimension(first, second);

            if(first.rank() == 2) {
                //Certain optimizations are possible on 2d that are not always possible on 3+d
                Tensor1DStats fs = get1DTensorStats(first, opAlongDimension);
                Tensor1DStats ss = get1DTensorStats(second, opAlongDimension);
                if (fs.tensorStartSeparation == fs.getTensorLength() * fs.getElementWiseStride() &&
                        ss.tensorStartSeparation == ss.getTensorLength() * ss.getElementWiseStride()) {
                    //One tensor ends and the next begins at same element-wise interval for both
                    doOpDirectly(first, second, op);
                } else {
                    doOpOnMatrix(first, second, op, fs, ss);
                }
            } else {
                doElementWiseOpGeneral(first,second,op);
            }
        }
    }

    /** Can we do the element-wise op directly on the arrays without breaking them up into 1d tensors first? */
    private static boolean canDoElementWiseOpDirectly(INDArray first, INDArray second){
        if(first.isVector()) return true;

        //Full buffer + matching strides -> implies all elements are contiguous (and match)
        int l1 = first.length();
        int dl1 = first.data().length();
        int l2 = second.length();
        int dl2 = second.data().length();
        int[] strides1 = first.stride();
        int[] strides2 = second.stride();
        boolean equalStrides = Arrays.equals(strides1, strides2);
        if(l1==dl1 && l2==dl2 && equalStrides) return true;

        //Strides match + are same as a zero offset NDArray -> all elements are contiguous (and match)
        int[] shape1 = first.shape();
        int[] stridesAsInit = (first.ordering()=='c' ? ArrayUtil.calcStrides(shape1) : ArrayUtil.calcStridesFortran(shape1));
        boolean stridesSameAsInit = Arrays.equals(strides1, stridesAsInit);
        if(equalStrides && stridesSameAsInit) return true;

        return false;
    }

    private static int chooseElementWiseTensorDimension(INDArray first, INDArray second){
        //doing argMin(max(first.stride(i),second.stride(i))) minimizes the maximum
        //separation between elements (helps CPU cache) BUT might result in a huge number
        //of tiny ops - i.e., addi on NDArrays with shape [5,10^6]
        int opAlongDimensionMinStride = ArrayUtil.argMinOfMax(first.stride(), second.stride());

        //doing argMax on shape gives us smallest number of largest tensors
        //but may not be optimal in terms of element separation (for CPU cache etc)
        int opAlongDimensionMaxLength = ArrayUtil.argMax(first.shape());

        //Edge cases: shapes with 1s in them can have stride of 1 on the dimensions of length 1
        if(first.size(opAlongDimensionMinStride)==1) return opAlongDimensionMaxLength;

        //Using a heuristic approach here: basically if we get >= 10x as many tensors using the minimum stride
        //dimension vs. the maximum size dimension, use the maximum size dimension instead
        //The idea is to avoid choosing wrong dimension in cases like shape=[10,10^6]
        //Might be able to do better than this with some additional thought
        int nOpsAlongMinStride = ArrayUtil.prod(ArrayUtil.keep(first.shape(), opAlongDimensionMinStride));
        int nOpsAlongMaxLength = ArrayUtil.prod(ArrayUtil.keep(first.shape(), opAlongDimensionMaxLength));
        if(nOpsAlongMinStride <= 10*nOpsAlongMaxLength) return opAlongDimensionMinStride;
        else return opAlongDimensionMaxLength;
    }

    /** Do an operation on the entire NDArray instead of breaking it up into tensors.
     * Can do this when elements are contiguous in buffer and first/second arrays
     * have same stride - i.e., when canDoElementWiseOpDirectly(...) returns true
     * */
    private static void doOpDirectly(INDArray first, INDArray second, char op){
        switch(op){
            case 'a':
            case 's':
                //first.addi(second) or first.subi(second)
                double a = (op == 'a' ? 1.0 : -1.0);
                Nd4j.getBlasWrapper().level1().axpy(first.length(), a, second, first);
                break;
            case 'm':   //muli
            case 'd':   //divi
                int incrFirst = first.elementWiseStride();
                int incrSecond = second.elementWiseStride();
                int offsetFirst = first.offset();
                int offsetSecond = second.offset();
                int opLength = first.length();
                Object arrayFirst = first.data().array();
                Object arraySecond = second.data().array();

                if(arrayFirst instanceof float[]) {
                    float[] fArr1 = (float[]) arrayFirst;
                    float[] fArr2 = (float[]) arraySecond;
                    if(op=='m') {   //muli
                        if(incrFirst == 1 && incrSecond == 1) {
                            if(offsetFirst==0 && offsetSecond == 0) {
                                muliSimpleFloat(fArr1,fArr2,opLength);
                            } else {
                                muliOffsetUnitIncrementFloat(fArr1,fArr2,opLength,offsetFirst,offsetSecond);
                            }
                        } else {
                            if(offsetFirst == 0 && offsetSecond == 0 ) {
                                muliIncrementNoOffsetFloat(fArr1,fArr2,opLength,incrFirst,incrSecond);
                            } else {
                                muliIncrementOffsetFloat(fArr1,fArr2,opLength,offsetFirst,offsetSecond,incrFirst,incrSecond);
                            }
                        }
                    } else {    //divi
                        if(incrFirst == 1 && incrSecond == 1) {
                            if(offsetFirst == 0 && offsetSecond == 0) {
                                diviSimpleFloat(fArr1,fArr2,opLength);
                            } else {
                                diviOffsetUnitIncrementFloat(fArr1,fArr2,opLength,offsetFirst,offsetSecond);
                            }
                        } else {
                            if(offsetFirst == 0 && offsetSecond == 0) {
                                diviIncrementNoOffsetFloat(fArr1,fArr2,opLength,incrFirst,incrSecond);
                            } else {
                                diviIncrementOffsetFloat(fArr1,fArr2,opLength,offsetFirst,offsetSecond,incrFirst,incrSecond);
                            }
                        }
                    }
                } else {    //double ops
                    double[] dArr1 = (double[])arrayFirst;
                    double[] dArr2 = (double[])arraySecond;
                    if(op=='m') {   //muli
                        if(incrFirst == 1 && incrSecond == 1) {
                            if(offsetFirst==0 && offsetSecond == 0) {
                                muliSimpleDouble(dArr1, dArr2, opLength);
                            } else {
                                muliOffsetUnitIncrementDouble(dArr1, dArr2, opLength, offsetFirst, offsetSecond);
                            }
                        } else {
                            if(offsetFirst==0 && offsetSecond==0){
                                muliIncrementNoOffsetDouble(dArr1, dArr2, opLength, incrFirst, incrSecond);
                            } else {
                                muliIncrementOffsetDouble(dArr1, dArr2, opLength, offsetFirst, offsetSecond, incrFirst, incrSecond);
                            }
                        }
                    } else {    //divi
                        if(incrFirst == 1 && incrSecond == 1) {
                            if(offsetFirst==0 && offsetSecond == 0) {
                                diviSimpleDouble(dArr1, dArr2, opLength);
                            } else {
                                diviOffsetUnitIncrementDouble(dArr1, dArr2, opLength, offsetFirst, offsetSecond);
                            }
                        } else {
                            if(offsetFirst == 0 && offsetSecond == 0) {
                                diviIncrementNoOffsetDouble(dArr1, dArr2, opLength, incrFirst, incrSecond);
                            } else {
                                diviIncrementOffsetDouble(dArr1, dArr2, opLength, offsetFirst, offsetSecond, incrFirst, incrSecond);
                            }
                        }
                    }
                }
                break;
            case 'p':   //put / copy
                Nd4j.getBlasWrapper().level1().copy(second,first); //first = second
                break;
            default:
                throw new UnsupportedOperationException("Unknown op: " + op);
        }
    }

    /** For 2d arrays (matrices) we can utilize the fact that each 1d tensor is separated by a fixed amount.
     * This allows us to avoid the cost of multiple calls to tensorAlongDimension
     * This does not always hold for 3+d matrices
     */
    private static void doOpOnMatrix(INDArray first, INDArray second, char op, Tensor1DStats fs, Tensor1DStats ss) {
        DataBuffer df = first.data();
        DataBuffer ds = second.data();
        int n = fs.getTensorLength();
        int nTensors = fs.getNumTensors();
        int incrF = fs.getElementWiseStride();
        int incrS = ss.getElementWiseStride();
        Level1 l1Blas = Nd4j.getBlasWrapper().level1();
        switch(op){
            case 'a':
            case 's':
                //first.addi(second) or first.subi(second)
                double a = (op == 'a' ? 1.0 : -1.0);
                for(int i = 0; i<nTensors; i++) {
                    int offset1 = fs.getFirstTensorOffset() + i*fs.getTensorStartSeparation();
                    int offset2 = ss.getFirstTensorOffset() + i*ss.getTensorStartSeparation();
                    l1Blas.axpy(n, a, ds, offset2, incrS, df, offset1, incrF);
                }
                break;
            case 'm':
                //muli
                Object arrayFirst = first.data().array();
                Object arraySecond = second.data().array();
                if(arrayFirst instanceof float[]) {
                    float[] f1 = (float[])arrayFirst;
                    float[] f2 = (float[])arraySecond;
                    if (incrF == 1 && incrS == 1) {
                        for (int i = 0; i < nTensors; i++) {
                            int offset1 = fs.getFirstTensorOffset() + i * fs.getTensorStartSeparation();
                            int offset2 = ss.getFirstTensorOffset() + i * ss.getTensorStartSeparation();
                            muliOffsetUnitIncrementFloat(f1,f2,n,offset1,offset2);
                        }
                    } else {
                        for (int i = 0; i < nTensors; i++) {
                            int offset1 = fs.getFirstTensorOffset() + i * fs.getTensorStartSeparation();
                            int offset2 = ss.getFirstTensorOffset() + i * ss.getTensorStartSeparation();
                            muliIncrementOffsetFloat(f1,f2,n,offset1,offset2,incrF,incrS);
                        }
                    }
                } else {
                    double[] f1 = (double[])arrayFirst;
                    double[] f2 = (double[])arraySecond;
                    if (incrF == 1 && incrS == 1) {
                        for (int i = 0; i < nTensors; i++) {
                            int offset1 = fs.getFirstTensorOffset() + i * fs.getTensorStartSeparation();
                            int offset2 = ss.getFirstTensorOffset() + i * ss.getTensorStartSeparation();
                            muliOffsetUnitIncrementDouble(f1, f2, n, offset1, offset2);
                        }
                    } else {
                        for (int i = 0; i < nTensors; i++) {
                            int offset1 = fs.getFirstTensorOffset() + i * fs.getTensorStartSeparation();
                            int offset2 = ss.getFirstTensorOffset() + i * ss.getTensorStartSeparation();
                            muliIncrementOffsetDouble(f1, f2, n, offset1, offset2, incrF, incrS);
                        }
                    }
                }
                break;
            case 'd':
                //divi
                Object arrayFirstd = first.data().array();
                Object arraySecondd = second.data().array();
                if(arrayFirstd instanceof float[]) {
                    float[] f1 = (float[])arrayFirstd;
                    float[] f2 = (float[])arraySecondd;
                    if (incrF == 1 && incrS == 1) {
                        for (int i = 0; i < nTensors; i++) {
                            int offset1 = fs.getFirstTensorOffset() + i * fs.getTensorStartSeparation();
                            int offset2 = ss.getFirstTensorOffset() + i * ss.getTensorStartSeparation();
                            diviOffsetUnitIncrementFloat(f1,f2,n,offset1,offset2);
                        }
                    } else {
                        for (int i = 0; i < nTensors; i++) {
                            int offset1 = fs.getFirstTensorOffset() + i * fs.getTensorStartSeparation();
                            int offset2 = ss.getFirstTensorOffset() + i * ss.getTensorStartSeparation();
                            diviIncrementOffsetFloat(f1,f2,n,offset1,offset2,incrF,incrS);
                        }
                    }
                } else {
                    double[] f1 = (double[])arrayFirstd;
                    double[] f2 = (double[])arraySecondd;
                    if (incrF == 1 && incrS == 1) {
                        for (int i = 0; i < nTensors; i++) {
                            int offset1 = fs.getFirstTensorOffset() + i * fs.getTensorStartSeparation();
                            int offset2 = ss.getFirstTensorOffset() + i * ss.getTensorStartSeparation();
                            diviOffsetUnitIncrementDouble(f1,f2,n,offset1,offset2);
                        }
                    } else {
                        for (int i = 0; i < nTensors; i++) {
                            int offset1 = fs.getFirstTensorOffset() + i * fs.getTensorStartSeparation();
                            int offset2 = ss.getFirstTensorOffset() + i * ss.getTensorStartSeparation();
                            diviIncrementOffsetDouble(f1,f2,n,offset1,offset2,incrF,incrS);
                        }
                    }
                }
                break;
            case 'p':   //put / copy
                for(int i=0; i<nTensors; i++ ) {
                    int offset1 = fs.getFirstTensorOffset() + i*fs.getTensorStartSeparation();
                    int offset2 = ss.getFirstTensorOffset() + i*ss.getTensorStartSeparation();
                    l1Blas.copy(n,ds,offset2,incrS,df,offset1,incrF);
                }
                break;
            default:
                throw new RuntimeException("Unknown op: " + op);
        }
    }

    private static void doElementWiseOpGeneral(INDArray first, INDArray second, char op){
        //Decide which dimension we want to split on
        int opAlongDimension = chooseElementWiseTensorDimension(first, second);
        int nTensors = first.tensorssAlongDimension(opAlongDimension);

        DataBuffer df = first.data();
        DataBuffer ds = second.data();
        Level1 l1Blas = Nd4j.getBlasWrapper().level1();
        switch(op){
            case 'a':
            case 's':
                //first.addi(second) or first.subi(second)
                double a = (op == 'a' ? 1.0 : -1.0);
                for(int i=0; i<nTensors; i++ ) {
                    INDArray tad1 = first.tensorAlongDimension(i,opAlongDimension);
                    INDArray tad2 = second.tensorAlongDimension(i,opAlongDimension);
                    l1Blas.axpy(tad1.length(),a,ds,tad2.offset(),tad2.elementWiseStride(),df,tad1.offset(),tad1.elementWiseStride());
                }
                break;
            case 'm':
                //muli
                Object arrayFirst = df.array();
                Object arraySecond = ds.array();
                if(arrayFirst instanceof float[]) {
                    float[] f1 = (float[])arrayFirst;
                    float[] f2 = (float[])arraySecond;
                    for (int i = 0; i < nTensors; i++) {
                        INDArray tad1 = first.tensorAlongDimension(i,opAlongDimension);
                        INDArray tad2 = second.tensorAlongDimension(i,opAlongDimension);
                        muliIncrementOffsetFloat(f1,f2,tad1.length(),tad1.offset(),tad2.offset(),tad1.elementWiseStride(),tad2.elementWiseStride());
                    }
                } else {
                    double[] f1 = (double[])arrayFirst;
                    double[] f2 = (double[])arraySecond;
                    for (int i = 0; i < nTensors; i++) {
                        INDArray tad1 = first.tensorAlongDimension(i,opAlongDimension);
                        INDArray tad2 = second.tensorAlongDimension(i,opAlongDimension);
                        muliIncrementOffsetDouble(f1, f2, tad1.length(), tad1.offset(), tad2.offset(), tad1.elementWiseStride(), tad2.elementWiseStride());
                    }
                }
                break;
            case 'd':
                //divi
                Object arrayFirstd = first.data().array();
                Object arraySecondd = second.data().array();
                if(arrayFirstd instanceof float[]) {
                    float[] f1 = (float[])arrayFirstd;
                    float[] f2 = (float[])arraySecondd;
                    for (int i = 0; i < nTensors; i++) {
                        INDArray tad1 = first.tensorAlongDimension(i,opAlongDimension);
                        INDArray tad2 = second.tensorAlongDimension(i,opAlongDimension);
                        diviIncrementOffsetFloat(f1,f2,tad1.length(),tad1.offset(),tad2.offset(),tad1.elementWiseStride(),tad2.elementWiseStride());
                    }
                } else {
                    double[] f1 = (double[])arrayFirstd;
                    double[] f2 = (double[])arraySecondd;
                    for (int i = 0; i < nTensors; i++) {
                        INDArray tad1 = first.tensorAlongDimension(i,opAlongDimension);
                        INDArray tad2 = second.tensorAlongDimension(i,opAlongDimension);
                        diviIncrementOffsetDouble(f1,f2,tad1.length(),tad1.offset(),tad2.offset(),tad1.elementWiseStride(),tad2.elementWiseStride());
                    }
                }
                break;
            case 'p':   //put / copy
                for(int i=0; i<nTensors; i++ ) {
                    INDArray tad1 = first.tensorAlongDimension(i,opAlongDimension);
                    INDArray tad2 = second.tensorAlongDimension(i,opAlongDimension);
                    l1Blas.copy(tad1.length(),ds,tad2.offset(),tad2.elementWiseStride(),df,tad1.offset(),tad1.elementWiseStride());
                }
                break;
            default:
                throw new RuntimeException("Unknown op: " + op);
        }

    }

    /** Do element-wise vector operation on an NDArray (using a vector NDArray)
     * Ops:
     * When vector is a row vector:
     * 'a': addiRowVector    first.eachRow += second
     * 's': subiRowVector    first.eachRow -= second
     * 'm': muliRowVector    first.eachRow *= second
     * 'd': diviRowVector    first.eachRow /= second
     * 'p': putRowVector     first.eachRow =  second
     * 'h': rsubiRowVector   first.eachRow = second - first.eachRow
     * 't': rdiviRowVector   first.eachRow = second / first.eachRow
     *
     * When vector is a column vector: do column-wise instead of row-wise ops
     */
    public static void doVectorOp(INDArray array, INDArray vector, char op){
        if(array.rank() != 2) throw new IllegalArgumentException("Cannot do row/column operation on non-2d matrix");
        boolean rowOp = Shape.isRowVectorShape(vector.shape());
        //Edge case: 'vector' is actually a scalar
        if(vector.length() == 1){
            //Expect 'array' to be a vector also
            if(array.isRowVector()){
                rowOp = false;
            } else if(array.isColumnVector()){
                rowOp = true;
            } else throw new IllegalArgumentException("Invalid input: vector input is a scalar but array is not a vector");
        }

        Tensor1DStats tensorStats = get1DTensorStats(array, (rowOp ? 1 : 0));

        int incr = tensorStats.getElementWiseStride();
        int incrVec = vector.elementWiseStride();
        int offsetVec = vector.offset();
        int opLength = vector.length();
        int nTensors = tensorStats.getNumTensors();

        DataBuffer buffer = array.data();
        DataBuffer bufferVec = vector.data();

        Object dataArray = buffer.array();
        Object dataArrayVec = bufferVec.array();
        float[] dataFloat;
        float[] dataFloatVec;
        double[] dataDouble;
        double[] dataDoubleVec;
        if(dataArray instanceof float[]){
            dataFloat = (float[])dataArray;
            dataFloatVec = (float[])dataArrayVec;
            dataDouble = null;
            dataDoubleVec = null;
        } else {
            dataFloat = null;
            dataFloatVec = null;
            dataDouble = (double[])dataArray;
            dataDoubleVec = (double[])dataArrayVec;
        }

        Level1 l1Blas = Nd4j.getBlasWrapper().level1();
        switch(op){
            case 'a':   //addiXVector
            case 's':   //subiXVector
                double a = (op == 'a' ? 1.0 : -1.0);
                for( int i=0; i<nTensors; i++){
                    int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                    l1Blas.axpy(opLength, a, bufferVec, offsetVec, incrVec, buffer, tOffset, incr);
                }
                break;
            case 'm':   //muliXVector
                if(dataFloat != null){
                    if(incr==1 && incrVec==1){
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            muliOffsetUnitIncrementFloat(dataFloat,dataFloatVec,opLength,tOffset,offsetVec);
                        }
                    } else {
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            muliIncrementOffsetFloat(dataFloat, dataFloatVec, opLength, tOffset, offsetVec, incr, incrVec);
                        }
                    }
                } else {
                    if(incr==1 && incrVec==1){
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            muliOffsetUnitIncrementDouble(dataDouble, dataDoubleVec, opLength, tOffset, offsetVec);
                        }
                    } else {
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            muliIncrementOffsetDouble(dataDouble, dataDoubleVec, opLength, tOffset, offsetVec, incr, incrVec);
                        }
                    }
                }
                break;
            case 'd':   //diviXVector
                if(dataFloat != null){
                    if(incr==1 && incrVec==1){
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            diviOffsetUnitIncrementFloat(dataFloat, dataFloatVec, opLength, tOffset, offsetVec);
                        }
                    } else {
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            diviIncrementOffsetFloat(dataFloat, dataFloatVec, opLength, tOffset, offsetVec, incr, incrVec);
                        }
                    }
                } else {
                    if(incr==1 && incrVec==1){
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            diviOffsetUnitIncrementDouble(dataDouble, dataDoubleVec, opLength, tOffset, offsetVec);
                        }
                    } else {
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            diviIncrementOffsetDouble(dataDouble, dataDoubleVec, opLength, tOffset, offsetVec, incr, incrVec);
                        }
                    }
                }
                break;
            case 'p':   //putiXVector
                for(int i=0; i<nTensors; i++ ) {
                    int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                    l1Blas.copy(opLength,bufferVec,offsetVec,incrVec,buffer,tOffset,incr);
                }
                break;
            case 'h':   //rsubiXVector
                if(dataFloat != null){
                    if(incr==1 && incrVec==1){
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            rsubiOffsetUnitIncrementFloat(dataFloat, dataFloatVec, opLength, tOffset, offsetVec);
                        }
                    } else {
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            rsubiIncrementOffsetFloat(dataFloat, dataFloatVec, opLength, tOffset, offsetVec, incr, incrVec);
                        }
                    }
                } else {
                    if(incr==1 && incrVec==1){
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            rsubiOffsetUnitIncrementDouble(dataDouble, dataDoubleVec, opLength, tOffset, offsetVec);
                        }
                    } else {
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            rsubiIncrementOffsetDouble(dataDouble, dataDoubleVec, opLength, tOffset, offsetVec, incr, incrVec);
                        }
                    }
                }
                break;
            case 't':   //rdiviXVector
                if(dataFloat != null){
                    if(incr==1 && incrVec==1){
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            rdiviOffsetUnitIncrementFloat(dataFloat, dataFloatVec, opLength, tOffset, offsetVec);
                        }
                    } else {
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            rdiviIncrementOffsetFloat(dataFloat, dataFloatVec, opLength, tOffset, offsetVec, incr, incrVec);
                        }
                    }
                } else {
                    if(incr==1 && incrVec==1){
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            rdiviOffsetUnitIncrementDouble(dataDouble, dataDoubleVec, opLength, tOffset, offsetVec);
                        }
                    } else {
                        for( int i=0; i<nTensors; i++ ){
                            int tOffset = tensorStats.getFirstTensorOffset() + i*tensorStats.getTensorStartSeparation();
                            rdiviIncrementOffsetDouble(dataDouble, dataDoubleVec, opLength, tOffset, offsetVec, incr, incrVec);
                        }
                    }
                }
                break;
            default:
                throw new RuntimeException("Unknown op: " + op);
        }

    }

    //muli - float
    private static void muliIncrementOffsetFloat(float[] first, float[] second, int opLength, int offsetFirst, int offsetSecond, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            first[offsetFirst + i * incrFirst] *= second[offsetSecond + i * incrSecond];
        }
    }

    private static void muliIncrementNoOffsetFloat(float[] first, float[] second, int opLength, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            first[i * incrFirst] *= second[i * incrSecond];
        }
    }

    private static void muliOffsetUnitIncrementFloat(float[] first, float[] second, int opLength, int offsetFirst, int offsetSecond){
        for (int i = 0; i < opLength; i++) {
            first[offsetFirst + i] *= second[offsetSecond + i];
        }
    }

    private static void muliSimpleFloat(float[] first, float[] second, int opLength){
        for (int i = 0; i < opLength; i++) {
            first[i] *= second[i];
        }
    }

    //muli - double
    private static void muliIncrementOffsetDouble(double[] first, double[] second, int opLength, int offsetFirst, int offsetSecond, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            first[offsetFirst + i * incrFirst] *= second[offsetSecond + i * incrSecond];
        }
    }

    private static void muliIncrementNoOffsetDouble(double[] first, double[] second, int opLength, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            first[i * incrFirst] *= second[i * incrSecond];
        }
    }

    private static void muliOffsetUnitIncrementDouble(double[] first, double[] second, int opLength, int offsetFirst, int offsetSecond){
        for (int i = 0; i < opLength; i++) {
            first[offsetFirst + i] *= second[offsetSecond + i];
        }
    }

    private static void muliSimpleDouble(double[] first, double[] second, int opLength){
        for (int i = 0; i < opLength; i++) {
            first[i] *= second[i];
        }
    }

    //divi - float
    private static void diviIncrementOffsetFloat(float[] first, float[] second, int opLength, int offsetFirst, int offsetSecond, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            first[offsetFirst + i * incrFirst] /= second[offsetSecond + i * incrSecond];
        }
    }

    private static void diviIncrementNoOffsetFloat(float[] first, float[] second, int opLength, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            first[i * incrFirst] /= second[i * incrSecond];
        }
    }

    private static void diviOffsetUnitIncrementFloat(float[] first, float[] second, int opLength, int offsetFirst, int offsetSecond){
        for (int i = 0; i < opLength; i++) {
            first[offsetFirst + i] /= second[offsetSecond + i];
        }
    }

    private static void diviSimpleFloat(float[] first, float[] second, int opLength){
        for (int i = 0; i < opLength; i++) {
            first[i] /= second[i];
        }
    }

    //divi - double
    private static void diviIncrementOffsetDouble(double[] first, double[] second, int opLength, int offsetFirst, int offsetSecond, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            first[offsetFirst + i * incrFirst] /= second[offsetSecond + i * incrSecond];
        }
    }

    private static void diviIncrementNoOffsetDouble(double[] first, double[] second, int opLength, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            first[i * incrFirst] /= second[i * incrSecond];
        }
    }

    private static void diviOffsetUnitIncrementDouble(double[] first, double[] second, int opLength, int offsetFirst, int offsetSecond){
        for (int i = 0; i < opLength; i++) {
            first[offsetFirst + i] /= second[offsetSecond + i];
        }
    }

    private static void diviSimpleDouble(double[] first, double[] second, int opLength){
        for (int i = 0; i < opLength; i++) {
            first[i] /= second[i];
        }
    }

    //rsubi - float
    private static void rsubiIncrementOffsetFloat(float[] first, float[] second, int opLength, int offsetFirst, int offsetSecond, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            int fIdx = offsetFirst + i * incrFirst;
            first[fIdx] = second[offsetSecond + i * incrSecond] - first[fIdx];
        }
    }

    private static void rsubiOffsetUnitIncrementFloat(float[] first, float[] second, int opLength, int offsetFirst, int offsetSecond){
        for (int i = 0; i < opLength; i++) {
            int fIdx = offsetFirst + i;
            first[fIdx] = second[offsetSecond + i] - first[fIdx];
        }
    }

    //rsubi - double
    private static void rsubiIncrementOffsetDouble(double[] first, double[] second, int opLength, int offsetFirst, int offsetSecond, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            int fIdx = offsetFirst + i * incrFirst;
            first[fIdx] = second[offsetSecond + i * incrSecond] - first[fIdx];
        }
    }

    private static void rsubiOffsetUnitIncrementDouble(double[] first, double[] second, int opLength, int offsetFirst, int offsetSecond){
        for (int i = 0; i < opLength; i++) {
            int fIdx = offsetFirst + i;
            first[fIdx] = second[offsetSecond + i] - first[fIdx];
        }
    }



    //rdivi - float
    private static void rdiviIncrementOffsetFloat(float[] first, float[] second, int opLength, int offsetFirst, int offsetSecond, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            int fIdx = offsetFirst + i * incrFirst;
            first[fIdx] = second[offsetSecond + i * incrSecond] / first[fIdx];
        }
    }

    private static void rdiviOffsetUnitIncrementFloat(float[] first, float[] second, int opLength, int offsetFirst, int offsetSecond){
        for (int i = 0; i < opLength; i++) {
            int fIdx = offsetFirst + i;
            first[fIdx] = second[offsetSecond + i] / first[fIdx];
        }
    }

    //rdivi - double
    private static void rdiviIncrementOffsetDouble(double[] first, double[] second, int opLength, int offsetFirst, int offsetSecond, int incrFirst, int incrSecond){
        for (int i = 0; i < opLength; i++) {
            int fIdx = offsetFirst + i * incrFirst;
            first[fIdx] = second[offsetSecond + i * incrSecond] / first[fIdx];
        }
    }

    private static void rdiviOffsetUnitIncrementDouble(double[] first, double[] second, int opLength, int offsetFirst, int offsetSecond){
        for (int i = 0; i < opLength; i++) {
            int fIdx = offsetFirst + i;
            first[fIdx] = second[offsetSecond + i] / first[fIdx];
        }
    }

}
