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
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
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

    /** Do element-wise operation on two NDArrays.
     * Ops:
     * 'a': addi    first += second
     * 's': subi    first -= second
     * 'm': muli    first *= second
     * 'd': divi    first /= second
     * 'p': put     first =  second
     */
    public static void doElementWiseOp(INDArray first, INDArray second, char op){
        if(canDoOpDirectly(first,second)){
            doOpDirectly(first,second,op);
        } else {
            //Decide which dimension we want to split on
            //doing argMax on shape gives us smallest number of largest tensors
            //but may not be optimal in terms of element separation (for CPU cache etc)
            int opAlongDimension = ArrayUtil.argMax(first.shape());

            Tensor1DStats fs = get1DTensorStats(first, opAlongDimension);
            Tensor1DStats ss = get1DTensorStats(second, opAlongDimension);
            if(fs.tensorStartSeparation == fs.getTensorLength()*fs.getElementWiseStride() &&
                    ss.tensorStartSeparation == ss.getTensorLength()*ss.getElementWiseStride() ){
                //One tensor ends and the next begins at same element-wise interval for both
                doOpDirectly(first,second,op);
            } else {
                doOp(first, second, op, fs, ss);
            }
        }
    }

    /** Do an operation on the entire NDArray instead of breaking it up into tensors.
     * Can do this under certain circumstances */
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
                    float[] fArr1 = (float[])arrayFirst;
                    float[] fArr2 = (float[])arraySecond;
                    if(op=='m') {   //muli
                        if(incrFirst == 1 && incrSecond == 1 ){
                            if(offsetFirst==0 && offsetSecond==0){
                                muliSimpleFloat(fArr1,fArr2,opLength);
                            } else {
                                muliOffsetUnitIncrementFloat(fArr1,fArr2,opLength,offsetFirst,offsetSecond);
                            }
                        } else {
                            if(offsetFirst==0 && offsetSecond==0){
                                muliIncrementNoOffsetFloat(fArr1,fArr2,opLength,incrFirst,incrSecond);
                            } else {
                                muliIncrementOffsetFloat(fArr1,fArr2,opLength,offsetFirst,offsetSecond,incrFirst,incrSecond);
                            }
                        }
                    } else {    //divi
                        if(incrFirst == 1 && incrSecond == 1 ){
                            if(offsetFirst==0 && offsetSecond==0){
                                diviSimpleFloat(fArr1,fArr2,opLength);
                            } else {
                                diviOffsetUnitIncrementFloat(fArr1,fArr2,opLength,offsetFirst,offsetSecond);
                            }
                        } else {
                            if(offsetFirst==0 && offsetSecond==0){
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
                        if(incrFirst == 1 && incrSecond == 1 ){
                            if(offsetFirst==0 && offsetSecond==0){
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
                        if(incrFirst == 1 && incrSecond == 1 ){
                            if(offsetFirst==0 && offsetSecond==0){
                                diviSimpleDouble(dArr1, dArr2, opLength);
                            } else {
                                diviOffsetUnitIncrementDouble(dArr1, dArr2, opLength, offsetFirst, offsetSecond);
                            }
                        } else {
                            if(offsetFirst==0 && offsetSecond==0){
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

    private static void doOp(INDArray first, INDArray second, char op, Tensor1DStats fs, Tensor1DStats ss ){
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
                for(int i=0; i<nTensors; i++ ) {
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

    private static boolean canDoOpDirectly(INDArray first, INDArray second){
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
        boolean rowOp = Shape.isRowVectorShape(vector.shape());

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
