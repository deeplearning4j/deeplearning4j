/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.api.ndarray;

import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import net.ericaro.neoitertools.Generator;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Entropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.LogEntropy;
import org.nd4j.linalg.api.ops.impl.reduce.floating.ShannonEntropy;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Assign;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.Nd4jNoSuchWorkspaceException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.LinAlgExceptions;

import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

import static org.nd4j.linalg.factory.Nd4j.createUninitialized;

/**
 * @author Audrey Loeffel
 */
@Slf4j
public abstract class BaseSparseNDArray implements ISparseNDArray {

    /*
    * TODO: extends baseNdArray
    * */

    protected static final double THRESHOLD_MEMORY_ALLOCATION = 2;
    protected long rows, columns;
    protected int rank;
    protected Boolean isVector = null;
    protected Boolean isMatrix = null;
    protected Boolean isScalar = null;
    protected long length = -1;
    public static final boolean isSparse = true;
    protected transient volatile DataBuffer shapeInformation;
    protected transient volatile long[] javaShapeInformation;
    protected transient volatile DataBuffer sparseInformation;
    protected transient DataBuffer shape;
    protected transient DataBuffer stride;

    protected DataBuffer reallocate(DataBuffer buffer) {
        int newSize = (int) buffer.length() * 2;
        DataBuffer newBuffer = Nd4j.createBuffer(newSize);

        switch (buffer.dataType()) {
            case INT:
                newBuffer.setData(buffer.asInt());
                break;
            case DOUBLE:
                newBuffer.setData(buffer.asDouble());
                break;
            case FLOAT:
                newBuffer.setData(buffer.asFloat());
                break;
            case HALF:
                //TODO
                throw new UnsupportedOperationException();
            case COMPRESSED:
                //TODO
                throw new UnsupportedOperationException();
            default:
                throw new UnsupportedOperationException();
        }
        return newBuffer;
    }


    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }

    @Override
    public boolean isColumnVectorOrScalar() {
        return isColumnVector() || isScalar();
    }

    @Override
    public boolean isRowVectorOrScalar() {
        return isRowVector() || isScalar();
    }

    @Override
    public INDArray get(INDArray indices) {
        if(indices.rank() > 2) {
            throw new ND4JIllegalArgumentException("Indices must be a vector or matrix.");
        }

        if(indices.rows() == rank()) {
            INDArray ret = Nd4j.create(indices.columns());

            for(int i = 0; i < indices.columns(); i++) {
                int[] specifiedIndex = indices.getColumn(i).dup().data().asInt();
                ret.putScalar(i,getDouble(specifiedIndex));
            }

            return ret;
        }
        else {
            List<INDArray> arrList = new ArrayList<>();

            if(indices.isMatrix() || indices.isColumnVector()) {
                for(int i = 0; i < indices.rows(); i++) {
                    if(i == 0)  {
                        INDArray row = indices.getRow(i);
                        for(int j = 0; j < row.length(); j++) {
                            arrList.add(slice(row.getInt(j)));
                        }
                    }
                    else {
                        INDArray row = indices.slice(i);
                        for(int j = 0; j < row.length(); j++) {
                            INDArray put = arrList.get(j).slice(row.getInt(j));
                            put = put.reshape(Longs.concat(new long[]{1}, put.shape()));
                            arrList.set(j,put);
                        }
                    }

                }
            }
            else if(indices.isRowVector()) {
                for(int i = 0; i < indices.length(); i++) {
                    arrList.add(slice(indices.getInt(i)));
                }
            }

            return Nd4j.concat(0,arrList.toArray(new INDArray[arrList.size()]));

        }


    }

    @Override
    public double[][] toDoubleMatrix() {
        return new double[0][];
    }

    @Override
    public double[] toDoubleVector() {
        return new double[0];
    }

    @Override
    public float[] toFloatVector() {
        return new float[0];
    }

    @Override
    public float[][] toFloatMatrix() {
        return new float[0][];
    }

    @Override
    public int[] toIntVector() {
        return new int[0];
    }

    @Override
    public long[] toLongVector() {
        return new long[0];
    }

    @Override
    public long[][] toLongMatrix() {
        return new long[0][];
    }

    @Override
    public int[][] toIntMatrix() {
        return new int[0][];
    }

    @Override
    public INDArray match(INDArray comp, Condition condition) {
        return null;
    }

    @Override
    public INDArray match(Number comp, Condition condition) {
        return null;
    }

    @Override
    public INDArray putWhereWithMask(INDArray mask, INDArray put) {
        return null;
    }

    @Override
    public INDArray putWhereWithMask(INDArray mask, Number put) {
        return null;
    }

    @Override
    public INDArray toDense() {
        return null;
    }

    @Override
    public INDArray getWhere(INDArray comp, Condition condition) {
        return null;
    }

    @Override
    public INDArray getWhere(Number comp, Condition condition) {
        return null;
    }

    @Override
    public INDArray putWhere(INDArray comp, INDArray put, Condition condition) {
        return null;
    }

    @Override
    public INDArray putWhere(Number comp, INDArray put, Condition condition) {
        return null;
    }

    @Override
    public INDArray putWhere(Number comp, Number put, Condition condition) {
        return null;
    }

    @Override
    public INDArray get(List<List<Integer>> indices) {
        return null;
    }

    @Override
    public INDArray put(List<List<Integer>> indices, INDArray element) {
        if(indices.size() == rank()) {
            NdIndexIterator ndIndexIterator = new NdIndexIterator(element.shape());
            INDArrayIndex[] indArrayIndices = new INDArrayIndex[indices.size()];
            for(int i = 0; i < indArrayIndices.length; i++) {
                indArrayIndices[i] = new SpecifiedIndex(Ints.toArray(indices.get(i)));
            }
            boolean hasNext = true;
            Generator<List<List<Long>>> iterate = SpecifiedIndex.iterate(indArrayIndices);
            while(hasNext) {
                try {
                    List<List<Long>> next = iterate.next();
                    for(int i = 0; i < next.size(); i++) {
                        int[] curr = Ints.toArray(next.get(i));
                        putScalar(curr,element.getDouble(ndIndexIterator.next()));
                    }
                }
                catch(NoSuchElementException e) {
                    hasNext = false;
                }
            }

        }
        else {
            List<INDArray> arrList = new ArrayList<>();

            if(indices.size() >= 2) {
                for(int i = 0; i < indices.size(); i++) {
                    List<Integer> row = indices.get(i);
                    for(int j = 0; j < row.size(); j++) {
                        INDArray slice = slice(row.get(j));
                        Nd4j.getExecutioner().exec(new Assign(new INDArray[]{slice,element},new INDArray[]{slice}));
                        arrList.add(slice(row.get(j)));
                    }


                }
            }
            else if(indices.size() == 1) {
                for(int i = 0; i < indices.size(); i++) {
                    arrList.add(slice(indices.get(0).get(i)));
                }
            }

        }


        return this;
    }

    @Override
    public INDArray put(INDArray indices, INDArray element) {
        INDArrayIndex[] realIndices = new INDArrayIndex[indices.rank()];
        for(int i = 0; i < realIndices.length; i++) {
            realIndices[i] = new SpecifiedIndex(indices.slice(i).dup().data().asInt());
        }


        return put(realIndices,element);

    }

    @Override
    public boolean isSparse() {
        return isSparse;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public int nnz() {
        // FIXME: int cast
        return (int) length();
    }

    @Override
    public long lengthLong() {
        return length;
    }

    protected void init(long[] shape) {

        if (shape.length == 1) {
            rows = 1;
            columns = shape[0];
        } else if (this.shape().length == 2) {
            rows = shape[0];
            columns = shape[1];
        }
        rank = shape.length;

    }

    // Override methods from INDArray
    // TODO: Most of them should be reimplemented for each format

    @Override
    public String shapeInfoToString() {
        return Shape.shapeToString(this);
    }

    @Override
    public DataBuffer shapeInfoDataBuffer() {
        return shapeInformation;
    }

    @Override
    public DataBuffer sparseInfoDataBuffer() {
        return sparseInformation;
    }


    @Override
    public LongBuffer shapeInfo() {
        return null;
    }


    @Override
    public boolean isCompressed() {
        return false;
    }

    @Override
    public void markAsCompressed(boolean reallyCompressed) {

    }

    @Override
    public void setWrapAround(boolean wrapAround) {

    }

    @Override
    public boolean isWrapAround() {
        return false;
    }

    @Override
    public int rank() {
        return Shape.rank(shapeInformation);
    }

    @Override
    public int[] flags() {
        return Shape.flags(sparseInformation);
    }

    @Override
    public int[] hiddenDimensions() {
        return Shape.hiddenDimension(sparseInformation);
    }

    @Override
    public int[] sparseOffsets() {
        return Shape.sparseOffsets(sparseInformation);
    }


    @Override
    public int stride(int dimension) {
        int rank = Shape.rank(shapeInformation);
        if (dimension < 0)
            return strideOf().getInt(dimension + rank);
        return strideOf().getInt(dimension);
    }

    protected DataBuffer strideOf() {
        if (stride == null)
            stride = Shape.stride(shapeInfoDataBuffer());
        return stride;
    }

    @Override
    public int elementStride() {
        return 0;
    }

    @Override
    public int elementWiseStride() {
        return 0;
    }

    @Override
    public boolean isCleanedUp() {
        return false;
    }

    @Override
    public void cleanup() {

    }

    @Override
    public void resetLinearView() {

    }

    @Override
    public int secondaryStride() {
        return 0;
    }

    @Override
    public double getDoubleUnsafe(long offset) {
        return 0;
    }

    @Override
    public INDArray putScalarUnsafe(long offset, double value) {
        return null;
    }

    @Override
    public int majorStride() {
        return 0;
    }

    @Override
    public int innerMostStride() {
        return 0;
    }

    @Override
    public INDArray linearView() {
        return null;
    }

    @Override
    public INDArray linearViewColumnOrder() {
        return null;
    }

    @Override
    public long vectorsAlongDimension(int dimension) {
        return 0;
    }

    @Override
    public INDArray vectorAlongDimension(int index, int dimension) {
        return null;
    }

    @Override
    public long tensorssAlongDimension(int... dimension) {
        return 0;
    }

    @Override
    public INDArray tensorAlongDimension(int index, int... dimension) {
        return null;
    }

    @Override
    public INDArray javaTensorAlongDimension(int index, int... dimension) {
        return null;
    }

    @Override
    public INDArray cumsumi(int dimension) {
        return null;
    }

    @Override
    public INDArray cumsum(int dimension) {
        return null;
    }

    @Override
    public INDArray assign(INDArray arr) {
        return null;
    }

    @Override
    public INDArray assignIf(INDArray arr, Condition condition) {
        return null;
    }

    @Override
    public INDArray replaceWhere(INDArray arr, Condition condition) {
        return null;
    }

    @Override
    public INDArray putScalar(long i, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(long i, float value) {
        return null;
    }

    @Override
    public INDArray putScalar(long i, int value) {
        return null;
    }

    @Override
    public INDArray putScalar(int[] i, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(long[] i, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(long[] i, float value) {
        return null;
    }

    @Override
    public INDArray putScalar(long[] i, int value) {
        return null;
    }

    @Override
    public void setStride(long... stride) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setShape(long... shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray putScalar(long row, long col, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(long dim0, long dim1, long dim2, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(long dim0, long dim1, long dim2, long dim3, double value) {
        return null;
    }

    @Override
    public INDArray lt(Number other) {
        return null;
    }

    @Override
    public INDArray lti(Number other) {
        return null;
    }

    @Override
    public INDArray putScalar(int[] indexes, float value) {
        return null;
    }

    @Override
    public INDArray putScalar(int[] indexes, int value) {
        return null;
    }

    @Override
    public INDArray eps(Number other) {
        return null;
    }

    @Override
    public INDArray epsi(Number other) {
        return null;
    }

    @Override
    public INDArray eq(Number other) {
        return null;
    }

    @Override
    public INDArray eqi(Number other) {
        return null;
    }

    @Override
    public INDArray gt(Number other) {
        return null;
    }

    @Override
    public INDArray gte(Number other) {
        return null;
    }

    @Override
    public INDArray lte(Number other) {
        return null;
    }

    @Override
    public INDArray gtei(Number other) {
        return null;
    }

    @Override
    public INDArray ltei(Number other) {
        return null;
    }

    @Override
    public INDArray gti(Number other) {
        return null;
    }

    @Override
    public INDArray lt(INDArray other) {
        return null;
    }

    @Override
    public INDArray lti(INDArray other) {
        return null;
    }

    @Override
    public INDArray eps(INDArray other) {
        return null;
    }

    @Override
    public INDArray epsi(INDArray other) {
        return null;
    }

    @Override
    public INDArray neq(Number other) {
        return null;
    }

    @Override
    public INDArray neqi(Number other) {
        return null;
    }

    @Override
    public INDArray neq(INDArray other) {
        return null;
    }

    @Override
    public INDArray neqi(INDArray other) {
        return null;
    }

    @Override
    public INDArray eq(INDArray other) {
        return null;
    }

    @Override
    public INDArray eqi(INDArray other) {
        return null;
    }

    @Override
    public INDArray gt(INDArray other) {
        return null;
    }

    @Override
    public INDArray gti(INDArray other) {
        return null;
    }

    @Override
    public INDArray neg() {
        return null;
    }

    @Override
    public INDArray negi() {
        return null;
    }

    @Override
    public INDArray rdiv(Number n) {
        return null;
    }

    @Override
    public INDArray rdivi(Number n) {
        return null;
    }

    @Override
    public INDArray rsub(Number n) {
        return null;
    }

    @Override
    public INDArray rsubi(Number n) {
        return null;
    }

    @Override
    public INDArray div(Number n) {
        return null;
    }

    @Override
    public INDArray divi(Number n) {
        return null;
    }

    @Override
    public INDArray mul(Number n) {
        return null;
    }

    @Override
    public INDArray muli(Number n) {
        return null;
    }

    @Override
    public INDArray sub(Number n) {
        return null;
    }

    @Override
    public INDArray subi(Number n) {
        return null;
    }

    @Override
    public INDArray add(Number n) {
        return null;
    }

    @Override
    public INDArray addi(Number n) {
        return null;
    }

    @Override
    public INDArray rdiv(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray rdivi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray rsub(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray rsubi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray div(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray divi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray mul(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray muli(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray sub(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray subi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray add(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray addi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray get(INDArrayIndex... indexes) {
        return null;
    }

    @Override
    public INDArray getColumns(int... columns) {
        return null;
    }

    @Override
    public INDArray getRows(int... rows) {
        return null;
    }

    @Override
    public INDArray rdiv(INDArray other) {
        return null;
    }

    @Override
    public INDArray rdivi(INDArray other) {
        return null;
    }

    @Override
    public INDArray rdiv(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray rdivi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray rsub(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray rsub(INDArray other) {
        return null;
    }

    @Override
    public INDArray rsubi(INDArray other) {
        return null;
    }

    @Override
    public INDArray rsubi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray assign(Number value) {
        return null;
    }

    @Override
    public long linearIndex(long i) {
        return 0;
    }

    @Override
    public void checkDimensions(INDArray other) {

    }

    @Override
    public void sliceVectors(List<INDArray> list) {

    }

    @Override
    public INDArray putSlice(int slice, INDArray put) {
        return null;
    }

    @Override
    public INDArray cond(Condition condition) {
        return null;
    }

    @Override
    public INDArray condi(Condition condition) {
        return null;
    }

    @Override
    public INDArray repmat(int... shape) {
        return null;
    }

    @Override
    public INDArray repeat(int dimension, int... repeats) {
        return null;
    }

    @Override
    public INDArray putRow(long row, INDArray toPut) {
        return null;
    }

    @Override
    public INDArray putColumn(int column, INDArray toPut) {
        return null;
    }

    @Override
    public INDArray getScalar(long row, long column) {
        return null;
    }

    @Override
    public INDArray getScalar(long i) {
        return null;
    }

    @Override
    public long index(long row, long column) {
        return 0;
    }

    @Override
    public double squaredDistance(INDArray other) {
        return 0;
    }

    @Override
    public double distance2(INDArray other) {
        return 0;
    }

    @Override
    public double distance1(INDArray other) {
        return 0;
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, Number element) {
        return null;
    }

    @Override
    public INDArray put(int[] indices, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(int i, int j, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(int i, int j, Number element) {
        return null;
    }

    @Override
    public INDArray put(int i, INDArray element) {
        return null;
    }

    @Override
    public INDArray diviColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray divColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray diviRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray divRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray rdiviColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray rdivColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray rdiviRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray rdivRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray muliColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray mulColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray muliRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray mulRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray rsubiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray rsubColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray rsubiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray rsubRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray subiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray subColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray subiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray subRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray addiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray addColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray addiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray addRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray mmul(INDArray other) {
        long[] shape = {rows(), other.columns()};
        INDArray result = createUninitialized(shape, 'f');
        if (result.isScalar())
            return Nd4j.scalar(Nd4j.getBlasWrapper().dot(this, other));
        return mmuli(other, result);
    }

    @Override
    public INDArray mmul(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray div(INDArray other) {
        return null;
    }

    @Override
    public INDArray div(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray mul(INDArray other) {
        return null;
    }

    @Override
    public INDArray mul(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray sub(INDArray other) {
        return null;
    }

    @Override
    public INDArray sub(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray add(INDArray other) {
        return null;
    }

    @Override
    public INDArray add(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray mmuli(INDArray other) {
        return null;
    }

    @Override
    public INDArray mmuli(INDArray other, INDArray result) {
        LinAlgExceptions.assertMultiplies(this, other);


        if (other.isScalar()) {
            return muli(other.getDouble(0), result);
        }
        if (isScalar()) {
            return other.muli(getDouble(0), result);
        }

        /* check sizes and resize if necessary */



        //We require that the result array is 'f' (fortran) order
        // However, user might have called mmuli with a c order array for the result
        // In which case, we need to allocate a temporary f order array, and later do an assign to the real result array

        boolean requiresTemp = result.ordering() == 'c';
        INDArray gemmResultArr;
        if (requiresTemp) {
            //Can use createUninitialized due to beta==0.0 parameter in gemm
            gemmResultArr = Nd4j.createUninitialized(result.shape(), 'f');
        } else {
            gemmResultArr = result;
        }

        if (other.columns() == 1) {
            Nd4j.getBlasWrapper().level2().gemv(ordering(), BlasBufferUtil.getCharForTranspose(other), 1.0, this, other,
                            0.0, gemmResultArr);
        } else {
            //gemm doesn't support strides so vectors and views
            //don't work
            if (isView() && isVector()) {
                return dup().mmuli(other, gemmResultArr);
            }

            Nd4j.getBlasWrapper().level3().gemm(ordering(), BlasBufferUtil.getCharForTranspose(other),
                            BlasBufferUtil.getCharForTranspose(gemmResultArr), 1.0, this, other, 0.0, gemmResultArr);
        }

        if (requiresTemp) {
            result.assign(gemmResultArr);
        }


        if (Nd4j.ENFORCE_NUMERICAL_STABILITY)
            Nd4j.clearNans(result);

        return result;
    }

    @Override
    public INDArray divi(INDArray other) {
        return null;
    }

    @Override
    public INDArray divi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray muli(INDArray other) {
        return null;
    }

    @Override
    public INDArray muli(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray subi(INDArray other) {
        return null;
    }

    @Override
    public INDArray subi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray addi(INDArray other) {
        return null;
    }

    @Override
    public INDArray addi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray normmax(int... dimension) {
        return null;
    }

    @Override
    public Number normmaxNumber() {
        return null;
    }

    @Override
    public INDArray norm2(int... dimension) {
        return null;
    }

    @Override
    public Number norm2Number() {
        return null;
    }

    @Override
    public INDArray norm1(int... dimension) {
        return null;
    }

    @Override
    public Number norm1Number() {
        return null;
    }

    @Override
    public INDArray std(int... dimension) {
        return null;
    }

    @Override
    public Number stdNumber() {
        return null;
    }

    @Override
    public INDArray std(boolean biasCorrected, int... dimension) {
        return null;
    }

    @Override
    public Number stdNumber(boolean biasCorrected) {
        return null;
    }

    @Override
    public INDArray prod(int... dimension) {
        return null;
    }

    @Override
    public Number prodNumber() {
        return null;
    }

    @Override
    public INDArray mean(int... dimension) {
        return null;
    }

    @Override
    public Number meanNumber() {
        return null;
    }

    @Override
    public INDArray var(int... dimension) {
        return null;
    }

    @Override
    public INDArray var(boolean biasCorrected, int... dimension) {
        return null;
    }

    @Override
    public Number varNumber() {
        return null;
    }

    @Override
    public INDArray max(int... dimension) {
        return null;
    }

    @Override
    public Number maxNumber() {
        return null;
    }

    @Override
    public INDArray min(int... dimension) {
        return null;
    }

    @Override
    public Number minNumber() {
        return null;
    }

    @Override
    public INDArray sum(int... dimension) {
        return null;
    }

    @Override
    public Number sumNumber() {
        return null;
    }

    @Override
    public void setShapeAndStride(int[] shape, int[] stride) {

    }

    @Override
    public void setOrder(char order) {

    }

    @Override
    public INDArray subArray(ShapeOffsetResolution resolution) {
        return null;
    }

    @Override
    public INDArray subArray(long[] offsets, int[] shape, int[] stride) {
        return null;
    }

    @Override
    public INDArray getScalar(int... indices) {
        return null;
    }

    @Override
    public INDArray getScalar(long... indices) {
        return null;
    }

    @Override
    public float getFloat(long[] indices) {
        return 0;
    }

    @Override
    public double getDouble(long... indices) {
        return 0;
    }

    @Override
    public int getInt(int... indices) {
        return 0;
    }

    @Override
    public double getDouble(int... indices) {
        return 0;
    }

    @Override
    public float getFloat(int[] indices) {
        return 0;
    }

    @Override
    public double getDouble(long i) {
        return 0;
    }

    @Override
    public double getDouble(long i, long j) {
        return 0;
    }

    @Override
    public float getFloat(long i) {
        return 0;
    }

    @Override
    public float getFloat(long i, long j) {
        return 0;
    }

    @Override
    public INDArray dup() {
        return null;
    }

    @Override
    public INDArray dup(char order) {
        return null;
    }

    @Override
    public INDArray ravel() {
        return null;
    }

    @Override
    public INDArray ravel(char order) {
        return null;
    }

    @Override
    public void setData(DataBuffer data) {

    }

    @Override
    public long slices() {
        return 0;
    }

    @Override
    public int getTrailingOnes() {
        return 0;
    }

    @Override
    public int getLeadingOnes() {
        return 0;
    }

    @Override
    public INDArray slice(long i, int dimension) {
        return null;
    }

    @Override
    public INDArray slice(long i) {
        return null;
    }

    @Override
    public long offset() {
        return 0;
    }

    @Override
    public long originalOffset() {
        return 0;
    }

    @Override
    public INDArray reshape(char order, long... newShape) {
        return null;
    }

    @Override
    public INDArray reshape(char order, int rows, int columns) {
        return null;
    }

    @Override
    public INDArray reshape(long... newShape) {
        return null;
    }

    @Override
    public INDArray reshape(long rows, long columns) {
        return null;
    }

    @Override
    public INDArray transpose() {
        return null;
    }

    @Override
    public INDArray transposei() {
        return null;
    }

    @Override
    public INDArray swapAxes(int dimension, int with) {
        return null;
    }

    @Override
    public INDArray permute(int... rearrange) {
        return null;
    }

    @Override
    public INDArray permutei(int... rearrange) {
        return null;
    }

    @Override
    public INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable) {
        return null;
    }

    @Override
    public INDArray broadcast(INDArray result) {
        return null;
    }

    @Override
    public INDArray dimShuffle(Object[] rearrange, long[] newOrder, boolean[] broadCastable) {
        return null;
    }

    @Override
    public INDArray getColumn(long i) {
        return null;
    }

    @Override
    public INDArray getRow(long i) {
        return null;
    }

    @Override
    public int columns() {
        return (int) columns;
    }

    @Override
    public int rows() {
        return (int) rows;
    }

    /**
     * Checks whether the matrix is a vector.
     */
    @Override
    public boolean isVector() {
        return isRowVector() || isColumnVector();
    }

    @Override
    public boolean isVectorOrScalar() {
        return isVector() || isScalar();
    }

    @Override
    public boolean isSquare() {

        return isMatrix() && rows() == columns();
    }

    /**
     * Checks whether the matrix is a row vector.
     */
    @Override
    public boolean isRowVector() {
        return rank == 2 && rows == 1;
    }

    /**
     * Checks whether the matrix is a column vector.
     */
    @Override
    public boolean isColumnVector() {
        return rank == 2 && columns == 1;
    }


    @Override
    public boolean isMatrix() {
        if (isMatrix != null)
            return isMatrix;
        isMatrix = (rank == 2 && (size(0) != 1 && size(1) != 1));
        return isMatrix;
    }

    @Override
    public boolean isScalar() {
        if (isScalar != null)
            return isScalar;
        if (Shape.rank(shapeInformation) > 2) {
            isScalar = false;
        } else if (Shape.rank(shapeInformation) == 1) {
            isScalar = shapeOf().getInt(0) == 1;
        } else if (Shape.rank(shapeInformation) == 2) {
            isScalar = shapeOf().getInt(0) == 1 && shapeOf().getInt(1) == 1;
        }

        else
            isScalar = false;

        return isScalar;
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long[] shape() {
        return Shape.shape(javaShapeInformation);
    }

    protected DataBuffer shapeOf() {
        if (shape == null)
            shape = Shape.shapeOf(shapeInfoDataBuffer());
        return shape;
    }

    @Override
    public long[] stride() {
        return Shape.stride(javaShapeInformation);
    }

    @Override
    public long size(int dimension) {
        if (isScalar()) {
            if (dimension == 0 || dimension == 1 || dimension < 0)
                return (int) length;
            else
                throw new IllegalArgumentException("Illegal dimension for scalar " + dimension);
        }

        if (dimension < 0) {
            return shapeOf().getInt(dimension + Shape.rank(shapeInformation));
        }

        if (dimension >= rank())
            throw new IllegalArgumentException("Invalid size: cannot get size of dimension " + dimension + " for rank "
                            + rank() + " NDArray (array shape: " + Arrays.toString(this.shape()) + ")");


        return shapeOf().getInt(dimension);
    }

    protected void setShapeInformation(Pair<DataBuffer, long[]> shapeInfo) {
        this.shapeInformation = shapeInfo.getFirst();
        this.javaShapeInformation = shapeInfo.getSecond();
    }

    @Override
    public INDArray broadcast(long... shape) {
        return null;
    }

    @Override
    public Object element() {
        return null;
    }

    @Override
    public boolean equals(Object o) {
        return equalsWithEps(o, Nd4j.EPS_THRESHOLD);
    }

    @Override
    public boolean equalShapes(@NonNull INDArray other){
        if(rank() != other.rank())
            return false;
        for( int i=0; i<rank(); i++ ){
            if(size(i) != other.size(i)){
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean equalsWithEps(Object o, double eps) {
        if (o == null)
            return false;

        if (!(o instanceof INDArray))
            return false;

        INDArray n = (INDArray) o;

        if (this.lengthLong() != n.lengthLong())
            return false;

        if (isScalar() && n.isScalar()) {
            // TODO
        } else if (isVector && n.isVector()) {
            // TODO
        }
        if (!Arrays.equals(this.shape(), n.shape()))
            return false;

        // TODO
        return false;
    }

    @Override
    public INDArray unsafeDuplication() {
        return null;
    }

    @Override
    public INDArray remainder(INDArray denominator) {
        return null;
    }

    @Override
    public INDArray remainder(INDArray denominator, INDArray result) {
        return null;
    }

    @Override
    public INDArray remainder(Number denominator) {
        return null;
    }

    @Override
    public INDArray remainder(Number denominator, INDArray result) {
        return null;
    }

    @Override
    public INDArray remainderi(INDArray denominator) {
        return null;
    }

    @Override
    public INDArray remainderi(Number denominator) {
        return null;
    }

    @Override
    public INDArray fmod(INDArray denominator) {
        return null;
    }

    @Override
    public INDArray fmod(INDArray denominator, INDArray result) {
        return null;
    }

    @Override
    public INDArray fmod(Number denominator) {
        return null;
    }

    @Override
    public INDArray fmod(Number denominator, INDArray result) {
        return null;
    }

    @Override
    public INDArray fmodi(INDArray denominator) {
        return null;
    }

    @Override
    public INDArray fmodi(Number denominator) {
        return null;
    }

    @Override
    public INDArray argMax(int... dimension) {
        return null;
    }

    @Override
    public boolean isAttached() {
        return false;
    }

    @Override
    public boolean isInScope() {
        return false;
    }

    @Override
    public INDArray detach() {
        return null;
    }

    @Override
    public INDArray leverage() {
        return null;
    }

    @Override
    public INDArray leverageTo(String id) {
        return leverageTo(id, false);
    }

    public INDArray leverageTo(String id, boolean enforceExistence) throws Nd4jNoSuchWorkspaceException {
        return null;
    }

    @Override
    public INDArray leverageOrDetach(String id){
        return null;
    }

    @Override
    public INDArray migrate() {
        return migrate(false);
    }

    @Override
    public INDArray migrate(boolean detachIfNoWs){
        return null;
    }

    @Override
    public INDArray sum(INDArray result, int... dimension) {
        return null;
    }

    @Override
    public INDArray mean(INDArray result, int... dimension) {
        return null;
    }

    @Override
    public INDArray amean(int... dimension) {
        return null;
    }

    @Override
    public Number ameanNumber() {
        return null;
    }

    @Override
    public INDArray amax(int... dimension) {
        return null;
    }

    @Override
    public Number amaxNumber() {
        return null;
    }

    @Override
    public INDArray amin(int... dimension) {
        return null;
    }

    @Override
    public Number aminNumber() {
        return null;
    }

    @Override
    public Number scan(Condition condition) {
        return null;
    }

    @Override
    public INDArray unsafeDuplication(boolean blocking) {
        return null;
    }

    /**
     * Returns entropy value for this INDArray
     * @return
     */
    @Override
    public Number entropyNumber() {
        return entropy(Integer.MAX_VALUE).getDouble(0);
    }

    /**
     * Returns non-normalized Shannon entropy value for this INDArray
     * @return
     */
    @Override
    public Number shannonEntropyNumber() {
        return shannonEntropy(Integer.MAX_VALUE).getDouble(0);
    }


    /**
     * Returns log entropy value for this INDArray
     * @return
     */
    @Override
    public Number logEntropyNumber() {
        return logEntropy(Integer.MAX_VALUE).getDouble(0);
    }

    /**
     * Returns entropy along dimension
     * @param dimension
     * @return
     */
    @Override
    public INDArray entropy(int... dimension) {
        return Nd4j.getExecutioner().exec(new Entropy(this), dimension);
    }

    /**
     * Returns non-normalized Shannon entropy along dimension
     * @param dimension
     * @return
     */
    @Override
    public INDArray shannonEntropy(int... dimension) {
        return Nd4j.getExecutioner().exec(new ShannonEntropy(this), dimension);
    }

    /**
     * Returns log entropy along dimension
     * @param dimension
     * @return
     */
    @Override
    public INDArray logEntropy(int... dimension) {
        return Nd4j.getExecutioner().exec(new LogEntropy(this), dimension);
    }

    @Override
    public Number percentileNumber(Number quantile) {
        if (quantile.intValue() < 0 || quantile.intValue() > 100)
            throw new ND4JIllegalStateException("Percentile value should be in 0...100 range");

        if (isScalar())
            return this.getDouble(0);

        INDArray sorted = Nd4j.sort(this.dup(this.ordering()), true);

        return getPercentile(quantile, sorted);
    }

    @Override
    public Number medianNumber() {
        return percentileNumber(50);
    }

    @Override
    public INDArray median(int... dimension) {
        return percentile(50, dimension);
    }

    protected double getPercentile(Number quantile, INDArray sorted) {
        if (quantile.intValue() == 0)
            return sorted.getDouble(0);
        else if (quantile.intValue() == 100)
            return sorted.getDouble(sorted.length() - 1);

        double pos = (quantile.doubleValue() / 100.0) * (double) (sorted.length() + 1);

        double fposition = FastMath.floor(pos);
        int position = (int) fposition;

        double diff = pos - fposition;

        double lower = sorted.getDouble(position - 1);
        double upper = sorted.getDouble(position);

        return lower + diff * (upper - lower);
    }

    @Override
    public INDArray percentile(Number quantile, int... dimension) {
        if (quantile.doubleValue() < 0 || quantile.doubleValue() > 100)
            throw new ND4JIllegalStateException("Percentile value should be in 0...100 range");

        if (isScalar())
            return Nd4j.scalar(this.getDouble(0));

        INDArray sorted = Nd4j.getNDArrayFactory().sort(this.dup(this.ordering()), false, dimension);

        // there's no practical sense doing this on GPU, stride will be just size of TAD.
        INDArray ret = Nd4j.createUninitialized(sorted.tensorssAlongDimension(dimension));
        for (int i = 0; i < ret.length(); i++) {
            ret.putScalar(i, getPercentile(quantile, sorted.tensorAlongDimension(i, dimension)));
        }

        return ret;

    }

    @Override
    public long[] shapeInfoJava() {
        return javaShapeInformation;
    }

    @Override
    public DataType dataType() {
        return data().dataType();
    }
}
