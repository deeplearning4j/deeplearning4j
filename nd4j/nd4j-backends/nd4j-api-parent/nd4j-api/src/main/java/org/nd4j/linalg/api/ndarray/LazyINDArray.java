/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ndarray;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.Getter;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;
import org.nd4j.linalg.exception.Nd4jNoSuchWorkspaceException;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.nio.LongBuffer;
import java.util.List;

/**
 * A lightweight proxy representing a future result during GraphScope tracing.
 *
 * During tracing (before scope.end()):
 * - shape(), dataType(), rank() return traced metadata
 * - Data access methods throw IllegalStateException
 * - Can be passed as input to subsequent Nd4j.exec() calls for graph wiring
 *
 * After materialization (after scope.end()):
 * - All methods delegate to the real materialized INDArray
 *
 * @author Adam Gibson
 */
public class LazyINDArray implements INDArray {

    private static final long serialVersionUID = 1L;

    @Getter
    private final String sdVariableName;
    @Getter
    private final long[] tracedShape;
    @Getter
    private final DataType tracedDataType;
    private volatile INDArray materialized;

    public LazyINDArray(String sdVariableName, long[] tracedShape, DataType tracedDataType) {
        this.sdVariableName = sdVariableName;
        this.tracedShape = tracedShape;
        this.tracedDataType = tracedDataType;
    }

    public boolean isMaterialized() {
        return materialized != null;
    }

    public void materialize(INDArray real) {
        this.materialized = real;
    }

    public INDArray getMaterialized() {
        return materialized;
    }

    /**
     * Returns the materialized array or throws if not yet materialized.
     */
    private INDArray requireMaterialized() {
        if (materialized == null) {
            throw new IllegalStateException(
                    "LazyINDArray '" + sdVariableName + "' is not yet materialized. " +
                            "Call scope.end() before accessing data.");
        }
        return materialized;
    }

    // --- Shape/metadata methods work during tracing ---

    @Override
    public int rank() {
        if (materialized != null) return materialized.rank();
        return tracedShape.length;
    }

    @Override
    public long[] shape() {
        if (materialized != null) return materialized.shape();
        return tracedShape;
    }

    @Override
    public DataType dataType() {
        if (materialized != null) return materialized.dataType();
        return tracedDataType;
    }

    @Override
    public long length() {
        if (materialized != null) return materialized.length();
        long len = 1;
        for (long s : tracedShape) len *= s;
        return len;
    }

    @Override
    public long size(int dimension) {
        if (materialized != null) return materialized.size(dimension);
        if (dimension < 0) dimension += tracedShape.length;
        return tracedShape[dimension];
    }

    @Override
    public long[] stride() {
        if (materialized != null) return materialized.stride();
        // Compute C-order strides from traced shape
        long[] strides = new long[tracedShape.length];
        if (tracedShape.length > 0) {
            strides[tracedShape.length - 1] = 1;
            for (int i = tracedShape.length - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * tracedShape[i + 1];
            }
        }
        return strides;
    }

    @Override
    public boolean isEmpty() {
        if (materialized != null) return materialized.isEmpty();
        for (long s : tracedShape) {
            if (s == 0) return true;
        }
        return false;
    }

    @Override
    public boolean isScalar() {
        if (materialized != null) return materialized.isScalar();
        return tracedShape.length == 0 || (tracedShape.length == 1 && tracedShape[0] == 1);
    }

    @Override
    public boolean isVector() {
        if (materialized != null) return materialized.isVector();
        return tracedShape.length == 1 || (tracedShape.length == 2 && (tracedShape[0] == 1 || tracedShape[1] == 1));
    }

    @Override
    public boolean isMatrix() {
        if (materialized != null) return materialized.isMatrix();
        return tracedShape.length == 2;
    }

    @Override
    public boolean isRowVector() {
        if (materialized != null) return materialized.isRowVector();
        return tracedShape.length == 2 && tracedShape[0] == 1;
    }

    @Override
    public boolean isColumnVector() {
        if (materialized != null) return materialized.isColumnVector();
        return tracedShape.length == 2 && tracedShape[1] == 1;
    }

    @Override
    public int rows() {
        if (materialized != null) return materialized.rows();
        return (int) (tracedShape.length >= 1 ? tracedShape[0] : 1);
    }

    @Override
    public int columns() {
        if (materialized != null) return materialized.columns();
        return (int) (tracedShape.length >= 2 ? tracedShape[1] : 1);
    }

    @Override
    public char ordering() {
        if (materialized != null) return materialized.ordering();
        return 'c';
    }

    // --- All data access methods delegate to materialized or throw ---

    @Override public OpaqueNDArray getOrCreateOpaqueNDArray() { return requireMaterialized().getOrCreateOpaqueNDArray(); }
    @Override public void clearOpaqueNDArray() { requireMaterialized().clearOpaqueNDArray(); }
    @Override public INDArray castTo(DataType dataType) { return requireMaterialized().castTo(dataType); }
    @Override public boolean isR() { return requireMaterialized().isR(); }
    @Override public boolean isZ() { return requireMaterialized().isZ(); }
    @Override public boolean isB() { return requireMaterialized().isB(); }
    @Override public boolean isS() { return requireMaterialized().isS(); }
    @Override public INDArray reshape(char order, long... newShape) { return requireMaterialized().reshape(order, newShape); }
    @Override public INDArray reshape(char order, boolean enforceView, long... newShape) { return requireMaterialized().reshape(order, enforceView, newShape); }
    @Override public INDArray reshape(long... newShape) { return requireMaterialized().reshape(newShape); }
    @Override public INDArray reshape(char order, int rows, int columns) { return requireMaterialized().reshape(order, rows, columns); }
    @Override public INDArray reshape(long rows, long columns) { return requireMaterialized().reshape(rows, columns); }
    @Override public INDArray transpose() { return requireMaterialized().transpose(); }
    @Override public INDArray transposei() { return requireMaterialized().transposei(); }
    @Override public INDArray swapAxes(int dimension, int with) { return requireMaterialized().swapAxes(dimension, with); }
    @Override public INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable) { return requireMaterialized().dimShuffle(rearrange, newOrder, broadCastable); }
    @Override public INDArray broadcast(long... shape) { return requireMaterialized().broadcast(shape); }
    @Override public INDArray broadcast(INDArray result) { return requireMaterialized().broadcast(result); }
    @Override public Object element() { return requireMaterialized().element(); }
    @Override public DataBuffer data() { return requireMaterialized().data(); }
    @Override public void setData(DataBuffer data) { requireMaterialized().setData(data); }
    @Override public long vectorsAlongDimension(int dimension) { return requireMaterialized().vectorsAlongDimension(dimension); }
    @Override public INDArray vectorAlongDimension(int index, int dimension) { return requireMaterialized().vectorAlongDimension(index, dimension); }
    @Override public INDArray getColumn(long i) { return requireMaterialized().getColumn(i); }
    @Override public INDArray getColumn(long i, boolean keepDim) { return requireMaterialized().getColumn(i, keepDim); }
    @Override public INDArray getRow(long i) { return requireMaterialized().getRow(i); }
    @Override public INDArray getRow(long i, boolean keepDim) { return requireMaterialized().getRow(i, keepDim); }
    @Override public int getTrailingOnes() { return requireMaterialized().getTrailingOnes(); }
    @Override public int getLeadingOnes() { return requireMaterialized().getLeadingOnes(); }
    @Override public INDArray slice(long i) { return requireMaterialized().slice(i); }
    @Override public INDArray slice(long i, int dimension) { return requireMaterialized().slice(i, dimension); }
    @Override public long slices() { return requireMaterialized().slices(); }
    @Override public INDArray get(INDArrayIndex... indexes) { return requireMaterialized().get(indexes); }
    @Override public INDArray getColumns(int... columns) { return requireMaterialized().getColumns(columns); }
    @Override public INDArray getRows(int... rows) { return requireMaterialized().getRows(rows); }
    @Override public INDArray dup() { return requireMaterialized().dup(); }
    @Override public INDArray dup(char order) { return requireMaterialized().dup(order); }
    @Override public INDArray ravel() { return requireMaterialized().ravel(); }
    @Override public INDArray ravel(char order) { return requireMaterialized().ravel(order); }
    @Override public double getDoubleUnsafe(long offset) { return requireMaterialized().getDoubleUnsafe(offset); }
    @Override public String getString(long index) { return requireMaterialized().getString(index); }
    @Override public INDArray putScalarUnsafe(long offset, double value) { return requireMaterialized().putScalarUnsafe(offset, value); }
    @Override public INDArray putScalar(long i, double value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray putScalar(long i, float value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray putScalar(long i, int value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray putScalar(long row, long col, double value) { return requireMaterialized().putScalar(row, col, value); }
    @Override public INDArray putScalar(long dim0, long dim1, long dim2, double value) { return requireMaterialized().putScalar(dim0, dim1, dim2, value); }
    @Override public INDArray putScalar(long dim0, long dim1, long dim2, long dim3, double value) { return requireMaterialized().putScalar(dim0, dim1, dim2, dim3, value); }
    @Override public INDArray putScalar(long[] i, double value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray putScalar(long[] i, float value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray putScalar(long[] i, int value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray lt(Number other) { return requireMaterialized().lt(other); }
    @Override public INDArray lte(Number other) { return requireMaterialized().lte(other); }
    @Override public INDArray eq(Number other) { return requireMaterialized().eq(other); }
    @Override public INDArray gt(Number other) { return requireMaterialized().gt(other); }
    @Override public INDArray gte(Number other) { return requireMaterialized().gte(other); }
    @Override public INDArray lt(INDArray other) { return requireMaterialized().lt(other); }
    @Override public INDArray eps(INDArray other) { return requireMaterialized().eps(other); }
    @Override public INDArray eq(INDArray other) { return requireMaterialized().eq(other); }
    @Override public INDArray neq(Number other) { return requireMaterialized().neq(other); }
    @Override public INDArray neq(INDArray other) { return requireMaterialized().neq(other); }
    @Override public INDArray gt(INDArray other) { return requireMaterialized().gt(other); }
    @Override public INDArray neg() { return requireMaterialized().neg(); }
    @Override public INDArray negi() { return requireMaterialized().negi(); }
    @Override public INDArray rdiv(Number n) { return requireMaterialized().rdiv(n); }
    @Override public INDArray rdivi(Number n) { return requireMaterialized().rdivi(n); }
    @Override public INDArray rsub(Number n) { return requireMaterialized().rsub(n); }
    @Override public INDArray rsubi(Number n) { return requireMaterialized().rsubi(n); }
    @Override public INDArray div(Number n) { return requireMaterialized().div(n); }
    @Override public INDArray divi(Number n) { return requireMaterialized().divi(n); }
    @Override public INDArray mul(Number n) { return requireMaterialized().mul(n); }
    @Override public INDArray muli(Number n) { return requireMaterialized().muli(n); }
    @Override public INDArray sub(Number n) { return requireMaterialized().sub(n); }
    @Override public INDArray subi(Number n) { return requireMaterialized().subi(n); }
    @Override public INDArray add(Number n) { return requireMaterialized().add(n); }
    @Override public INDArray addi(Number n) { return requireMaterialized().addi(n); }
    @Override public INDArray rdiv(Number n, INDArray result) { return requireMaterialized().rdiv(n, result); }
    @Override public INDArray rdivi(Number n, INDArray result) { return requireMaterialized().rdivi(n, result); }
    @Override public INDArray rsub(Number n, INDArray result) { return requireMaterialized().rsub(n, result); }
    @Override public INDArray rsubi(Number n, INDArray result) { return requireMaterialized().rsubi(n, result); }
    @Override public INDArray div(Number n, INDArray result) { return requireMaterialized().div(n, result); }
    @Override public INDArray divi(Number n, INDArray result) { return requireMaterialized().divi(n, result); }
    @Override public INDArray mul(Number n, INDArray result) { return requireMaterialized().mul(n, result); }
    @Override public INDArray muli(Number n, INDArray result) { return requireMaterialized().muli(n, result); }
    @Override public INDArray sub(Number n, INDArray result) { return requireMaterialized().sub(n, result); }
    @Override public INDArray subi(Number n, INDArray result) { return requireMaterialized().subi(n, result); }
    @Override public INDArray add(Number n, INDArray result) { return requireMaterialized().add(n, result); }
    @Override public INDArray addi(Number n, INDArray result) { return requireMaterialized().addi(n, result); }
    @Override public INDArray mmul(INDArray other) { return requireMaterialized().mmul(other); }
    @Override public double[][] toDoubleMatrix() { return requireMaterialized().toDoubleMatrix(); }
    @Override public double[] toDoubleVector() { return requireMaterialized().toDoubleVector(); }
    @Override public float[] toFloatVector() { return requireMaterialized().toFloatVector(); }
    @Override public float[][] toFloatMatrix() { return requireMaterialized().toFloatMatrix(); }
    @Override public int[] toIntVector() { return requireMaterialized().toIntVector(); }
    @Override public long[] toLongVector() { return requireMaterialized().toLongVector(); }
    @Override public long[][] toLongMatrix() { return requireMaterialized().toLongMatrix(); }
    @Override public int[][] toIntMatrix() { return requireMaterialized().toIntMatrix(); }
    @Override public INDArray mmul(INDArray other, INDArray result) { return requireMaterialized().mmul(other, result); }
    @Override public INDArray mmul(INDArray other, MMulTranspose mMulTranspose) { return requireMaterialized().mmul(other, mMulTranspose); }
    @Override public INDArray div(INDArray other) { return requireMaterialized().div(other); }
    @Override public INDArray div(INDArray other, INDArray result) { return requireMaterialized().div(other, result); }
    @Override public INDArray mul(INDArray other) { return requireMaterialized().mul(other); }
    @Override public INDArray mul(INDArray other, INDArray result) { return requireMaterialized().mul(other, result); }
    @Override public INDArray sub(INDArray other) { return requireMaterialized().sub(other); }
    @Override public INDArray sub(INDArray other, INDArray result) { return requireMaterialized().sub(other, result); }
    @Override public INDArray add(INDArray other) { return requireMaterialized().add(other); }
    @Override public INDArray add(INDArray other, INDArray result) { return requireMaterialized().add(other, result); }
    @Override public INDArray mmuli(INDArray other) { return requireMaterialized().mmuli(other); }
    @Override public INDArray mmuli(INDArray other, INDArray result) { return requireMaterialized().mmuli(other, result); }
    @Override public INDArray divi(INDArray other) { return requireMaterialized().divi(other); }
    @Override public INDArray divi(INDArray other, INDArray result) { return requireMaterialized().divi(other, result); }
    @Override public INDArray muli(INDArray other) { return requireMaterialized().muli(other); }
    @Override public INDArray muli(INDArray other, INDArray result) { return requireMaterialized().muli(other, result); }
    @Override public INDArray subi(INDArray other) { return requireMaterialized().subi(other); }
    @Override public INDArray subi(INDArray other, INDArray result) { return requireMaterialized().subi(other, result); }
    @Override public INDArray addi(INDArray other) { return requireMaterialized().addi(other); }
    @Override public INDArray addi(INDArray other, INDArray result) { return requireMaterialized().addi(other, result); }
    @Override public INDArray rdiv(INDArray other) { return requireMaterialized().rdiv(other); }
    @Override public INDArray rdivi(INDArray other) { return requireMaterialized().rdivi(other); }
    @Override public INDArray rdiv(INDArray other, INDArray result) { return requireMaterialized().rdiv(other, result); }
    @Override public INDArray rdivi(INDArray other, INDArray result) { return requireMaterialized().rdivi(other, result); }
    @Override public INDArray rsub(INDArray other, INDArray result) { return requireMaterialized().rsub(other, result); }
    @Override public INDArray rsub(INDArray other) { return requireMaterialized().rsub(other); }
    @Override public INDArray rsubi(INDArray other) { return requireMaterialized().rsubi(other); }
    @Override public INDArray rsubi(INDArray other, INDArray result) { return requireMaterialized().rsubi(other, result); }
    @Override public Number stdNumber(boolean biasCorrected) { return requireMaterialized().stdNumber(biasCorrected); }
    @Override public Number getNumber(long i) { return requireMaterialized().getNumber(i); }
    @Override public Number getNumber(long... idx) { return requireMaterialized().getNumber(idx); }
    @Override public double getDouble(long i) { return requireMaterialized().getDouble(i); }
    @Override public double getDouble(long i, long j) { return requireMaterialized().getDouble(i, j); }
    @Override public float getFloat(long i) { return requireMaterialized().getFloat(i); }
    @Override public float getFloat(long i, long j) { return requireMaterialized().getFloat(i, j); }
    @Override public double getDouble(long... indices) { return requireMaterialized().getDouble(indices); }
    @Override public float getFloat(long... indices) { return requireMaterialized().getFloat(indices); }
    @Override public int getInt(int... indices) { return requireMaterialized().getInt(indices); }
    @Override public long getLong(long index) { return requireMaterialized().getLong(index); }
    @Override public long getLong(long... indices) { return requireMaterialized().getLong(indices); }
    @Override public INDArray putRow(long row, INDArray toPut) { return requireMaterialized().putRow(row, toPut); }
    @Override public INDArray putColumn(int column, INDArray toPut) { return requireMaterialized().putColumn(column, toPut); }
    @Override public INDArray put(INDArrayIndex[] indices, INDArray element) { return requireMaterialized().put(indices, element); }
    @Override public INDArray put(INDArrayIndex[] indices, Number element) { return requireMaterialized().put(indices, element); }
    @Override public INDArray put(int[] indices, INDArray element) { return requireMaterialized().put(indices, element); }
    @Override public INDArray put(int i, int j, INDArray element) { return requireMaterialized().put(i, j, element); }
    @Override public INDArray put(int i, int j, Number element) { return requireMaterialized().put(i, j, element); }
    @Override public INDArray put(int i, INDArray element) { return requireMaterialized().put(i, element); }
    @Override public Number scan(Condition condition) { return requireMaterialized().scan(condition); }
    @Override public INDArray assign(INDArray arr) { return requireMaterialized().assign(arr); }
    @Override public INDArray putScalar(int[] i, double value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray putScalar(int[] i, float value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray putScalar(int[] i, int value) { return requireMaterialized().putScalar(i, value); }
    @Override public INDArray cond(Condition condition) { return requireMaterialized().cond(condition); }
    @Override public INDArray repmat(int... shape) { return requireMaterialized().repmat(shape); }
    @Override public INDArray repeat(int dimension, long... repeats) { return requireMaterialized().repeat(dimension, repeats); }
    @Override public INDArray assign(Number value) { return requireMaterialized().assign(value); }
    @Override public void sliceVectors(List<INDArray> list) { requireMaterialized().sliceVectors(list); }
    @Override public INDArray putSlice(int slice, INDArray put) { return requireMaterialized().putSlice(slice, put); }
    @Override public INDArray match(INDArray comp, Condition condition) { return requireMaterialized().match(comp, condition); }
    @Override public INDArray match(Number comp, Condition condition) { return requireMaterialized().match(comp, condition); }
    @Override public INDArray getWhere(INDArray comp, Condition condition) { return requireMaterialized().getWhere(comp, condition); }
    @Override public INDArray getWhere(Number comp, Condition condition) { return requireMaterialized().getWhere(comp, condition); }
    @Override public INDArray putWhere(INDArray comp, INDArray put, Condition condition) { return requireMaterialized().putWhere(comp, put, condition); }
    @Override public INDArray putWhere(Number comp, INDArray put, Condition condition) { return requireMaterialized().putWhere(comp, put, condition); }
    @Override public INDArray putWhere(Number comp, Number put, Condition condition) { return requireMaterialized().putWhere(comp, put, condition); }
    @Override public INDArray putWhereWithMask(INDArray mask, INDArray put) { return requireMaterialized().putWhereWithMask(mask, put); }
    @Override public INDArray putWhereWithMask(INDArray mask, Number put) { return requireMaterialized().putWhereWithMask(mask, put); }
    @Override public Number ameanNumber() { return requireMaterialized().ameanNumber(); }
    @Override public long offset() { return requireMaterialized().offset(); }
    @Override public int elementWiseStride() { return requireMaterialized().elementWiseStride(); }
    @Override public boolean isView() { if (materialized != null) return materialized.isView(); return false; }
    @Override public boolean isSparse() { if (materialized != null) return materialized.isSparse(); return false; }
    @Override public DataBuffer shapeInfoDataBuffer() { return requireMaterialized().shapeInfoDataBuffer(); }
    @Override public LongBuffer shapeInfo() { return requireMaterialized().shapeInfo(); }
    @Override public boolean equalShapes(INDArray other) { return requireMaterialized().equalShapes(other); }
    @Override public INDArray getScalar(long row, long column) { return requireMaterialized().getScalar(row, column); }
    @Override public INDArray getScalar(long i) { return requireMaterialized().getScalar(i); }
    @Override public boolean equalsWithEps(Object o, double eps) { return requireMaterialized().equalsWithEps(o, eps); }
    @Override public INDArray remainder(INDArray denominator) { return requireMaterialized().remainder(denominator); }
    @Override public INDArray remainder(INDArray denominator, INDArray result) { return requireMaterialized().remainder(denominator, result); }
    @Override public INDArray remainder(Number denominator) { return requireMaterialized().remainder(denominator); }
    @Override public INDArray remainder(Number denominator, INDArray result) { return requireMaterialized().remainder(denominator, result); }
    @Override public INDArray remainderi(INDArray denominator) { return requireMaterialized().remainderi(denominator); }
    @Override public INDArray remainderi(Number denominator) { return requireMaterialized().remainderi(denominator); }
    @Override public INDArray fmod(INDArray denominator) { return requireMaterialized().fmod(denominator); }
    @Override public INDArray fmod(INDArray denominator, INDArray result) { return requireMaterialized().fmod(denominator, result); }
    @Override public INDArray fmod(Number denominator) { return requireMaterialized().fmod(denominator); }
    @Override public INDArray fmod(Number denominator, INDArray result) { return requireMaterialized().fmod(denominator, result); }
    @Override public INDArray fmodi(INDArray denominator) { return requireMaterialized().fmodi(denominator); }
    @Override public INDArray fmodi(Number denominator) { return requireMaterialized().fmodi(denominator); }
    @Override public boolean isAttached() { if (materialized != null) return materialized.isAttached(); return false; }
    @Override public boolean isInScope() { if (materialized != null) return materialized.isInScope(); return false; }
    @Override public INDArray detach() { return requireMaterialized().detach(); }
    @Override public INDArray leverage() { return requireMaterialized().leverage(); }
    @Override public INDArray leverageTo(String id) { return requireMaterialized().leverageTo(id); }
    @Override public INDArray leverageTo(String id, boolean enforceExistence) throws Nd4jNoSuchWorkspaceException { return requireMaterialized().leverageTo(id, enforceExistence); }
    @Override public INDArray leverageOrDetach(String id) { return requireMaterialized().leverageOrDetach(id); }
    @Override public INDArray migrate() { return requireMaterialized().migrate(); }
    @Override public INDArray migrate(boolean detachable) { return requireMaterialized().migrate(detachable); }
    @Override public Number percentileNumber(Number percentile) { return requireMaterialized().percentileNumber(percentile); }
    @Override public Number medianNumber() { return requireMaterialized().medianNumber(); }
    @Override public INDArray percentile(Number percentile, long... dimension) { return requireMaterialized().percentile(percentile, dimension); }
    @Override public long getId() { return requireMaterialized().getId(); }
    @Override public INDArray like() { return requireMaterialized().like(); }
    @Override public INDArray ulike() { return requireMaterialized().ulike(); }
    @Override public boolean closeable() { if (materialized != null) return materialized.closeable(); return false; }
    @Override public void close() { if (materialized != null) materialized.close(); }
    @Override public boolean wasClosed() { if (materialized != null) return materialized.wasClosed(); return false; }
    @Override public void setCloseable(boolean closeable) { if (materialized != null) materialized.setCloseable(closeable); }
    @Override public INDArray assign(boolean value) { return requireMaterialized().assign(value); }
    @Override public boolean all() { return requireMaterialized().all(); }
    @Override public boolean any() { return requireMaterialized().any(); }
    @Override public boolean none() { return requireMaterialized().none(); }
    @Override public LongShapeDescriptor shapeDescriptor() {
        if (materialized != null) return materialized.shapeDescriptor();
        return LongShapeDescriptor.fromShape(tracedShape, tracedDataType);
    }

    @Override
    public void addEvent(NDArrayEvent event) {
        if (materialized != null) materialized.addEvent(event);
    }

    @Override
    public boolean isCompressed() {
        if (materialized != null) return materialized.isCompressed();
        return false;
    }

    @Override
    public void markAsCompressed(boolean compressed) {
        requireMaterialized().markAsCompressed(compressed);
    }

    @Override
    public int toFlatArray(FlatBufferBuilder builder) {
        return requireMaterialized().toFlatArray(builder);
    }

    @Override
    public INDArray unsafeDuplication() {
        return requireMaterialized().unsafeDuplication();
    }

    @Override
    public INDArray unsafeDuplication(boolean blocking) {
        return requireMaterialized().unsafeDuplication(blocking);
    }

    @Override
    public long[] shapeInfoJava() {
        return requireMaterialized().shapeInfoJava();
    }

    public long lengthLong() {
        return length();
    }

    @Override
    public String toString() {
        if (materialized != null) return materialized.toString();
        return "LazyINDArray{var=" + sdVariableName + ", shape=" + java.util.Arrays.toString(tracedShape)
                + ", dtype=" + tracedDataType + ", materialized=false}";
    }

    @Override
    public boolean equals(Object o) {
        if (materialized != null) return materialized.equals(o);
        return this == o;
    }

    @Override
    public int hashCode() {
        if (materialized != null) return materialized.hashCode();
        return System.identityHashCode(this);
    }

    @Override
    public void setShapeAndStride(int[] shape, int[] stride) {
        requireMaterialized().setShapeAndStride(shape, stride);
    }

    // --- Additional missing INDArray methods ---

    @Override public Nd4jEventLog log() { if (materialized != null) return materialized.log(); return null; }
    @Override public String shapeInfoToString() { return requireMaterialized().shapeInfoToString(); }
    @Override public int stride(int dimension) { return requireMaterialized().stride(dimension); }
    @Override public INDArray cumsumi(int dimension) { return requireMaterialized().cumsumi(dimension); }
    @Override public INDArray cumsum(int dimension) { return requireMaterialized().cumsum(dimension); }
    @Override public INDArray assignIf(INDArray arr, Condition condition) { return requireMaterialized().assignIf(arr, condition); }
    @Override public INDArray replaceWhere(INDArray arr, Condition condition) { return requireMaterialized().replaceWhere(arr, condition); }
    @Override public INDArray putScalar(long i, boolean b) { return requireMaterialized().putScalar(i, b); }
    @Override public INDArray eps(Number other) { return requireMaterialized().eps(other); }
    @Override public INDArray isInfinite() { return requireMaterialized().isInfinite(); }
    @Override public INDArray isNaN() { return requireMaterialized().isNaN(); }
    @Override public INDArray get(INDArray indices) { return requireMaterialized().get(indices); }
    @Override public INDArray put(INDArray indices, INDArray element) { return requireMaterialized().put(indices, element); }
    @Override public INDArray diviColumnVector(INDArray columnVector) { return requireMaterialized().diviColumnVector(columnVector); }
    @Override public INDArray divColumnVector(INDArray columnVector) { return requireMaterialized().divColumnVector(columnVector); }
    @Override public INDArray diviRowVector(INDArray rowVector) { return requireMaterialized().diviRowVector(rowVector); }
    @Override public INDArray divRowVector(INDArray rowVector) { return requireMaterialized().divRowVector(rowVector); }
    @Override public INDArray rdiviColumnVector(INDArray columnVector) { return requireMaterialized().rdiviColumnVector(columnVector); }
    @Override public INDArray rdivColumnVector(INDArray columnVector) { return requireMaterialized().rdivColumnVector(columnVector); }
    @Override public INDArray rdiviRowVector(INDArray rowVector) { return requireMaterialized().rdiviRowVector(rowVector); }
    @Override public INDArray rdivRowVector(INDArray rowVector) { return requireMaterialized().rdivRowVector(rowVector); }
    @Override public INDArray muliColumnVector(INDArray columnVector) { return requireMaterialized().muliColumnVector(columnVector); }
    @Override public INDArray mulColumnVector(INDArray columnVector) { return requireMaterialized().mulColumnVector(columnVector); }
    @Override public INDArray muliRowVector(INDArray rowVector) { return requireMaterialized().muliRowVector(rowVector); }
    @Override public INDArray mulRowVector(INDArray rowVector) { return requireMaterialized().mulRowVector(rowVector); }
    @Override public INDArray rsubiColumnVector(INDArray columnVector) { return requireMaterialized().rsubiColumnVector(columnVector); }
    @Override public INDArray rsubColumnVector(INDArray columnVector) { return requireMaterialized().rsubColumnVector(columnVector); }
    @Override public INDArray rsubiRowVector(INDArray rowVector) { return requireMaterialized().rsubiRowVector(rowVector); }
    @Override public INDArray rsubRowVector(INDArray rowVector) { return requireMaterialized().rsubRowVector(rowVector); }
    @Override public INDArray subiColumnVector(INDArray columnVector) { return requireMaterialized().subiColumnVector(columnVector); }
    @Override public INDArray subColumnVector(INDArray columnVector) { return requireMaterialized().subColumnVector(columnVector); }
    @Override public INDArray subiRowVector(INDArray rowVector) { return requireMaterialized().subiRowVector(rowVector); }
    @Override public INDArray subRowVector(INDArray rowVector) { return requireMaterialized().subRowVector(rowVector); }
    @Override public INDArray addiColumnVector(INDArray columnVector) { return requireMaterialized().addiColumnVector(columnVector); }
    @Override public INDArray putiColumnVector(INDArray columnVector) { return requireMaterialized().putiColumnVector(columnVector); }
    @Override public INDArray addColumnVector(INDArray columnVector) { return requireMaterialized().addColumnVector(columnVector); }
    @Override public INDArray addiRowVector(INDArray rowVector) { return requireMaterialized().addiRowVector(rowVector); }
    @Override public INDArray putiRowVector(INDArray rowVector) { return requireMaterialized().putiRowVector(rowVector); }
    @Override public INDArray addRowVector(INDArray rowVector) { return requireMaterialized().addRowVector(rowVector); }
    @Override public INDArray mmul(INDArray other, char resultOrder) { return requireMaterialized().mmul(other, resultOrder); }
    @Override public INDArray mmul(INDArray other, INDArray result, MMulTranspose mMulTranspose) { return requireMaterialized().mmul(other, result, mMulTranspose); }
    @Override public INDArray mmuli(INDArray other, MMulTranspose transpose) { return requireMaterialized().mmuli(other, transpose); }
    @Override public INDArray mmuli(INDArray other, INDArray result, MMulTranspose transpose) { return requireMaterialized().mmuli(other, result, transpose); }
    @Override public double squaredDistance(INDArray other) { return requireMaterialized().squaredDistance(other); }
    @Override public double distance2(INDArray other) { return requireMaterialized().distance2(other); }
    @Override public double distance1(INDArray other) { return requireMaterialized().distance1(other); }
    @Override public Number normmaxNumber() { return requireMaterialized().normmaxNumber(); }
    @Override public Number norm2Number() { return requireMaterialized().norm2Number(); }
    @Override public Number norm1Number() { return requireMaterialized().norm1Number(); }
    @Override public Number stdNumber() { return requireMaterialized().stdNumber(); }
    @Override public Number prodNumber() { return requireMaterialized().prodNumber(); }
    @Override public Number meanNumber() { return requireMaterialized().meanNumber(); }
    @Override public Number varNumber() { return requireMaterialized().varNumber(); }
    @Override public Number maxNumber() { return requireMaterialized().maxNumber(); }
    @Override public Number amaxNumber() { return requireMaterialized().amaxNumber(); }
    @Override public Number minNumber() { return requireMaterialized().minNumber(); }
    @Override public Number aminNumber() { return requireMaterialized().aminNumber(); }
    @Override public Number sumNumber() { return requireMaterialized().sumNumber(); }
    @Override public Number entropyNumber() { return requireMaterialized().entropyNumber(); }
    @Override public Number shannonEntropyNumber() { return requireMaterialized().shannonEntropyNumber(); }
    @Override public Number logEntropyNumber() { return requireMaterialized().logEntropyNumber(); }
    @Override public INDArray repmat(long... shape) { return requireMaterialized().repmat(shape); }
    @Override public long originalOffset() { return requireMaterialized().originalOffset(); }
    @Override public INDArray reshape(char order, int... newShape) { return requireMaterialized().reshape(order, newShape); }
    @Override public INDArray reshape(int[] shape) { return requireMaterialized().reshape(shape); }
    @Override public INDArray permute(long... rearrange) { return requireMaterialized().permute(rearrange); }
    @Override public INDArray permutei(long... rearrange) { return requireMaterialized().permutei(rearrange); }
    @Override public INDArray dimShuffle(Object[] rearrange, long[] newOrder, boolean[] broadCastable) { return requireMaterialized().dimShuffle(rearrange, newOrder, broadCastable); }
    @Override public INDArray getScalar(int... indices) { return requireMaterialized().getScalar(indices); }
    @Override public INDArray getScalar(long... indices) { return requireMaterialized().getScalar(indices); }
    @Override public double getDouble(int... indices) { return requireMaterialized().getDouble(indices); }
    @Override public float getFloat(int... indices) { return requireMaterialized().getFloat(indices); }
    @Override public boolean isColumnVectorOrScalar() { if (materialized != null) return materialized.isColumnVectorOrScalar(); return isColumnVector() || isScalar(); }
    @Override public boolean isRowVectorOrScalar() { if (materialized != null) return materialized.isRowVectorOrScalar(); return isRowVector() || isScalar(); }
    @Override public boolean isVectorOrScalar() { if (materialized != null) return materialized.isVectorOrScalar(); return isVector() || isScalar(); }
    @Override public boolean isSquare() { if (materialized != null) return materialized.isSquare(); return tracedShape.length == 2 && tracedShape[0] == tracedShape[1]; }
    @Override public long tensorsAlongDimension(long... dimension) { return requireMaterialized().tensorsAlongDimension(dimension); }
    @Override public INDArray tensorAlongDimension(long index, long... dimension) { return requireMaterialized().tensorAlongDimension(index, dimension); }
    @Override public INDArray norm1(long... dimension) { return requireMaterialized().norm1(dimension); }
    @Override public INDArray norm1(boolean keepDims, long... dimension) { return requireMaterialized().norm1(keepDims, dimension); }
    @Override public INDArray norm2(long... dimension) { return requireMaterialized().norm2(dimension); }
    @Override public INDArray norm2(boolean keepDims, long... dimension) { return requireMaterialized().norm2(keepDims, dimension); }
    @Override public INDArray normmax(long... dimension) { return requireMaterialized().normmax(dimension); }
    @Override public INDArray normmax(boolean keepDims, long... dimension) { return requireMaterialized().normmax(keepDims, dimension); }
    @Override public INDArray std(long... dimension) { return requireMaterialized().std(dimension); }
    @Override public INDArray std(boolean biasCorrected, long... dimension) { return requireMaterialized().std(biasCorrected, dimension); }
    @Override public INDArray std(boolean biasCorrected, boolean keepDims, long... dimension) { return requireMaterialized().std(biasCorrected, keepDims, dimension); }
    @Override public INDArray prod(long... dimension) { return requireMaterialized().prod(dimension); }
    @Override public INDArray prod(boolean keepDims, long... dimension) { return requireMaterialized().prod(keepDims, dimension); }
    @Override public INDArray mean(long... dimension) { return requireMaterialized().mean(dimension); }
    @Override public INDArray mean(INDArray result, long... dimension) { return requireMaterialized().mean(result, dimension); }
    @Override public INDArray mean(boolean keepDims, long... dimension) { return requireMaterialized().mean(keepDims, dimension); }
    @Override public INDArray mean(INDArray result, boolean keepDims, long... dimension) { return requireMaterialized().mean(result, keepDims, dimension); }
    @Override public INDArray amean(long... dimension) { return requireMaterialized().amean(dimension); }
    @Override public INDArray var(long... dimension) { return requireMaterialized().var(dimension); }
    @Override public INDArray var(boolean biasCorrected, long... dimension) { return requireMaterialized().var(biasCorrected, dimension); }
    @Override public INDArray max(long... dimension) { return requireMaterialized().max(dimension); }
    @Override public INDArray max(boolean keepDims, long... dimension) { return requireMaterialized().max(keepDims, dimension); }
    @Override public INDArray amax(long... dimension) { return requireMaterialized().amax(dimension); }
    @Override public INDArray min(long... dimension) { return requireMaterialized().min(dimension); }
    @Override public INDArray min(boolean keepDims, long... dimension) { return requireMaterialized().min(keepDims, dimension); }
    @Override public INDArray amin(long... dimension) { return requireMaterialized().amin(dimension); }
    @Override public INDArray sum(long... dimension) { return requireMaterialized().sum(dimension); }
    @Override public INDArray sum(boolean keepDims, long... dimension) { return requireMaterialized().sum(keepDims, dimension); }
    @Override public INDArray sum(INDArray result, long... dimension) { return requireMaterialized().sum(result, dimension); }
    @Override public INDArray sum(INDArray result, boolean keepDims, long... dimension) { return requireMaterialized().sum(result, keepDims, dimension); }
    @Override public INDArray entropy(long... dimension) { return requireMaterialized().entropy(dimension); }
    @Override public INDArray shannonEntropy(long... dimension) { return requireMaterialized().shannonEntropy(dimension); }
    @Override public INDArray logEntropy(long... dimension) { return requireMaterialized().logEntropy(dimension); }
    @Override public void setOrder(char order) { requireMaterialized().setOrder(order); }
    @Override public INDArray argMax(long... dimension) { return requireMaterialized().argMax(dimension); }
    @Override public INDArray median(long... dimension) { return requireMaterialized().median(dimension); }
    @Override public String toString(NDArrayStrings options) { return requireMaterialized().toString(options); }
    @Override public String toString(long maxElements, boolean forceSummarize, int precision) { return requireMaterialized().toString(maxElements, forceSummarize, precision); }
    @Override public String toStringFull() { return requireMaterialized().toStringFull(); }
    @Override public StackTraceElement[] allocationTrace() { if (materialized != null) return materialized.allocationTrace(); return null; }
    @Override public List<NDArrayEvent> writeEvents() { return requireMaterialized().writeEvents(); }
}
