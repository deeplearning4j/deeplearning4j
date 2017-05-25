package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.profiler.OpProfiler;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * @author Audrey Loeffel
 */
public class BaseSparseNDArrayCOO extends BaseSparseNDArray {
    protected static final SparseFormat format = SparseFormat.COO;
    protected transient volatile DataBuffer values;
    protected transient volatile DataBuffer indices;
    protected transient volatile DataBuffer slices;

    public BaseSparseNDArrayCOO(double[] values, int[][] indices, int[] shape){

        checkArgument(values.length == indices.length);
        checkArgument(values.length == 0 || indices[0].length == shape.length);

        this.values = Nd4j.createBuffer(values);
        this.indices = Nd4j.createBuffer(ArrayUtil.flatten(indices));
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);
        init(shape);
        this.nnz = values.length;
    }

    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, int[] shape){
        // TODO
    }
    public BaseSparseNDArrayCOO(float[] values, int[][] indices, int[] shape){
        // TODO
    }

    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, int[] strides, int offset, int[] shape){
        // TODO - create a view if strides and offset != 0
    }


    @Override
    public INDArray putScalar(int i, double value) {
        if (i < 0)
            i += rank();
        if (isScalar()) {
            if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.DISABLED)
                OpProfiler.getInstance().processScalarCall();

// TODO
            //data.put(i, value);
            return this;
        }

        if (isRowVector()) {
            return putScalar(0, i, value);
        } else if (isColumnVector()) {
            return putScalar(i, 0, value);
        }
        int[] indexes = ordering() == 'c' ? Shape.ind2subC(this, i) : Shape.ind2sub(this, i);
        return putScalar(indexes, value);
    }

    @Override
    public INDArray putScalar(int i, float value) {
        return putScalar(i, (double) value); //todo - move in baseSparse?
    }

    @Override
    public INDArray putScalar(int i, int value) {
        return putScalar(i, (double) value); //todo
    }

    @Override
    public INDArray putScalar(int[] indexes, double value) {

        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < 0)
                indexes[i] += rank();
        }
        //add directly without calling those methods ?
        if (indexes.length == 1) {
            return putScalar(indexes[0], value);
        } else if (indexes.length == 2) {
            return putScalar(indexes[0], indexes[1], value);
        } else if (indexes.length == 3) {
            return putScalar(indexes[0], indexes[1], indexes[2], value);
        } else if (indexes.length == 4) {
            return putScalar(indexes[0], indexes[1], indexes[2], indexes[3], value);
        } else {
            if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.DISABLED)
                OpProfiler.getInstance().processScalarCall();

            long offset = Shape.getOffset(shapeInformation, indexes);
            // TODO data.put(offset, value);
        }
        return this;
    }

    @Override
    public INDArray putScalar(int row, int col, double value) {
        return super.putScalar(row, col, value);
    }

    @Override
    public INDArray putScalar(int dim0, int dim1, int dim2, double value) {
        return super.putScalar(dim0, dim1, dim2, value);
    }

    @Override
    public INDArray putScalar(int dim0, int dim1, int dim2, int dim3, double value) {
        return super.putScalar(dim0, dim1, dim2, dim3, value);
    }

    @Override
    public INDArray putRow(int row, INDArray toPut) {
        return super.putRow(row, toPut);
    }

    @Override
    public INDArray putColumn(int column, INDArray toPut) {
        return super.putColumn(column, toPut);
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, INDArray element) {
        return super.put(indices, element);
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, Number element) {
        return super.put(indices, element);
    }

    @Override
    public INDArray put(int[] indices, INDArray element) {
        return super.put(indices, element);
    }

    @Override
    public INDArray put(int i, int j, INDArray element) { // TODO in base
        return put(new int[] {i, j}, element);
    }

    @Override
    public INDArray put(int i, int j, Number element) { // TODO in base
        return putScalar(new int[] {i, j}, element.doubleValue());
    }

    @Override
    public INDArray put(int i, INDArray element) { // TODO in base
        if (!element.isScalar())
            throw new IllegalArgumentException("Element must be a scalar");
        return putScalar(i, element.getDouble(0));
    }

    public void addOrUpdate(int[] indexes, double value) {
        for(int i = 0; i < nnz; i++){
            int[] idx = getIndicesOf(i).asInt();
            if(Arrays.equals(idx, indexes)){
                // There is already a non-null value at this index
                // -> update the current value
               values.put(i, value);
               break;
            } else {
                if(ArrayUtil.anyMore(indexes, idx)){
                    // It's a new non-null element
                    // The index and value should be added at the position i
                    // Need to shift the tail to make room for the new element
                    // /!\ buffer overflow
                    shiftRight(indices, i*rank(), rank(), rank() * length());
                    for(int j = 0; j < rank(); j++){
                        int currentIdx = rank() * i + j;
                        indices.put(currentIdx, indexes[j]);
                    }
                    shiftRight(values, i, 1, length());
                    values.put(i, value);
                    break;
                }

            }



        }
    }


    public DataBuffer shiftRight(DataBuffer buffer, int from, int offset, int dataLength){
        double[] tail = buffer.getDoublesAt(from, (int) dataLength - from);
        if(dataLength + offset > buffer.length()) {
            // TODO reallocate more memory space
        }
        for(int i = 0; i < tail.length; i++) {
            buffer.put(i + from + offset, tail[i]);
        }
        // TODO set "free" space to 0 ? or leave the old values there ?
        return buffer;
    }

    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    @Override
    public INDArray get(INDArrayIndex... indexes) {
        return null;
    }

    public SparseFormat getFormat(){
        return format;
    }

    @Override
    public DataBuffer data(){
        return Nd4j.getDataBufferFactory().create(values, 0, length());
    }

    /**
     * Returns the indices of non-zero element of the vector
     *
     * @return indices in Databuffer
     * */
    @Override
    public DataBuffer getVectorCoordinates() {
        if(isVector()){
            return indices;
        } else {
            throw new UnsupportedOperationException();
        }
    }

    /**
     * Converts the sparse ndarray into a dense one
     * @return a dense ndarray
     */
    @Override
    public INDArray toDense() {
        // TODO support view conversion
        INDArray result = Nd4j.zeros(shape());

        switch (data().dataType()){
            case DOUBLE:
                for(int i = 0; i < nnz; i++) {
                    int[] idx = getIndicesOf(i).asInt();
                    double value = values.getDouble(i);
                    result.putScalar(idx, value);
                }
                break;
            case FLOAT:
                for(int i = 0; i < nnz; i++) {
                    int[] idx = getIndicesOf(i).asInt();
                    float value = values.getFloat(i);
                    result.putScalar(idx, value);
                }
                break;
            default:
                throw new UnsupportedOperationException();
        }
        return result;
    }

    @Override
    public DataBuffer shapeInfoDataBuffer() {
        return shapeInformation;
    }

    @Override
    public INDArray subArray(ShapeOffsetResolution resolution) {
        return null;
    }

    @Override
    public INDArray subArray(int[] offsets, int[] shape, int[] stride) {
        throw new UnsupportedOperationException();
    }

    /**
     * Returns the indices of the element in the given index
     *
     * @param i the index of the element
     * @return a dataBuffer containing the indices of element
     * */
    public DataBuffer getIndicesOf(int i){
        int[] arr = Arrays.copyOfRange(indices.asInt(), rank()*i, rank());
        return Nd4j.getDataBufferFactory().createInt(arr);
    }

}
