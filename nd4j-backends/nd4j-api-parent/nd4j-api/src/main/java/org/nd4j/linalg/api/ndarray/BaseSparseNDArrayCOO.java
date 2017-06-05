package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;
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

        checkArgument(indices.length == shape.length);
        checkArgument(values.length == indices[0].length);
        ;
        // TODO - check if the coordinates are correctly sorted
        this.values = Nd4j.createBuffer(values);
        this.indices = Nd4j.createBuffer(ArrayUtil.flatten(indices));
        System.out.println(indices.toString());
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);
        init(shape);
        this.nnz = values.length;
    }

    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, int[] shape){
        checkArgument(values.length() * shape.length == indices.length());
        // TODO - check if the coordinates are correctly sorted
        this.values =  Nd4j.createBuffer(values, 0, values.length());
        this.indices = indices;
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);
        init(shape);
        this.nnz = values.length();
    }
    public BaseSparseNDArrayCOO(float[] values, int[][] indices, int[] shape){
        checkArgument(values.length == indices.length);
        checkArgument(values.length == 0 || indices[0].length == shape.length);
        // TODO - check if the coordinates are correctly sorted
        this.values = Nd4j.createBuffer(values);
        this.indices = Nd4j.createBuffer(ArrayUtil.flatten(indices));
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);
        init(shape);
        this.nnz = values.length;
    }

    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, int[] stride, int offset, int[] shape, char ordering){
        // TODO - create a view if strides and offset != 0
        checkArgument(values.length() * shape.length == indices.length());
        // TODO - check if the coordinates are correctly sorted
        this.values = Nd4j.createBuffer(values, 0, values.length());
        this.indices = indices;

        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, stride, offset,
                Shape.elementWiseStride(shape, stride, ordering == 'f'), ordering));
    }

    @Override
    public INDArray assign(final INDArray arr) {
        // TODO - set via native op
        return this;
    }


    @Override
    public INDArray putScalar(int i, double value) {
        if (i < 0)
            i += rank();
        if (isScalar()) {
            if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.DISABLED)
                OpProfiler.getInstance().processScalarCall();

            addOrUpdate(getIndicesOf(i).asInt(), value);

            return this;
        }

        if (isRowVector()) {
            addOrUpdate(new int[]{0, i}, value);
            return this;
        } else if (isColumnVector()) {
           addOrUpdate(new int[]{i, 0}, value);
           return this;
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
        return putScalar(new int[]{row, col}, value);
    }

    @Override
    public INDArray putScalar(int dim0, int dim1, int dim2, double value) {
        return putScalar(new int[]{dim0, dim1, dim2}, value);
    }

    @Override
    public INDArray putScalar(int dim0, int dim1, int dim2, int dim3, double value) {
        return putScalar(new int[] {dim0, dim1, dim2, dim3}, value);
    }

    @Override
    public INDArray putRow(int row, INDArray toPut) { // todo move in base ?
        if (isRowVector() && toPut.isVector()) {
            return assign(toPut);
        }
        return put(new INDArrayIndex[] {NDArrayIndex.point(row), NDArrayIndex.all()}, toPut);

    }

    @Override
    public INDArray putColumn(int column, INDArray toPut) {
        return super.putColumn(column, toPut);
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, INDArray element) {
        if (indices[0] instanceof SpecifiedIndex && element.isVector()) {
            indices[0].reset();
            int cnt = 0;
            while (indices[0].hasNext()) {
                int idx = indices[0].next();
                putScalar(idx, element.getDouble(cnt));
                cnt++;
            }
            return this;
        } else {
            return get(indices).assign(element);
        }
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, Number element) {
        if(element.doubleValue() == 0) return this;

        INDArray get = get(indices);
        for (int i = 0; i < get.length(); i++)
            get.putScalar(i, element.doubleValue());
        return this;
    }

    @Override
    public INDArray put(int[] indices, INDArray element) {
        if (!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
        if (isRowVector() && indices[0] == 0 && indices.length == 2) {
            int ix = Shape.offset(shapeInformation);
            for (int i = 1; i < indices.length; i++)
                ix += indices[i] * stride(i);
            if (ix >= values.length())
                throw new IllegalArgumentException("Illegal indices " + Arrays.toString(indices));
            System.out.println(indices[1] + " vs " +ix);
            addOrUpdate(new int[]{0, ix}, element.getDouble(0));
        } else {
            int ix = Shape.offset(shapeInformation);
            for (int i = 0; i < indices.length; i++)
                if (size(i) != 1)
                    ix += indices[i] * stride(i);
            if (ix >= values.length())
                throw new IllegalArgumentException("Illegal indices " + Arrays.toString(indices));
            System.out.println(indices[1] + " vs " +ix);
            addOrUpdate(new int[]{0, ix}, element.getDouble(0));
        }


        return this;
    }

    @Override
    public INDArray put(int i, int j, INDArray element) { // TODO in base ?
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
                if(value == 0){
                    removeEntry(i);
                    length--;
                } else {
                    values.put(i, value);
                }

               break;
            } else {
                if(ArrayUtil.anyMore(indexes, idx) && value != 0){
                    /* It's a new non-null element
                    * The index and value should be added at the position i
                    * Need to shift the tail to make room for the new element
                    * /!\ buffer overflow
                    */
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
            int newSize = (int) (THRESHOLD_MEMORY_ALLOCATION * buffer.length());
            //buffer.reallocate(newSize);
        }
        for(int i = 0; i < tail.length; i++) {
            buffer.put(i + from + offset, tail[i]);
        }
        for(int i = 0; i< offset; i++){
            buffer.put(i+from, 0);
        }
        return buffer;
    }
    public DataBuffer shiftLeft(DataBuffer buffer, int from, int offset, long datalength){
        for(int i = from; i<datalength; i++) {
            buffer.put(i - offset, buffer.getDouble(i));
        }
        return buffer;
    }

    public INDArray removeEntry(int idx){ // TODO move the logic into datatbuffer
        values = shiftLeft(values, idx + 1, 1, length());
        indices = shiftLeft(indices, (int)(idx * shape.length() + shape.length()), (int) shape.length(), length()* shape.length());
        return this;
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
        if (indexes.length == 1 && indexes[0] instanceof NDArrayIndexAll || (indexes.length == 2 && (isRowVector()
                && indexes[0] instanceof PointIndex && indexes[0].offset() == 0
                && indexes[1] instanceof NDArrayIndexAll
                || isColumnVector() && indexes[1] instanceof PointIndex && indexes[0].offset() == 0
                && indexes[0] instanceof NDArrayIndexAll)))
            return this;

        indexes = NDArrayIndex.resolve(shapeInfoDataBuffer(), indexes);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(this);
        resolution.exec(indexes);

        if (indexes.length < 1)
            throw new IllegalStateException("Invalid index found of zero length");

        int[] shape = resolution.getShapes();
        int numSpecifiedIndex = 0;

        for (int i = 0; i < indexes.length; i++)
            if (indexes[i] instanceof SpecifiedIndex)
                numSpecifiedIndex++;


        if (shape != null && numSpecifiedIndex > 0) {
            // TODO create a new ndarray with the specified indexes
            return null;

        }

        INDArray ret = subArray(resolution);
        return ret;



    }

    public SparseFormat getFormat(){
        return format;
    }

    @Override
    public DataBuffer data(){
        return values;
    }

    /**
     * Returns the indices of non-zero element of the vector
     *
     * @return indices in Databuffer
     * */
    @Override
    public DataBuffer getVectorCoordinates() {
        int idx;
        if (isRowVector()) {
            idx = 1;
        } else if (isColumnVector()){
            idx = 0;
        } else {
            throw new UnsupportedOperationException();
        }

        int[] temp = new int[length()];
        for (int i = 0; i < length(); i++) {
            temp[i] = getIndicesOf(i).getInt(idx);
        }
        return Nd4j.createBuffer(temp);
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
        int[] offsets = resolution.getOffsets();
        int[] shape = resolution.getShapes();
        int[] stride = resolution.getStrides();

        if (offset() + resolution.getOffset() >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Offset of array can not be >= Integer.MAX_VALUE");
        int offset = (int) (offset() + resolution.getOffset());

        int n = shape.length;
        if (shape.length < 1)
            //return create(Nd4j.createBuffer(shape)); TODO - how could the shape length be < 1 ?
        if (offsets.length != n)
            throw new IllegalArgumentException("Invalid offset " + Arrays.toString(offsets));
        if (stride.length != n)
            throw new IllegalArgumentException("Invalid stride " + Arrays.toString(stride));

        if (shape.length == rank() && Shape.contentEquals(shape, shapeOf())) {
            if (ArrayUtil.isZero(offsets)) {
                return this;
            } else {
                throw new IllegalArgumentException("Invalid subArray offsets");
            }
        }

        char newOrder = Shape.getOrder(shape, stride, 1);

        return create(values, indices, Arrays.copyOf(shape, shape.length), stride, offset, newOrder);

    }

    private INDArray create(DataBuffer values, DataBuffer indices, int[] shape, int[] stride, int offset, char newOrder) {
        return Nd4j.createSparseCOO(values, indices, stride, offset, shape, newOrder);
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
        int from = rank() * i;
        int to = from + rank();
        int[] arr = Arrays.copyOfRange(indices.asInt(), from, to);
        return Nd4j.getDataBufferFactory().createInt(arr);
    }

    @Override
    public boolean isView() {
        return Shape.offset(shapeInformation) > 0 ||  data().originalDataBuffer() != null;
    }
}
