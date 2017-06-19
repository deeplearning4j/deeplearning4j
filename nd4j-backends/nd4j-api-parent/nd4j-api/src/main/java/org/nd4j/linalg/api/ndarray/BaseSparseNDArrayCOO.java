package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;
import org.nd4j.linalg.profiler.OpProfiler;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import com.google.common.primitives.Ints;

/**
 * @author Audrey Loeffel
 */
public class BaseSparseNDArrayCOO extends BaseSparseNDArray {
    protected static final SparseFormat format = SparseFormat.COO;
    protected transient volatile DataBuffer values;
    protected transient volatile DataBuffer indices;
    protected transient volatile DataBuffer fixed;
    protected transient volatile DataBuffer  sparseOffsets;
    protected transient volatile boolean isSorted = false;

    public BaseSparseNDArrayCOO(double[] values, int[][] indices, int[] shape){

        checkNotNull(values);
        checkNotNull(indices);
        checkNotNull(shape);
        for(int[] i : indices){
            checkNotNull(i);
        }

        if(indices.length == shape.length && indices[0].length == values.length){
            this.indices = Nd4j.createBuffer(ArrayUtil.flatten(indices));
        } else if(indices.length == values.length && indices[0].length == shape.length){
            this.indices = Nd4j.createBuffer(ArrayUtil.flattenF(indices));
        } else {
            throw new IllegalArgumentException("Sizes of values, indices and shape are incoherent.");
        }
        this.values = Nd4j.createBuffer(values);
        this.indices = Nd4j.createBuffer(ArrayUtil.flatten(indices));
        System.out.println(indices.toString());
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);

        init(shape);
        this.length = values.length;
        this.fixed = Nd4j.createBuffer(new int[shape.length]);
        this.sparseOffsets = Nd4j.createBuffer(new int[shape.length]);
    }

    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, int[] shape){
        checkArgument(values.length() * shape.length == indices.length());

        this.values =  Nd4j.createBuffer(values, 0, values.length());
        this.indices = indices;
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);
        init(shape);
        this.length = values.length();
        this.fixed = Nd4j.createBuffer(new int[shape.length]);
        this.sparseOffsets = Nd4j.createBuffer(new int[shape.length]);
    }
    public BaseSparseNDArrayCOO(float[] values, int[][] indices, int[] shape){
        checkArgument(values.length == indices.length);
        checkArgument(values.length == 0 || indices[0].length == shape.length);

        this.values = Nd4j.createBuffer(values);
        this.indices = Nd4j.createBuffer(ArrayUtil.flatten(indices));
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);
        init(shape);
        this.length = values.length;
        this.fixed = Nd4j.createBuffer(new int[shape.length]);
        this.sparseOffsets = Nd4j.createBuffer(new int[shape.length]);
    }

    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, int[] sparseOffsets, int[] fixed, int[] shape, char ordering){

        checkArgument(values.length() * shape.length == indices.length());
        this.values = Nd4j.createBuffer(values, 0, values.length());
        this.indices = indices;
        this.sparseOffsets = Nd4j.createBuffer(sparseOffsets);
        this.fixed =  Nd4j.createBuffer(fixed);
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape, ordering));
        init(shape);
        this.length = countNNZ();
    }

    /**
     * Count the number of value that are included in the ndarray (view) according to the sparse offsets and the shape
     * @return nnz
     * */
    public long countNNZ(){
        long count = 0;

        for(int i = 0; i< values.length(); i++){
            int[] idx = getIndicesOf(i).asInt();
            boolean isIn = true;
            for(int j = 0; j < idx.length; j++){
                if(!(idx[j] >= sparseOffsets.getInt(i) || idx[i] < sparseOffsets.getInt(i) + shape.getInt(i))) {
                    isIn = false;
                }
            }
            count = isIn ? count + 1 : count;
        }
        return count;
    }

    @Override
    public INDArray assign(final INDArray arr) {
        if(!isSorted){
            // sort -> this should be done before every op or get
            // isSorted = true;
        }
        // TODO - set via native op
        return this;
    }

    /**
     * Translate the view index to the corresponding index of the original ndarray
     * @param virtualIndexes the view indexes
     * @return the original indexes
     * */
    public int[] translateToPhysical(int[] virtualIndexes) {
        int underlyingRank = (int) fixed.length();
        int[] physicalIndexes = new int[underlyingRank];
        int currentIdx = 0;
        /*
        If the dimension varies, we take the index from indexes,
        but if the dimension is fixed, then we take its fixes value from the offsets
        */
        for(int i =  0; i < underlyingRank; i++){
            if(!isDimensionFixed(i)){
                physicalIndexes[i] = virtualIndexes[currentIdx] + sparseOffsets.getInt(i);
                currentIdx++;
            } else {
                physicalIndexes[i] = sparseOffsets.getInt(i);
            }
        }
        return physicalIndexes;

    }
    /**
     * Return if a given dimension is fixed in the view.
     * */
    public boolean isDimensionFixed(int i){
        return fixed.getInt(i) == 1;
    }

    @Override
    public INDArray putScalar(int i, double value) {
        if (i < 0)
            i += rank();
        if (isScalar()) {
            if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.DISABLED)
                OpProfiler.getInstance().processScalarCall();

            addOrUpdate(new int[]{i}, value);
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

        if(indexes.length == 1){
            return putScalar(indexes[0], value);
        }
        if(indexes.length != rank) {
            throw new IllegalStateException("Cannot use putScalar with indexes length "
                    + indexes.length + " on rank " + rank);
        }

        addOrUpdate(indexes, value);
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
    public INDArray putColumn(int column, INDArray toPut) {  // todo move in base ?
        if (isColumnVector() && toPut.isVector()) {
            return assign(toPut);
        }
        return put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.point(column)}, toPut);
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
        INDArray get = get(indices);
        for (int i = 0; i < get.length(); i++)
            get.putScalar(i, element.doubleValue());
        return this;
    }

    @Override
    public INDArray put(int[] indexes, INDArray element) {

        if (!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
        if (indexes.length  != rank)
            throw new IllegalStateException("Cannot use putScalar with indexes length "
                    + indexes.length + " on rank " + rank);

        addOrUpdate(indexes, element.getDouble(0));
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
        // TODO - can an original array have offsets ??
        int[] physicalIndexes = isView() ? translateToPhysical(indexes) : indexes;

        for(int i = 0; i < length; i++) {
            int[] idx = getIndicesOf(i).asInt();
            if (Arrays.equals(idx, physicalIndexes)) {
                // There is already a non-null value at this index
                // -> update the current value, the sort is maintained
                if (value == 0) {
                    removeEntry(i);
                    length--;
                } else {
                    values.put(i, value);
                }
                return;
            }
        }

        /* It's a new non-null element. We add the value and the indexes at the end of their respective databuffers.
        * The buffers are no longer sorted !
        * /!\ We need to reallocate the buffers if they are full
        */
        while(!canInsert(values, 1)){
            long size = (long)(values.capacity() * THRESHOLD_MEMORY_ALLOCATION);
            values.reallocate(size);
        }
        values.put(values.length(), value);
        while(!canInsert(indices, physicalIndexes.length)){
            long size = (long)(indices.capacity() * THRESHOLD_MEMORY_ALLOCATION);
            indices.reallocate(size);
        }
        for(int i = 0; i< physicalIndexes.length; i++){
            indices.put(indices.length() + i, physicalIndexes[i]);
        }
    }

    public boolean canInsert(DataBuffer buffer, int length){
        return buffer.capacity() - buffer.length() >= length;
    }

    public DataBuffer shiftLeft(DataBuffer buffer, int from, int offset, long datalength){
        for(int i = from; i<datalength; i++) {
            buffer.put(i - offset, buffer.getDouble(i));
        }
        return buffer;
    }

    public INDArray removeEntry(int idx){
        values = shiftLeft(values, idx + 1, 1, length());
        indices = shiftLeft(indices, (int)(idx * shape.length() + shape.length()), (int) shape.length(), indices.length());
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

        if(!isSorted){
            // TODO - Sort !
        }

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

    public DataBuffer getIndices(){
        return indices;
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
                for(int i = 0; i < length; i++) {
                    int[] idx = getIndicesOf(i).asInt();
                    double value = values.getDouble(i);
                    result.putScalar(idx, value);
                }
                break;
            case FLOAT:
                for(int i = 0; i < length; i++) {
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
        int[] fixed = resolution.getFixed();
        int offset = (int) (offset() + resolution.getOffset());
        int newRank = shape.length;
        int[] sparseOffsets = createSparseOffsets(offset);


        if (offset() + resolution.getOffset() >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Offset of array can not be >= Integer.MAX_VALUE");

        if (offsets.length != newRank)
            throw new IllegalArgumentException("Invalid offset " + Arrays.toString(offsets));
        if (stride.length != newRank)
            throw new IllegalArgumentException("Invalid stride " + Arrays.toString(stride));

        if (shape.length == rank() && Shape.contentEquals(shape, shapeOf())) {
            if (ArrayUtil.isZero(offsets)) {
                return this;
            } else {
                throw new IllegalArgumentException("Invalid subArray offsets");
            }
        }

        return create(values, indices, Arrays.copyOf(shape, shape.length), sparseOffsets, fixed, ordering());

    }

    /**
     * Compute the sparse offsets of the view we are getting, for each dimension according to the original ndarray
     * @param offset the offset of the view
     * @return an int array containing the sparse offsets
     * */
    private int[] createSparseOffsets(int offset){
        int[] underlyingShape = shape();
        int underlyingRank = (int) sparseOffsets.length();
        int[] newOffsets = new int[underlyingRank];
        List<Integer> shapeList = Ints.asList(underlyingShape);
        int penultimate = underlyingRank -1;
        for(int i = 0; i < penultimate; i++){
            int prod = ArrayUtil.prod(shapeList.subList(i, penultimate));
            newOffsets[i] = offset / prod;
            offset = offset - newOffsets[i] * prod;
        }
        newOffsets[underlyingRank-1] =  offset % underlyingShape[underlyingRank-1];
        return newOffsets;
    }


    private INDArray create(DataBuffer values, DataBuffer indices, int[] shape, int[] sparseOffsets, int[] fixed, char newOrder) {
        return Nd4j.createSparseCOO(values, indices, sparseOffsets, fixed, shape, newOrder);
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
        return Shape.offset(shapeInformation) > 0 ||  data().originalDataBuffer() != null; // TODO - if fixed.length != rank ?
    }
}
