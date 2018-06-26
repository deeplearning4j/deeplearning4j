package org.nd4j.linalg.api.ndarray;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.google.flatbuffers.FlatBufferBuilder;
import net.ericaro.neoitertools.Generator;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;
import org.nd4j.linalg.profiler.OpProfiler;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LongUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

import static org.nd4j.base.Preconditions.checkArgument;
import static org.nd4j.base.Preconditions.checkNotNull;

/**
 * @author Audrey Loeffel
 */

/*
* TODO :
* - Implement the INDArray methods
* - Sort the databuffers
* - Check at the creation if there are any 0 values and remove them
* - add indexesOrdering in constructor
* - BaseSparseNDArray should extend from BaseNDArray : remove the duplicate methods
* */
public class BaseSparseNDArrayCOO extends BaseSparseNDArray {
    protected static final SparseFormat format = SparseFormat.COO;
    protected transient volatile DataBuffer values;
    protected transient volatile DataBuffer indices;
    protected transient volatile boolean isSorted = false;

    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, long[] shape) {
        checkArgument(values.length() * shape.length == indices.length());

        this.values = values;
        this.indices = indices;
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape));
        init(shape);
        this.length = values.length();
        int[] flags = new int[rank()];
        long[] sparseOffsets = new long[rank()];
        int[] hiddenDimension = new int[] {-1};
        this.sparseInformation = Nd4j.getSparseInfoProvider().createSparseInformation(flags, sparseOffsets,
                hiddenDimension, rank());

    }

    public BaseSparseNDArrayCOO(double[] values, long[][] indices, long[] shape) {
        this.indices = createIndiceBuffer(indices, shape);
        this.values = createValueBuffer(values);
        length = values.length;

        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape));
        init(shape);
        this.sparseInformation = createSparseInformationBuffer(rank());
        checkBufferCoherence();
    }

    public BaseSparseNDArrayCOO(float[] values, long[][] indices, long[] shape) {
        this.indices = createIndiceBuffer(indices, shape);
        this.values = createValueBuffer(values);
        length = values.length;

        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape));
        init(shape);
        this.sparseInformation = createSparseInformationBuffer(rank());
        checkBufferCoherence();
    }

    public BaseSparseNDArrayCOO(double[] values, int[][] indices, long[] shape) {
        this.indices = createIndiceBuffer(indices, shape);
        this.values = createValueBuffer(values);
        length = values.length;

        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape));
        init(shape);
        this.sparseInformation = createSparseInformationBuffer(rank());
        checkBufferCoherence();
    }

    public BaseSparseNDArrayCOO(float[] values, int[][] indices, long[] shape) {
        this.indices = createIndiceBuffer(indices, shape);
        this.values = createValueBuffer(values);
        length = values.length;

        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape));
        init(shape);
        this.sparseInformation = createSparseInformationBuffer(rank());
        checkBufferCoherence();
    }


    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation, long[] shape) {
        this.values = Nd4j.createBuffer(values, 0, values.length());
        this.indices = indices;
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape));
        init(shape);
        this.sparseInformation = sparseInformation;
        this.length = countNNZ();

    }

    public BaseSparseNDArrayCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags,
                                int[] hiddenDimensions, int underlyingRank, long[] shape) {
        this(values, indices, Nd4j.getSparseInfoProvider().createSparseInformation(flags, sparseOffsets,
                        hiddenDimensions, underlyingRank), shape);
    }


    /**
     * Check that the length of indices and values are coherent and matches the rank of the matrix.
     */
    protected void checkBufferCoherence(){
        if (values.length() < length){
            throw new IllegalStateException("nnz is larger than capacity of buffers");
        }

        if (values.length() * rank() != indices.length()){
            throw new IllegalArgumentException("Sizes of values, indices and shape are incoherent.");
        }
    }

    /**
     * Create a SparseInfo databuffer given rank if of the sparse matrix.
     * @param rank
     * @return
     */
    protected static DataBuffer createSparseInformationBuffer(int rank){
        int[] flags = new int[rank];
        long[] sparseOffsets = new long[rank];
        int[] hiddenDimension = new int[] {-1};
        return Nd4j.getSparseInfoProvider().createSparseInformation(flags, sparseOffsets,
                hiddenDimension, rank);
    }


    /**
     * Create a DataBuffer for values of given array of values.
     * @param values
     * @return
     */
    protected static DataBuffer createValueBuffer(float[] values) {
        checkNotNull(values);
        if (values.length == 0){
            return Nd4j.createBuffer(1);
        }
        return Nd4j.createBuffer(values);
    }


    /**
     * Create a DataBuffer for values of given array of values.
     * @param values
     * @return
     */
    protected static DataBuffer createValueBuffer(double[] values) {
        checkNotNull(values);
        if (values.length == 0){
            return Nd4j.createBuffer(1);
        }
        return Nd4j.createBuffer(values);
    }



    /**
     * Create a DataBuffer for indices of given arrays of indices.
     * @param indices
     * @param shape
     * @return
     */
    protected static DataBuffer createIndiceBuffer(long[][] indices, long[] shape){
        checkNotNull(indices);
        checkNotNull(shape);
        if(indices.length == 0){
            return Nd4j.getDataBufferFactory().createLong(shape.length);
        }

        if (indices.length == shape.length) {
            return Nd4j.createBuffer(ArrayUtil.flattenF(indices));
        }

        return Nd4j.createBuffer(ArrayUtil.flatten(indices));
    }

    /**
     * Create a DataBuffer for indices of given arrays of indices.
     * @param indices
     * @param shape
     * @return
     */
    protected static DataBuffer createIndiceBuffer(int[][] indices, long[] shape){
        checkNotNull(indices);
        checkNotNull(shape);
        if(indices.length == 0){
            return Nd4j.getDataBufferFactory().createLong(shape.length);
        }

        if (indices.length == shape.length) {
            return Nd4j.createBuffer(ArrayUtil.toLongArray(ArrayUtil.flattenF(indices)));
        }

        return Nd4j.createBuffer(ArrayUtil.toLongArray(ArrayUtil.flatten(indices)));
    }

    @Override
    public int toFlatArray(FlatBufferBuilder builder) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray convertToHalfs() {
        return null;
    }


    /**
     * Count the number of value that are included in the ndarray (view) according to the sparse offsets and the shape
     * @return nnz
     * */
    public long countNNZ() {
        long count = 0;

        for (int i = 0; i < values.length(); i++) {
            int[] idx = getUnderlyingIndicesOf(i).asInt();
            boolean isIn = true;
            int idxNotFixed = 0;
            for (int dim = 0; dim < idx.length; dim++) {

                if (flags()[dim] == 1) {
                    if (sparseOffsets()[dim] != idx[dim]) {
                        isIn = false;
                        break;
                    }
                } else {
                    int lowerBound = sparseOffsets()[dim];
                    long upperBound = sparseOffsets()[dim] + shape()[idxNotFixed];
                    if (!(idx[dim] >= lowerBound && idx[dim] < upperBound)) {
                        isIn = false;
                        break;
                    }
                    idxNotFixed++;
                }
            }
            count = isIn ? count + 1 : count;
        }
        return count;
    }

    @Override
    public INDArray assign(final INDArray arr) {
        sort();
        // TODO - set via native op
        return this;
    }

    /**
     * Sort the indexes and the values buffers
     * */
    public void sort() {
        if (!isSorted) {
            Nd4j.sparseFactory().sortCooIndices(this);
            isSorted = true;
        }
    }

    /**
     * Translate the view index to the corresponding index of the original ndarray
     * @param virtualIndexes the view indexes
     * @return the original indexes
     * */
    public long[] translateToPhysical(long[] virtualIndexes) {

        long[] physicalIndexes = new long[underlyingRank()];
        int idxPhy = 0;
        int hidden = 0;

        for (int idxVir = 0; idxVir < virtualIndexes.length; idxVir++) {
            if (hidden < getNumHiddenDimension() && hiddenDimensions()[hidden] == idxVir) {
                hidden++;
            } else {
                while (idxPhy < underlyingRank() && isDimensionFixed(idxPhy)) {
                    physicalIndexes[idxPhy] = sparseOffsets()[idxPhy];
                    idxPhy++;
                }
                if (idxPhy < underlyingRank() && !isDimensionFixed(idxPhy)) {
                    physicalIndexes[idxPhy] = sparseOffsets()[idxPhy] + virtualIndexes[idxVir];
                    idxPhy++;
                }
            }
        }
        return physicalIndexes;
    }

    public int[] translateToPhysical(int[] virtualIndexes) {

        int[] physicalIndexes = new int[underlyingRank()];
        int idxPhy = 0;
        int hidden = 0;

        for (int idxVir = 0; idxVir < virtualIndexes.length; idxVir++) {
            if (hidden < getNumHiddenDimension() && hiddenDimensions()[hidden] == idxVir) {
                hidden++;
            } else {
                while (idxPhy < underlyingRank() && isDimensionFixed(idxPhy)) {
                    physicalIndexes[idxPhy] = sparseOffsets()[idxPhy];
                    idxPhy++;
                }
                if (idxPhy < underlyingRank() && !isDimensionFixed(idxPhy)) {
                    physicalIndexes[idxPhy] = sparseOffsets()[idxPhy] + virtualIndexes[idxVir];
                    idxPhy++;
                }
            }
        }
        return physicalIndexes;
    }

    /**
     * Return if the dimension in argument is a fixed dimension.
     * */
    public boolean isDimensionFixed(int i) {
        return flags()[i] == 1;
    }


    @Override
    public INDArray putScalar(long i, double value) {
        if (i < 0)
            i += rank();
        if (isScalar()) {
            if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.DISABLED && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.SCOPE_PANIC)
                OpProfiler.getInstance().processScalarCall();

            addOrUpdate(new long[] {0, 0}, value);
            return this;
        }
        if (isRowVector()) {
            addOrUpdate(new long[] {0, i}, value);
            return this;
        } else if (isColumnVector()) {
            addOrUpdate(new long[] {i, 0}, value);
            return this;
        }
        long[] indexes = ordering() == 'c' ? Shape.ind2subC(this, i) : Shape.ind2sub(this, i);
        return putScalar(indexes, value);
    }

    @Override
    public INDArray putScalar(long i, float value) {
        return putScalar(i, (double) value); //todo - move in baseSparse?
    }

    @Override
    public INDArray putScalar(long i, int value) {
        return putScalar(i, (double) value); //todo
    }

    @Override
    public INDArray putScalar(int[] indexes, double value) {
        return putScalar(ArrayUtil.toLongArray(indexes), value);
    }

    @Override
    public INDArray putScalar(long[] indexes, double value) {
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < 0)
                indexes[i] += rank();
        }

        if (indexes.length == 1) {
            return putScalar(indexes[0], value);
        }
        if (indexes.length != rank) {
            throw new IllegalStateException(
                    "Cannot use putScalar with indexes length " + indexes.length + " on rank " + rank);
        }
        addOrUpdate(indexes, value);
        return this;
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
    public INDArray putScalar(long row, long col, double value) {
        return putScalar(new long[] {row, col}, value);
    }

    @Override
    public INDArray putScalar(long dim0, long dim1, long dim2, double value) {
        return putScalar(new long[] {dim0, dim1, dim2}, value);
    }

    @Override
    public INDArray putScalar(long dim0, long dim1, long dim2, long dim3, double value) {
        return putScalar(new long[] {dim0, dim1, dim2, dim3}, value);
    }

    @Override
    public INDArray putRow(long row, INDArray toPut) {
        if (isRowVector() && toPut.isVector()) {
            return assign(toPut);
        }
        return put(new INDArrayIndex[] {NDArrayIndex.point(row), NDArrayIndex.all()}, toPut);
    }

    @Override
    public INDArray putColumn(int column, INDArray toPut) {
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
                long idx = indices[0].next();
                putScalar((int) idx, element.getDouble(cnt));
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
        if (indexes.length != rank)
            throw new IllegalStateException(
                            "Cannot use putScalar with indexes length " + indexes.length + " on rank " + rank);

        addOrUpdate(ArrayUtil.toLongArray(indexes), element.getDouble(0));
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
    public INDArray put(int i, INDArray element) { // TODO remove and use basendarray method
        if (!element.isScalar())
            throw new IllegalArgumentException("Element must be a scalar");
        return putScalar(i, element.getDouble(0));
    }

    /**
     * Add a new element in the ndarray or update the value if there is already a non-null element at this position
     * @param indexes the indexes of the element to be added
     * @param value the value of the element to be added
     * */
    public void addOrUpdate(long[] indexes, double value) {

        long[] physicalIndexes = isView() ? translateToPhysical(indexes) : indexes;

        for (int i = 0; i < length; i++) {
            long[] idx = getUnderlyingIndicesOf(i).asLong();
            if (Arrays.equals(idx, physicalIndexes)) {
                // There is already a non-null value at this index
                // -> update the current value, the sort is maintained
                if (value == 0) {
                    removeEntry(i);
                    length--;
                } else {
                    values.put(i, value);
                    length++;
                }
                return;
            }
        }

        // If the value is 0 and there is no existing non-null value at the given index
        if (value == 0) {
            return;
        }

        /* It's a new non-null element. We add the value and the indexes at the end of their respective databuffers.
        * The buffers are no longer sorted !
        * /!\ We need to reallocate the buffers if they are full
        */
        while (!canInsert(values, 1)) {
            long size = (long) Math.ceil((values.capacity() * THRESHOLD_MEMORY_ALLOCATION));
            values.reallocate(size);
        }
        values.put(length, value);
        while (!canInsert(indices, physicalIndexes.length)) {
            long size = (long) Math.ceil((indices.capacity() * THRESHOLD_MEMORY_ALLOCATION));
            indices.reallocate(size);
        }
        for (int i = 0; i < physicalIndexes.length; i++) {
            indices.put(length * rank() + i, physicalIndexes[i]);
        }
        length++;
        isSorted = false;
    }

    /**
     * Return if there is enough allocated memory space to add data of a given length in the databuffer
     * @param buffer a databuffer in which we want to add data
     * @param length the length of the data
     * @return a boolean if the insertion is possible
     * */
    public boolean canInsert(DataBuffer buffer, int length) {
        return buffer.capacity() - buffer.length() >= length;
    }

    public DataBuffer shiftLeft(DataBuffer buffer, int from, int offset, long datalength) {
        for (int i = from; i < datalength; i++) {
            buffer.put(i - offset, buffer.getDouble(i));
        }
        return buffer;
    }

    /**
     * Remove an element of the ndarray
     * @param idx the index of the element to be removed
     * @return the ndarray
     * */
    public INDArray removeEntry(int idx) {
        values = shiftLeft(values, idx + 1, 1, length());
        indices = shiftLeft(indices, (int) (idx * shape.length() + shape.length()), (int) shape.length(),
                        indices.length());
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

        sort();

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

        // FIXME: LONG
        long[] shape = resolution.getShapes();
        int numSpecifiedIndex = 0;

        for (int i = 0; i < indexes.length; i++)
            if (indexes[i] instanceof SpecifiedIndex)
                numSpecifiedIndex++;

        if (shape != null && numSpecifiedIndex > 0) {
            Generator<List<List<Long>>> gen = SpecifiedIndex.iterateOverSparse(indexes);
            INDArray ret = Nd4j.createSparseCOO(new double[] {}, new int[][] {}, shape);
            int count = 0;
            int maxValue = ArrayUtil.prod(shape());
            while (count < maxValue) {
                try {
                    List<List<Long>> next = gen.next();
                    List<Integer> coordsCombo = new ArrayList<>();
                    List<Integer> cooIdx = new ArrayList<>();
                    for (int i = 0; i < next.size(); i++) {
                        if (next.get(i).size() != 2)
                            throw new IllegalStateException("Illegal entry returned");
                        coordsCombo.add(next.get(i).get(0).intValue());
                        cooIdx.add(next.get(i).get(1).intValue());
                    }
                    count++;

                    /*
                    * if the coordinates are in the original array
                    *   -> add it in the new sparse ndarray
                    * else
                    *   -> do nothing
                    * */
                    int[] idx = Ints.toArray(coordsCombo);
                    if (!isZero(idx)) {
                        double val = getDouble(idx);
                        ret.putScalar(filterOutFixedDimensions(resolution.getFixed(), cooIdx), val);
                    }

                } catch (NoSuchElementException e) {
                    break;
                }
            }

            return ret;
        }

        int numNewAxis = 0;
        for (int i = 0; i < indexes.length; i++)
            if (indexes[i] instanceof NewAxis)
                numNewAxis++;
        if (numNewAxis != 0) {

        }

        INDArray ret = subArray(resolution);
        return ret;
    }

    @Override
    public INDArray repeat(int dimension, long... repeats) {
        return null;
    }


    public int[] filterOutFixedDimensions(int[] flags, List<Integer> idx) {
        checkArgument(flags.length == idx.size());
        int lastIdx = idx.size() - 1;
        for (int i = lastIdx; i >= 0; i--) {
            if (flags[i] == 1) {
                idx.remove(i);
            }
        }
        return Ints.toArray(idx);
    }

    /**
     * Return the index of the value corresponding to the indexes
     * @param indexes
     * @return index of the value
     * */
    public int reverseIndexes(int... indexes) {
        long[] idx = translateToPhysical(ArrayUtil.toLongArray(indexes));
        sort();

        // FIXME: int cast
        return indexesBinarySearch(0, (int) length(), ArrayUtil.toInts(idx));
    }

    /**
     * Return the position of the idx array into the indexes buffer between the lower and upper bound.
     * @param idx a set of coordinates
     * @param lowerBound the lower bound of the position
     * @param upperBound the upper bound of the position
     * @return the position of the idx array into the indexes buffers, which corresponds to the position of
     * the corresponding value in the values data.
     * */
    public int indexesBinarySearch(int lowerBound, int upperBound, int[] idx) {
        int min = lowerBound;
        int max = upperBound;

        int mid = (max + min) / 2;
        int[] midIdx = getUnderlyingIndicesOf(mid).asInt();
        if (Arrays.equals(idx, midIdx)) {
            return mid;
        }
        if (ArrayUtil.lessThan(idx, midIdx)) {
            max = mid;
        }
        if (ArrayUtil.greaterThan(idx, midIdx)) {
            min = mid;
        }
        if (min == max) {
            return -1;
        }
        return indexesBinarySearch(min, max, idx);
    }

    @Override
    public INDArray getScalar(int... indices) {
        return super.getScalar(indices);
    }

    @Override
    public INDArray getScalar(long... indices) {
        return null;
    }

    @Override
    public int getInt(int... indices) {
        return super.getInt(indices);
    }

    @Override
    public double getDouble(int... indices) {
        int valIdx = reverseIndexes(indices);
        if (valIdx == -1) {
            return 0;
        } else {
            return values.getDouble(valIdx);
        }
    }

    @Override
    public double getDouble(long... indices) {
        return 0;
    }

    @Override
    public float getFloat(int[] indices) {
        return (float) getDouble(indices);
    }

    @Override
    public float getFloat(long[] indices) {
        return 0;
    }

    @Override
    public double getDouble(long i) {
        if (i >= length()) {
            throw new IllegalArgumentException("Unable to get linear index >= " + length());
        }

        if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.DISABLED && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.SCOPE_PANIC)
            OpProfiler.getInstance().processScalarCall();

        if (i == 0)
            return data().getDouble(i);

        long[] dimensions = ordering() == 'c' ? Shape.ind2subC(this, i) : Shape.ind2sub(this, i);
        Shape.assertShapeLessThan(dimensions, shape());
        return getDouble(dimensions);
    }

    @Override
    public double getDouble(long i, long j) {
        return getDouble(new long[] {i, j});
    }

    @Override
    public float getFloat(long i) {
        return (float) getDouble(i);
    }

    @Override
    public float getFloat(long i, long j) {
        return (float) getDouble(i, j);
    }

    @Override
    public INDArray reshape(char order, int... newShape) {
        return null;
    }

    @Override
    public INDArray reshape(int[] shape) {
        return null;
    }

    public SparseFormat getFormat() {
        return format;
    }

    @Override
    public int underlyingRank() {
        return Shape.underlyingRank(sparseInformation);
    }

    @Override
    public DataBuffer data() {
        return values;
    }

    /**
     * Return the indices buffer
     * @return indices
     * */
    public DataBuffer getUnderlyingIndices() {
        return indices;
    }

    /**
    * Return a copy of the indices included in the view.
    * /!\ Change this DataBuffer won't change the ndarray!
    * @return an array containing the virtual indexes of the values (think about views).
    * */
    public DataBuffer getIncludedIndices() {

        if (isScalar()) {
            return Nd4j.createBuffer(new int[] {0, 0});
        }

        List<Integer> ind = new ArrayList<>();

        for (int i = 0; i < values.length(); i++) {
            boolean isIn = true;
            int idxNotFixed = 0;
            int[] idx = getUnderlyingIndicesOf(i).asInt(); // TODO change for getIndicesOf(i)

            for (int dim = 0; dim < idx.length; dim++) {
                if (flags()[dim] == 1) {
                    if (sparseOffsets()[dim] != idx[dim]) {
                        isIn = false;
                        break;
                    }
                } else {
                    int lowerBound = sparseOffsets()[dim];
                    long upperBound = sparseOffsets()[dim] + shape()[idxNotFixed];
                    if (!(idx[dim] >= lowerBound && idx[dim] < upperBound)) {
                        isIn = false;
                        break;
                    }
                    idxNotFixed++;
                }
            }
            if (isIn) {
                int notFixedDim = 0;
                for (int dim = 0; dim < idx.length; dim++) {
                    if (flags()[dim] == 0) {
                        if (shape()[notFixedDim] == 1) {
                            ind.add(0);
                            notFixedDim++;
                        } else {
                            ind.add(idx[dim] - sparseOffsets()[dim]);
                        }
                    }
                }
            }
        }
        return Nd4j.createBuffer(Ints.toArray(ind));
    }

    /**
     * Return the values buffer
     * @return values
     * */
    public DataBuffer getUnderlyingValues() {
        return values;
    }

    /**
     * Return a copy of the values included in the array.
     * /!\ Change this DataBuffer won't change the ndarray!
     * @return an array containing the values
     * */
    public DataBuffer getIncludedValues() {
        List<Double> val = new ArrayList<>();

        for (int i = 0; i < values.length(); i++) {
            boolean isIn = true;
            int idxNotFixed = 0;
            int[] idx = getUnderlyingIndicesOf(i).asInt();
            for (int dim = 0; dim < idx.length; dim++) {
                if (flags()[dim] == 1) {
                    if (sparseOffsets()[dim] != idx[dim]) {
                        isIn = false;
                        break;
                    }
                } else {
                    int lowerBound = sparseOffsets()[dim];
                    long upperBound = sparseOffsets()[dim] + shape()[idxNotFixed];
                    if (!(idx[dim] >= lowerBound && idx[dim] < upperBound)) {
                        isIn = false;
                        break;
                    }
                    idxNotFixed++;
                }
            }
            if (isIn) {
                val.add(values.getDouble(i));
            }
        }
        return Nd4j.createBuffer(Doubles.toArray(val));
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
        } else if (isColumnVector()) {
            idx = 0;
        } else {
            throw new UnsupportedOperationException();
        }

        // FIXME: int cast
        int[] temp = new int[(int) length()];
        for (int i = 0; i < length(); i++) {
            temp[i] = getUnderlyingIndicesOf(i).getInt(idx);
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

        switch (data().dataType()) {
            case DOUBLE:
                for (int i = 0; i < length; i++) {
                    int[] idx = getUnderlyingIndicesOf(i).asInt();
                    double value = values.getDouble(i);
                    result.putScalar(idx, value);
                }
                break;
            case FLOAT:
                for (int i = 0; i < length; i++) {
                    int[] idx = getUnderlyingIndicesOf(i).asInt();
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
        long[] offsets = resolution.getOffsets();
        long[] shape = resolution.getShapes();
        int[] stride = LongUtils.toInts(resolution.getStrides());
        int[] flags = resolution.getFixed();
        flags = updateFlags(flags, shape);
        long offset = (int) (offset() + resolution.getOffset());
        int newRank = shape.length;
        long[] sparseOffsets = createSparseOffsets(offset);
        int[] newAxis = createHiddenDimensions(resolution.getPrependAxis());


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
        DataBuffer newSparseInformation = Nd4j.getSparseInfoProvider().createSparseInformation(flags, sparseOffsets,
                        newAxis, underlyingRank());
        return create(values, indices, newSparseInformation, Arrays.copyOf(shape, shape.length));

    }

    //@Override
    public INDArray subArray(ShapeOffsetResolution resolution, ShapeOffsetResolution resolutionWithoutNewAxis) {
        return null;
    }

    /**
     * Compute the sparse offsets of the view we are getting, for each dimension according to the original ndarray
     * @param offset the offset of the view
     * @return an int array containing the sparse offsets
     * */
    private long[] createSparseOffsets(long offset) {

        // resolve the offsets in the view dimension
        int underlyingRank = sparseOffsets().length;
        long[] newOffsets = new long[rank()];
        List<Long> shapeList = Longs.asList(shape());
        int penultimate = rank() - 1;
        for (int i = 0; i < penultimate; i++) {
            long prod = ArrayUtil.prodLong(shapeList.subList(i + 1, rank()));
            newOffsets[i] = offset / prod;
            offset = offset - newOffsets[i] * prod;
        }
        newOffsets[rank() - 1] = offset % shape()[rank() - 1];

        // Merge the offsets with the original sparseOffsets
        long[] finalOffsets = new long[underlyingRank];
        int dimNotFixed = 0;
        for (int dim = 0; dim < underlyingRank; dim++) {
            if (flags()[dim] == 1) {
                finalOffsets[dim] = sparseOffsets()[dim];
            } else {
                finalOffsets[dim] = newOffsets[dimNotFixed] + sparseOffsets()[dim];
                dimNotFixed++;
            }
        }
        return finalOffsets;
    }

    private int[] createHiddenDimensions(int[] newAxis) {
        if (newAxis == null || newAxis.length == 0) {
            return hiddenDimensions();
        }
        if (getNumHiddenDimension() == 0) {
            return newAxis;
        }
        int size = newAxis.length + hiddenDimensions().length;
        int[] newHiddenDim = new int[size];
        int newDim = 0;
        int actualArrayIdx = 0;
        for (int oldDim = 0; oldDim < getNumHiddenDimension(); oldDim++) {
            while ((newDim < newAxis.length) && (oldDim >=getNumHiddenDimension() || newAxis[newDim] <= hiddenDimensions()[oldDim])) {
                newHiddenDim[actualArrayIdx] = newAxis[newDim] + oldDim;
                actualArrayIdx++;
                newDim++;
            }
            newHiddenDim[actualArrayIdx] = hiddenDimensions()[oldDim] + newDim;
            actualArrayIdx++;
        }
        return newHiddenDim;
    }

    /**
    * Adjust the flags array according on the current context:
    * - In case of a vector or a scalar, we need to keep flags to 0
    * - There must always be at least 2 non-flags dimensions
    * - We must keep the flags dimensions of the original array
    *
    * @param viewFlags the Fixed array calculate within the view
    * @param newShape the shape of the view
    * */
    private int[] updateFlags(int[] viewFlags, long[] newShape) {
        // Check if flags is well-formed
        int count = 0;
        for (int i = 0; i < viewFlags.length; i++) {
            if (viewFlags[i] == 0) {
                count++;
            }
        }

        for (int dim = 0; dim < viewFlags.length; dim++) {
            if (viewFlags[dim] == 1) {
                if (newShape[dim] == 1 && count < 2) {
                    viewFlags[dim] = 0;
                }

            }
        }
        // Take the original Fixed into account
        int[] extendedFlags = new int[underlyingRank()];
        int notFixedDim = 0;
        for (int dim = 0; dim < underlyingRank(); dim++) {
            int[] temp = flags();
            if (flags()[dim] == 0) {
                extendedFlags[dim] = viewFlags[notFixedDim];
                notFixedDim++;
            } else {
                extendedFlags[dim] = 1;
            }
        }
        return extendedFlags;
    }

    private INDArray create(DataBuffer values, DataBuffer indices, DataBuffer sparseInfo, long[] newShape) {
        return Nd4j.createSparseCOO(values, indices, sparseInfo, newShape);
    }

    @Override
    public INDArray subArray(long[] offsets, int[] shape, int[] stride) {
        throw new UnsupportedOperationException();
    }



    /**
     * Returns the underlying indices of the element of the given index
     * such as there really are in the original ndarray
     *
     * @param i the index of the element+
     * @return a dataBuffer containing the indices of element
     * */
    public DataBuffer getUnderlyingIndicesOf(int i) {
        int from = underlyingRank() * i;
        //int to = from + underlyingRank();
        int[] res = new int[underlyingRank()];
        for(int j = 0; j< underlyingRank(); j++){
            res[j] = indices.getInt(from + j);
        }

        ///int[] arr = Arrays.copyOfRange(indices.asInt(), from, to);
        return Nd4j.getDataBufferFactory().createInt(res);
    }

    /**
     * Returns the indices of the element of the given index in the array context
     *
     * @param i the index of the element
     * @return a dataBuffer containing the indices of element
     * */
    public DataBuffer getIndicesOf(int i) {
        int from = underlyingRank() * i;
        int to = from + underlyingRank(); //not included

        int[] arr = new int[rank];
        int j = 0; // iterator over underlying indices
        int k = 0; //iterator over hiddenIdx
        for (int dim = 0; dim < rank; dim++) {
            if (k < hiddenDimensions().length && hiddenDimensions()[k] == j) {
                arr[dim] = 0;
                k++;
            } else {
                arr[dim] = indices.getInt(j);
                j++;
            }
        }
        return Nd4j.getDataBufferFactory().createInt(arr);
    }


    public boolean isZero(int... indexes) {
        for (int i = 0; i < length(); i++) {
            int[] idx = getUnderlyingIndicesOf(i).asInt();
            if (Arrays.equals(idx, translateToPhysical(indexes))) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean isView() {
        return Shape.offset(shapeInformation) > 0 || data().originalDataBuffer() != null; // TODO or if sparseOffset/flags != [0, ..,0]
    }


    public int getNumHiddenDimension() {
        if (hiddenDimensions() == null || hiddenDimensions().length == 0) {
            throw new IllegalStateException("HiddenDimension array is malformed");
        }
        return hiddenDimensions()[0] == -1 ? 0 : hiddenDimensions().length;
    }

    public boolean isSorted() {
        return isSorted;
    }

    public DataBuffer getValues() {
        return values;
    }

    public DataBuffer getIndices() {
        return indices;
    }

    @Override
    public INDArray putiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray putiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray mmul(INDArray other, MMulTranspose mMulTranspose) {
        return null;
    }

    /**
     * Perform an copy matrix multiplication
     *
     * @param other         the other matrix to perform matrix multiply with
     * @param result        the result ndarray
     * @param mMulTranspose the transpose status of each array
     * @return the result of the matrix multiplication
     */
    @Override
    public INDArray mmul(INDArray other, INDArray result, MMulTranspose mMulTranspose) {
        return null;
    }

    @Override
    public INDArray mmuli(INDArray other, MMulTranspose transpose) {
        return null;
    }

    @Override
    public INDArray mmuli(INDArray other, INDArray result, MMulTranspose transpose) {
        return null;
    }

    @Override
    public void setStride(long... stride) {

    }

    @Override
    public void setShape(long... shape) {

    }

    @Override
    public INDArray convertToFloats() {
        return null;
    }

    @Override
    public INDArray convertToDoubles() {
        return null;
    }
}
