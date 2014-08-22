package org.deeplearning4j.linalg.jcublas.complex;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.DimensionSlice;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.api.ndarray.SliceOp;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.ops.reduceops.Ops;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.ComplexIterationResult;
import org.deeplearning4j.linalg.util.LinAlgExceptions;
import org.deeplearning4j.linalg.util.Shape;
import org.jblas.ComplexDouble;
import org.jblas.ComplexDoubleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.deeplearning4j.linalg.util.ArrayUtil.calcStrides;

/**
 * Created by mjk on 8/20/14.
 */
public class JCublasComplexNDArray implements IComplexNDArray {
    private int[] shape;
    private int[] stride;
    private int offset = 0;
    public int rows;
    public int columns;
    public int length;
    public double[] data = null; // rows are contiguous

    public JCublasComplexNDArray(INDArray arr) {
    }

    public JCublasComplexNDArray(IComplexNumber[] data, int[] shape) {
    }

    public JCublasComplexNDArray(List<IComplexNDArray> arrs, int[] shape) {
    }

    public JCublasComplexNDArray(float[] data, int[] shape, int[] stride, int offset) {

    }

    @Override
    public IComplexNDArray cumsumi(int dimension) {

        if(isVector()) {
            IComplexNumber s = NDArrays.createDouble(0, 0);
            for (int i = 0; i < length; i++) {
                s .addi((IComplexNumber) getScalar(i).element());
                putScalar(i, s);
            }
        }

        else if(dimension == Integer.MAX_VALUE || dimension == shape.length - 1) {
            IComplexNDArray flattened = ravel().dup();
            IComplexNumber prevVal = (IComplexNumber) flattened.getScalar(0).element();
            for(int i = 1; i < flattened.length(); i++) {
                IComplexNumber d = prevVal.add((IComplexNumber) flattened.getScalar(i).element());
                flattened.putScalar(i,d);
                prevVal = d;
            }

            return flattened;
        }



        else {
            for(int i = 0; i < vectorsAlongDimension(dimension); i++) {
                IComplexNDArray vec = vectorAlongDimension(i,dimension);
                vec.cumsumi(0);

            }
        }


        return this;
    }

    @Override
    public IComplexNDArray cumsum(int dimension) {

        return dup().cumsumi(dimension);

    }

    @Override
    public INDArray assign(INDArray arr) {
        return null;
    }

    @Override
    public int vectorsAlongDimension(int dimension) {
        return length / size(dimension);
    }

    /*
    public JCublasComplexNDArray(double[] data,int[] shape,int[] stride) {
        this(data,shape,stride,0);
    }
    public JCublasComplexNDArray(double[] newData) {
        this(newData.length/2);

        data = newData;
    }
    */
    public JCublasComplexNDArray(int[] shape) {
        this(shape,0);
    }
    public JCublasComplexNDArray(int len) {
        this(len, 1, new double[2 * len]);
    }

    public JCublasComplexNDArray(int[] shape,int offset) {
        this(shape,calcStrides(shape,2),offset);
    }
    public JCublasComplexNDArray(int newRows, int newColumns, double... newData) {
        rows = newRows;
        columns = newColumns;
        length = rows * columns;

        if (newData.length != 2 * newRows * newColumns)
            throw new IllegalArgumentException(
                    "Passed data must match matrix dimensions.");
        data = newData;
    }
    public JCublasComplexNDArray(int[] shape,int[] stride,int offset) {
        this(new double[ArrayUtil.prod(shape) * 2],shape,stride,offset);
    }

    public JCublasComplexNDArray(double[] newData) {
        this(newData.length/2);

        data = newData;
    }
    public JCublasComplexNDArray(double[] data,int[] shape,int[] stride,int offset) {
        this(data);
        if(offset >= data.length)
            throw new IllegalArgumentException("Invalid offset: must be < data.length");

        this.stride = stride;
        initShape(shape);



        this.offset = offset;



        if(data != null  && data.length > 0)
            this.data = data;
    }
    @Override
    public IComplexNDArray vectorAlongDimension(int index, int dimension) {

        return new JCublasComplexNDArray(data,
                new int[]{shape[dimension]}
                ,new int[]{stride[dimension]},
                offset + index);

    }

    @Override
    public IComplexNDArray assign(IComplexNDArray arr) {

        LinAlgExceptions.assertSameShape(this, arr);
        INDArray other = arr.ravel();
        INDArray thisArr = ravel();
        for(int i = 0; i < other.length(); i++)
            thisArr.put(i,other.getScalar(i));
        return this;
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, IComplexNumber element) {
        return null;
    }

    private void initShape(int[] shape) {

        this.shape = shape;

        if(this.shape.length == 2) {
            if(this.shape[0] == 1) {
                this.shape = new int[1];
                this.shape[0] = shape[1];
            }

            if(this.shape.length == 1) {
                rows = 1;
                columns = shape[1];
            }
            else {
                rows = shape[0];
                columns = shape[1];
            }

        }
        //default row vector
        else if(this.shape.length == 1) {
            columns = this.shape[0];
            rows = 1;
        }

        this.length = ArrayUtil.prod(this.shape);
        if(this.stride == null) {
            this.stride = ArrayUtil.calcStrides(this.shape,2);

        }

        if(this.stride.length != this.shape.length) {
            this.stride = ArrayUtil.calcStrides(this.shape,2);

        }

    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, IComplexNDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, Number element) {

        return null;
    }

    @Override
    public IComplexNDArray putScalar(int i, IComplexNumber value) {
        return put(i, NDArrays.scalar(value));
    }

    @Override
    public IComplexNDArray putScalar(int i, Number value) {

        return put(i, NDArrays.scalar(value));
    }

    @Override
    public INDArray putScalar(int[] i, Number value) {

        return null;
    }

    @Override
    public IComplexNDArray lt(Number other) {

        return dup().lti(other);
    }

    @Override
    public IComplexNDArray lti(Number other) {

        return lti(NDArrays.scalar(other));
    }

    @Override
    public IComplexNDArray eq(Number other) {
        return dup().eqi(other);
    }

    @Override
    public IComplexNDArray eqi(Number other) {
        return eqi(NDArrays.scalar(other));
    }

    @Override
    public IComplexNDArray gt(Number other) {
        return dup().gti(other);
    }

    @Override
    public IComplexNDArray gti(Number other) {
        return gti(NDArrays.scalar(other));
    }

    @Override
    public IComplexNDArray lt(INDArray other) {

        return dup().lti(other);
    }

    @Override
    public IComplexNDArray lti(INDArray other) {

        return (JCublasComplexNDArray) Transforms.lt(other);
    }

    @Override
    public IComplexNDArray eq(INDArray other) {

        return dup().eqi(other);
    }

    @Override
    public IComplexNDArray eqi(INDArray other) {
        return (JCublasComplexNDArray) Transforms.eq(other);
    }

    @Override
    public IComplexNDArray gt(INDArray other) {
        return dup().gti(other);
    }

    @Override
    public IComplexNDArray gti(INDArray other) {

        return (JCublasComplexNDArray) Transforms.gt(other);    }

    @Override
    public IComplexNDArray neg() {

        return dup().negi();
    }

    @Override
    public IComplexNDArray negi() {

        return (IComplexNDArray) Transforms.neg(this);
    }

    @Override
    public IComplexNDArray rdiv(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray div(Number n) {
        return dup().divi(n);
    }

    @Override
    public IComplexNDArray divi(Number n) {
        return divi(NDArrays.scalar(n));
    }

    @Override
    public IComplexNDArray mul(Number n) {

        return dup().muli(n);    }

    @Override
    public IComplexNDArray muli(Number n) {
        return muli(NDArrays.scalar(n));

    }

    @Override
    public IComplexNDArray sub(Number n) {

        return dup().subi(n);
    }

    @Override
    public IComplexNDArray subi(Number n) {

        return subi(NDArrays.scalar(n));
    }

    @Override
    public IComplexNDArray add(Number n) {
        return dup().addi(n);
    }

    @Override
    public IComplexNDArray addi(Number n) {
        return addi(NDArrays.scalar(n));
    }

    @Override
    public IComplexNDArray get(NDArrayIndex... indexes) {
        return null;
    }

    @Override
    public IComplexNDArray getColumns(int[] cindices) {
        INDArray rows = NDArrays.create(rows(),cindices.length);
        for(int i = 0; i < cindices.length; i++) {
            rows.putColumn(i,getColumn(cindices[i]));
        }
        return (JCublasComplexNDArray) rows;    }

    @Override
    public IComplexNDArray getRows(int[] rindices) {
        INDArray rows = NDArrays.create(rindices.length,columns());
        for(int i = 0; i < rindices.length; i++) {
            rows.putRow(i,getRow(rindices[i]));
        }
        return (JCublasComplexNDArray) rows;    }


    @Override
    public IComplexNDArray min(int dimension) {
        return null; // todo fix this one
    }

    @Override
    public IComplexNDArray max(int dimension) {
        return null;
    }


    public IComplexNumber max() {
        IComplexNDArray reshape = ravel();
        IComplexNumber max = (IComplexNumber) reshape.getScalar(0).element();

        for(int i = 1; i < reshape.length(); i++) {
            IComplexNumber curr = (IComplexNumber) reshape.getScalar(i).element();
            double val = curr.realComponent().doubleValue();
            if(val > curr.realComponent().doubleValue())
                max = curr;

        }
        return max;
    }
/*
    @Override
    public IComplexNDArray max(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return NDArrays.scalar(reshape(new int[]{1,length}).max());
        }

        else if(isVector()) {
            return NDArrays.scalar(max());
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(), dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    INDArray arr2 = (INDArray) nd.getResult();
                    arr.put(i.get(),arr2.max(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 *
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.max(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape).transpose();
        }
    }
*/

    public JCublasComplexDouble get(int i) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");
        int idx = linearIndex(i);
        return new JCublasComplexDouble(data[idx],data[idx + 1]);
    }
        
    @Override
    public IComplexNDArray put(int i, int j, INDArray element) {
        return put(new int[]{i,j},element);
    }


    public JCublasComplexDouble scalar() {
        return new JCublasComplexDouble(get(0));
    }

    @Override
    public INDArray put(int i, int j, Number element) {
        return put(new int[]{i,j},element);    }

    private INDArray put(int[] ints, Number element) {
        return null;
    }

    @Override
    public IComplexNDArray put(int[] indices, INDArray element) {
        if(!element.isScalar())
            throw new IllegalArgumentException("Unable to insert anything but a scalar");
        int ix = offset;
        if (indices.length != shape.length)
            throw new IllegalArgumentException("Unable to applyTransformToDestination values: number of indices must be equal to the shape");

        for (int i = 0; i< shape.length; i++)
            ix += indices[i] * stride[i];

        if(element instanceof IComplexNDArray) {
            IComplexNumber element2 = (IComplexNumber) element.element();
            data[ix] = (double) element2.realComponent();
            data[ix + 1]= (double) element2.imaginaryComponent();
        }
        else {
            double element2 = (double) element.element();
            data[ix] = element2;
            data[ix + 1]= 0;
        }

        return this;    }

    private void assertSlice(INDArray put,int slice) {
        assert slice <= slices() : "Invalid slice specified " + slice;
        int[] sliceShape = put.shape();
        int[] requiredShape = ArrayUtil.removeIndex(shape(),0);
        //no need to compare for scalar; primarily due to shapes either being [1] or length 0
        if(put.isScalar())
            return;

        assert Shape.shapeEquals(sliceShape, requiredShape) : String.format("Invalid shape size of %s . Should have been %s ", Arrays.toString(sliceShape),Arrays.toString(requiredShape));

    }

    @Override
    public IComplexNDArray putSlice(int slice, INDArray put) {
        if(isScalar()) {
            assert put.isScalar() : "Invalid dimension. Can only insert a scalar in to another scalar";
            put(0,put.getScalar(0));
            return this;
        }

        else if(isVector()) {
            assert put.isScalar() : "Invalid dimension on insertion. Can only insert scalars input vectors";
            put(slice,put.getScalar(0));
            return this;
        }


        assertSlice(put,slice);


        INDArray view = slice(slice);

        if(put.isScalar())
            put(slice,put.getScalar(0));
        else if(put.isVector())
            for(int i = 0; i < put.length(); i++)
                view.put(i,put.getScalar(i));
        else if(put.shape().length == 2)
            for(int i = 0; i < put.rows(); i++)
                for(int j = 0; j < put.columns(); j++) {
                    view.put(i,j,NDArrays.scalar((IComplexNumber) put.getScalar(i, j).element()));

                }

        else {

            assert put.slices() == view.slices() : "Slices must be equivalent.";
            for(int i = 0; i < put.slices(); i++)
                view.slice(i).putSlice(i,view.slice(i));

        }

        return this;
    }
    //@Override
    public JCublasComplexNDArray put(int i, org.jblas.ComplexDouble v) {
        if(i > length)
            throw new IllegalArgumentException("Unable to insert element " + v + " at index " + i + " with length " + length);
        int linearIndex = linearIndex(i);
        data[linearIndex] = v.real();
        data[linearIndex + 1] = v.imag();
        return this;
    }
    public DimensionSlice vectorForDimensionAndOffsetPair(int dimension, int offset,int currOffsetForSlice) {
        int count = 0;
        JCublasComplexNDArray ret = new JCublasComplexNDArray(new int[]{shape[dimension]});
        boolean newSlice = false;
        List<Integer> indices = new ArrayList<>();
        for(int j = offset; count < this.shape[dimension]; j+= this.stride[dimension] ) {
            ComplexDouble d = new ComplexDouble(data[j],data[j + 1]);
            indices.add(j);
            ret.put(count++,d);
            if(j >= currOffsetForSlice)
                newSlice = true;

        }

        return new DimensionSlice(newSlice,ret,ArrayUtil.toArray(indices));
    }
    public DimensionSlice vectorForDimensionAndOffset(int dimension, int offset) {
        if(isScalar() && dimension == 0 && offset == 0)
            return new DimensionSlice(false,JCublasComplexNDArray.scalar(get(offset)),new int[]{offset});


            //need whole vector
        else  if (isVector()) {
            if(dimension == 0) {
                int[] indices = new int[length];
                for(int i = 0; i < indices.length; i++)
                    indices[i] = i;
                return new DimensionSlice(false,dup(),indices);
            }

            else if(dimension == 1) {
                return new DimensionSlice(false,JCublasComplexNDArray.scalar(get(offset)),new int[]{offset});
            }

            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);
        }


        else {
            int count = 0;
            JCublasComplexNDArray ret = new JCublasComplexNDArray(new int[]{shape[dimension]});
            List<Integer> indices = new ArrayList<>();
            for (int j = offset; count < this.shape[dimension]; j += this.stride[dimension] ) {
                ComplexDouble d = new ComplexDouble(data[j], data[j + 1]);
                ret.put(count++, d);
                indices.add(j);
            }

            return new DimensionSlice(false, ret, ArrayUtil.toArray(indices));
        }

    }
    @Override
    public void iterateOverDimension(int dimension, SliceOp op, boolean modify) {
        if(isScalar()) {
            if(dimension > 1)
                throw new IllegalArgumentException("Dimension must be 0 for a scalar");
            else {
                DimensionSlice slice = this.vectorForDimensionAndOffset(0,0);
                op.operate(slice);

                if(modify && slice.getIndices() != null) {
                    JCublasComplexNDArray result = (JCublasComplexNDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = result.get(i).realComponent();
                        data[slice.getIndices()[i] + 1] = result.get(i).imaginaryComponent();
                    }
                }
            }
        }



        else if(isVector()) {
            if(dimension == 0) {
                DimensionSlice slice = vectorForDimensionAndOffset(0,0);
                op.operate(slice);
                if(modify && slice.getIndices() != null) {
                    JCublasComplexNDArray result = (JCublasComplexNDArray) slice.getResult();
                    for(int i = 0; i < slice.getIndices().length; i++) {
                        data[slice.getIndices()[i]] = result.get(i).realComponent();
                        data[slice.getIndices()[i] + 1] = result.get(i).imaginaryComponent();
                    }
                }
            }
            else if(dimension == 1) {
                for(int i = 0; i < length; i++) {
                    DimensionSlice slice = vectorForDimensionAndOffset(dimension,i);
                    op.operate(slice);
                    if(modify && slice.getIndices() != null) {
                        JCublasComplexNDArray result = (JCublasComplexNDArray) slice.getResult();
                        for(int j = 0; j < slice.getIndices().length; j++) {
                            data[slice.getIndices()[j]] = result.get(j).realComponent();
                            data[slice.getIndices()[j] + 1] = result.get(j).imaginaryComponent();
                        }
                    }
                }
            }
            else
                throw new IllegalArgumentException("Illegal dimension for vector " + dimension);
        }

        else {
            if(dimension >= shape.length)
                throw new IllegalArgumentException("Unable to remove dimension  " + dimension + " was >= shape length");


            int[] shape = ArrayUtil.removeIndex(this.shape,dimension);

            if(dimension == 0) {
                //iterating along the dimension is relative to the number of slices
                //in the return dimension
                int numTimes = ArrayUtil.prod(shape);
                //note difference here from ndarray, the offset is incremented by 2 every time
                //note also numtimes is multiplied by 2, this is due to the complex and imaginary components
                for(int offset = this.offset; offset < numTimes ; offset += 2) {
                    DimensionSlice vector = vectorForDimensionAndOffset(dimension,offset);
                    op.operate(vector);
                    if(modify && vector.getIndices() != null) {
                        JCublasComplexNDArray result = (JCublasComplexNDArray) vector.getResult();
                        for(int i = 0; i < vector.getIndices().length; i++) {
                            data[vector.getIndices()[i]] = result.get(i).realComponent();
                            data[vector.getIndices()[i] + 1] = result.get(i).imaginaryComponent();
                        }
                    }

                }

            }

            else {
                //needs to be 2 * shape: this is due to both realComponent and imaginary components
                double[] data2 = new double[ArrayUtil.prod(shape) ];
                int dataIter = 0;
                //want the milestone to slice[1] and beyond
                int[] sliceIndices = endsForSlices();
                int currOffset = 0;

                //iterating along the dimension is relative to the number of slices
                //in the return dimension
                //note here the  and +=2 this is for iterating over realComponent and imaginary components
                for(int offset = this.offset;;) {
                    if(dataIter >= data2.length || currOffset >= sliceIndices.length)
                        break;

                    //do the operation,, and look for whether it exceeded the current slice
                    DimensionSlice pair = vectorForDimensionAndOffsetPair(dimension, offset,sliceIndices[currOffset]);
                    //append the result
                    op.operate(pair);


                    if(modify && pair.getIndices() != null) {
                        JCublasComplexNDArray result = (JCublasComplexNDArray) pair.getResult();
                        for(int i = 0; i < pair.getIndices().length; i++) {
                            data[pair.getIndices()[i]] = result.get(i).realComponent();
                            data[pair.getIndices()[i] + 1] = result.get(i).imaginaryComponent();
                        }
                    }

                    //go to next slice and iterate over that
                    if(pair.isNextSlice()) {
                        //DO NOT CHANGE
                        currOffset++;
                        if(currOffset >= sliceIndices.length)
                            break;
                        //will update to next step
                        offset = sliceIndices[currOffset];
                    }

                }

            }


        }
    }
    private ComplexDouble reduceVector(Ops.DimensionOp op,JCublasComplexNDArray vector) {

        switch(op) {
            case SUM:
                return (ComplexDouble) vector.sum(0).element();
            case MEAN:
                return (ComplexDouble) vector.mean(0).element();
            case NORM_1:
                return new ComplexDouble(vector.norm1());
            case NORM_2:
                return new ComplexDouble(vector.norm2());
            case NORM_MAX:
                return new ComplexDouble(vector.normmax());
            case FFT:
            default: throw new IllegalArgumentException("Illegal operation");
        }
    }

    @Override
    public IComplexNDArray reduce(Ops.DimensionOp op, int dimension) {
        if(isScalar())
            return this;


        if(isVector())
            return NDArrays.scalar(reduceVector(op, this));


        int[] shape = ArrayUtil.removeIndex(this.shape,dimension);

        if(dimension == 0) {
            double[] data2 = new double[ArrayUtil.prod(shape) * 2];
            int dataIter = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for(int offset = this.offset; offset < numTimes; offset++) {
                ComplexDouble reduce = op(dimension, offset, op);
                data2[dataIter++] = reduce.real();
                data2[dataIter++] = reduce.imag();


            }

            return NDArrays.createComplex(data2,shape);
        }

        else {
            double[] data2 = new double[ArrayUtil.prod(shape)];
            int dataIter = 0;
            //want the milestone to slice[1] and beyond
            int[] sliceIndices = endsForSlices();
            int currOffset = 0;

            //iterating along the dimension is relative to the number of slices
            //in the return dimension
            int numTimes = ArrayUtil.prod(shape);
            for(int offset = this.offset; offset < numTimes; offset++) {
                if(dataIter >= data2.length || currOffset >= sliceIndices.length)
                    break;

                //do the operation,, and look for whether it exceeded the current slice
                ComplexIterationResult pair = op(dimension, offset, op,sliceIndices[currOffset]);
                //append the result
                IComplexNumber reduce = pair.getNumber();
                data2[dataIter++] = reduce.realComponent().doubleValue();
                data2[dataIter++] = reduce.imaginaryComponent().doubleValue();
                //go to next slice and iterate over that
                if(pair.isNextIteration()) {
                    //will update to next step
                    offset = sliceIndices[currOffset];
                    numTimes +=  sliceIndices[currOffset];
                    currOffset++;
                }

            }

            return NDArrays.createComplex(data2,shape);
        }
    }

    //getFromOrigin one result along one dimension based on the given offset
    private ComplexIterationResult op(int dimension, int offset, Ops.DimensionOp op,int currOffsetForSlice) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        boolean newSlice = false;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
            if(j >= currOffsetForSlice)
                newSlice = true;
        }

        ComplexDoubleMatrix r = new ComplexDoubleMatrix(dim);
        JCublasComplexNDArray wrapped = JCublasComplexNDArray.wrap(r);
        ComplexDouble r2 = reduceVector(op,wrapped);
        return new ComplexIterationResult(newSlice,r2);
    }
    public static JCublasComplexNDArray wrap(JCublasComplexNDArray toWrap) {
        if(toWrap instanceof JCublasComplexNDArray)
            return (JCublasComplexNDArray) toWrap;
        int[] shape;
        if(toWrap.isColumnVector())
            shape = new int[]{toWrap.columns};
        else if(toWrap.isRowVector())
            shape = new int[]{ toWrap.rows};
        else
            shape = new int[]{toWrap.rows,toWrap.columns};
        JCublasComplexNDArray ret = new JCublasComplexNDArray(toWrap.data,shape);
        return ret;
    }

    public static JCublasComplexNDArray wrap(JCublasComplexNDArray ndArray,JCublasComplexNDArray toWrap) {
        if(toWrap instanceof JCublasComplexNDArray)
            return (JCublasComplexNDArray) toWrap;
        int[] stride = ndArray.stride();
        JCublasComplexNDArray ret = new JCublasComplexNDArray(toWrap.data,ndArray.shape(),stride,ndArray.offset());
        return ret;
    }
    @Override
    public IComplexNDArray getScalar(int... indexes) {
        return null;
    }

    @Override
    public void checkDimensions(INDArray other) {

    }

    @Override
    public int[] endsForSlices() {
        return new int[0];
    }

    @Override
    public IComplexNDArray assign(Number value) {
        return null;
    }

    @Override
    public int linearIndex(int i) {
        return 0;
    }

    @Override
    public void iterateOverAllRows(SliceOp op) {

    }

    @Override
    public IComplexNDArray rdiv(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray rdiv(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray hermitian() {
        return null;
    }

    @Override
    public IComplexNDArray conj() {
        return null;
    }

    @Override
    public IComplexNDArray conji() {
        return null;
    }

    @Override
    public INDArray getReal() {
        return null;
    }

    @Override
    public IComplexNDArray repmat(int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray putRow(int row, INDArray toPut) {
        return null;
    }

    @Override
    public IComplexNDArray putColumn(int column, INDArray toPut) {
        return null;
    }

    @Override
    public IComplexNDArray getScalar(int row, int column) {
        return null;
    }

    @Override
    public IComplexNDArray getScalar(int i) {
        return null;
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
    public INDArray put(NDArrayIndex[] indices, INDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray put(int i, INDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray diviColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray divColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray diviRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray divRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray muliColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray mulColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray muliRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray mulRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray subiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray subColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray subiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray subRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray addiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray addColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray addiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray addRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray mmul(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray mmul(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray div(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray div(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray mul(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray mul(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray sub(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray sub(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray add(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray add(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray mmuli(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray mmuli(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray divi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray divi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray muli(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray muli(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray subi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray subi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray addi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray addi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray normmax(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray norm2(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray norm1(int dimension) {
        return null;
    }

    @Override
    public INDArray std(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray prod(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray mean(int dimension) {
        return null;
    }

    @Override
    public INDArray var(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray sum(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray get(int[] indices) {
        return null;
    }

    @Override
    public IComplexNDArray dup() {
        return null;
    }

    @Override
    public IComplexNDArray ravel() {
        return null;
    }

    @Override
    public int slices() {
        return 0;
    }

    @Override
    public IComplexNDArray slice(int i, int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray slice(int i) {
        return null;
    }

    @Override
    public int offset() {
        return 0;
    }

    @Override
    public IComplexNDArray reshape(int[] newShape) {
        return null;
    }

    @Override
    public INDArray reshape(int rows, int columns) {
        return null;
    }

    @Override
    public IComplexNDArray transpose() {
        return null;
    }

    @Override
    public IComplexNDArray swapAxes(int dimension, int with) {
        return null;
    }

    @Override
    public IComplexNDArray permute(int[] rearrange) {
        return null;
    }

    @Override
    public IComplexNDArray getColumn(int i) {
        return null;
    }

    @Override
    public IComplexNDArray getRow(int i) {
        return null;
    }

    @Override
    public int columns() {
        return 0;
    }

    @Override
    public int rows() {
        return 0;
    }

    @Override
    public boolean isColumnVector() {
        return false;
    }

    @Override
    public boolean isRowVector() {
        return false;
    }

    @Override
    public boolean isVector() {
        return false;
    }

    @Override
    public boolean isMatrix() {
        return false;
    }

    @Override
    public boolean isScalar() {
        return false;
    }

    @Override
    public int[] shape() {
        return new int[0];
    }

    @Override
    public int[] stride() {
        return new int[0];
    }

    @Override
    public int size(int dimension) {
        return 0;
    }

    @Override
    public int length() {
        return 0;
    }

    @Override
    public IComplexNDArray broadcast(int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray broadcasti(int[] shape) {
        return null;
    }

    @Override
    public Object element() {
        return null;
    }

    @Override
    public double[] data() {
        return new double[0];
    }

    @Override
    public void setData(double[] data) {

    }

    @Override
    public float[] floatData() {
        return new float[0];
    }

    @Override
    public void setData(float[] data) {

    }
}
