package org.deeplearning4j.linalg.jcublas.complex;

import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.DimensionSlice;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.api.ndarray.SliceOp;
import org.deeplearning4j.linalg.factory.NDArrayFactory;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
//import org.deeplearning4j.linalg.jblas.complex.ComplexDouble;
import org.deeplearning4j.linalg.jcublas.JCublasNDArray;
import org.deeplearning4j.linalg.jcublas.SimpleJCublas;
import org.deeplearning4j.linalg.ops.TwoArrayOps;
import org.deeplearning4j.linalg.ops.elementwise.AddOp;
import org.deeplearning4j.linalg.ops.elementwise.DivideOp;
import org.deeplearning4j.linalg.ops.elementwise.MultiplyOp;
import org.deeplearning4j.linalg.ops.elementwise.SubtractOp;
import org.deeplearning4j.linalg.ops.reduceops.Ops;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.linalg.util.ArrayUtil;
import org.deeplearning4j.linalg.util.ComplexIterationResult;
import org.deeplearning4j.linalg.util.LinAlgExceptions;
import org.deeplearning4j.linalg.util.Shape;
import org.jblas.NativeBlas;
import org.jblas.exceptions.SizeException;
import org.jblas.ranges.Range;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.deeplearning4j.linalg.util.ArrayUtil.*;

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
    private char ordering;

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
    public JCublasComplexNDArray(double[] data,int[] shape) {this(data,shape,0);}
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
    public JCublasComplexNDArray divi(Number n) {
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


    public JCublasComplexNDArray(IComplexNumber[] newData,int[] shape,int[] stride) {
        this(newData.length/2);

        data = new double[ArrayUtil.prod(shape) * 2];
        this.stride = stride;
        initShape(shape);
        for(int i = 0;i  < length; i++)
            put(i,(JCublasComplexDouble) newData[i].asDouble());

    }
    public JCublasComplexDouble scalar() {
        return new JCublasComplexDouble(get(0));
    }
    public JCublasComplexNDArray(List<IComplexNDArray> slices,int[] shape,int[] stride) {
        this(slices,shape,stride,NDArrays.order());
    }
    public JCublasComplexNDArray(List<IComplexNDArray> slices,int[] shape,int[] stride,char ordering) {
        this(new double[ArrayUtil.prod(shape) ].length);

        data = new double[ArrayUtil.prod(shape) * 2];

        List<JCublasComplexDouble> list = new ArrayList<>();
        for(int i = 0; i < slices.size(); i++) {
            IComplexNDArray flattened = slices.get(i).ravel();
            for(int j = 0; j < flattened.length(); j++)
                list.add((JCublasComplexDouble) flattened.getScalar(j).element());
        }


        this.ordering = ordering;
        this.data = new double[ArrayUtil.prod(shape) * 2 ];
        this.stride = stride;
        int count = 0;
        for (int i = 0; i < list.size(); i++) {
            data[count] = list.get(i).realComponent();
            data[count + 1] = list.get(i).imaginaryComponent();
            count += 2;
        }

        initShape(shape);
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
    public JCublasComplexNDArray put(int i, JCublasComplexDouble v) {
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
            JCublasComplexDouble d = new JCublasComplexDouble(data[j],data[j + 1]);
            indices.add(j);
            ret.put(count++,d);
            if(j >= currOffsetForSlice)
                newSlice = true;

        }

        return new DimensionSlice(newSlice,ret,ArrayUtil.toArray(indices));
    }

    public static JCublasComplexNDArray scalar(JCublasComplexNDArray from,int index) {
        return new JCublasComplexNDArray(from.data,new int[]{1},new int[]{1},index);
    }
    public static JCublasComplexNDArray scalar(JCublasComplexNDArray from) {
        return scalar(from,0);
    }
    public static JCublasComplexNDArray scalar(double num) {
        return new JCublasComplexNDArray(new double[]{num,0},new int[]{1},new int[]{1},0);
    }

    public static JCublasComplexNDArray scalar(JCublasComplexDouble num) {
        return new JCublasComplexNDArray(new double[]{num.realComponent(),num.imaginaryComponent()},new int[]{1},new int[]{1},0);
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
                JCublasComplexDouble d = new JCublasComplexDouble(data[j], data[j + 1]);
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

    public JCublasComplexNDArray _columnSums() {
        JCublasComplexNDArray v =
                new JCublasComplexNDArray(1, columns);

        for (int c = 0; c < columns; c++)
            v.put(c, getColumn(c).sum());

        return v;
    }
    //@Override
    public JCublasComplexNDArray columnSums() {
        if(shape().length == 2) {
            return JCublasComplexNDArray.wrap(this._columnSums());

        }
        else
            return JCublasComplexNDArrayUtil.doSliceWise(JCublasComplexNDArrayUtil.MatrixOp.COLUMN_SUM,this);

    }

    public JCublasComplexNDArray _columnMeans() {
        return _columnSums().divi(rows);
    }

    public JCublasComplexNDArray columnMeans() {
        if(shape().length == 2) {
            return JCublasComplexNDArray.wrap(this._columnMeans());

        }

        else
            return JCublasComplexNDArrayUtil.doSliceWise(JCublasComplexNDArrayUtil.MatrixOp.COLUMN_MEAN,this);

    }

    public JCublasComplexNDArray _rowSums() {
        JCublasComplexNDArray v = new JCublasComplexNDArray(rows);

        for (int r = 0; r < rows; r++)
            v.put(r, getRow(r).sum());

        return v;
    }

    //@Override
    public JCublasComplexNDArray rowSums() {
        if(shape().length == 2) {
            return JCublasComplexNDArray.wrap(this._rowSums());

        }

        else
            return JCublasComplexNDArrayUtil.doSliceWise(JCublasComplexNDArrayUtil.MatrixOp.ROW_SUM,this);

    }
    //@Override
    public double normmax() {
        if(isVector() ) {
            int i = SimpleJCublas.iamax(this);
            return get(i).abs();
        }
        return JCublasComplexNDArrayUtil.doSliceWise(JCublasComplexNDArrayUtil.ScalarOp.NORM_MAX,this).real();

    }

    public JCublasComplexNDArray _rowMeans() {
        return rowSums().divi(columns);
    }

    //@Override
    public JCublasComplexNDArray rowMeans() {
        if(shape().length == 2) {
            return JCublasComplexNDArray.wrap(this._rowMeans());

        }
        else
            return JCublasComplexNDArrayUtil.doSliceWise(JCublasComplexNDArrayUtil.MatrixOp.ROW_MEAN,this);

    }
    private JCublasComplexDouble reduceVector(Ops.DimensionOp op,JCublasComplexNDArray vector) {

        switch(op) {
            case SUM:
                return (JCublasComplexDouble) vector.sum(0).element();
            case MEAN:
                return (JCublasComplexDouble) vector.mean(0).element();
            case NORM_1:
                return new JCublasComplexDouble(vector.norm1());
            case NORM_2:
                return new JCublasComplexDouble(vector.norm2());
            case NORM_MAX:
                return new JCublasComplexDouble(vector.normmax());
            case FFT:
            default: throw new IllegalArgumentException("Illegal operation");
        }
    }

    public double norm2() {
        return SimpleJCublas.nrm2(this);
    }
    //@Override
    public double norm1() {
        if(isVector())
            return norm2();
        return JCublasComplexNDArrayUtil.doSliceWise(JCublasComplexNDArrayUtil.ScalarOp.NORM_1, this).real();

    }

    /** Get real part of the matrix. */
    public JCublasNDArray real() {
        JCublasNDArray result = new JCublasNDArray(rows, columns);
        NativeBlas.dcopy(length, data, 0, 2, result.data, 0, 1);
        return result;
    }

    /** Get imaginary part of the matrix. */
    public JCublasNDArray imag() {
        JCublasNDArray result = new JCublasNDArray(rows, columns);
        NativeBlas.dcopy(length, data, 1, 2, result.data, 0, 1);
        return result;
    }

    private JCublasComplexDouble op(int dimension, int offset, Ops.DimensionOp op) {
        double[] dim = new double[this.shape[dimension]];
        int count = 0;
        for(int j = offset; count < dim.length; j+= this.stride[dimension]) {
            double d = data[j];
            dim[count++] = d;
        }

        return reduceVector(op,JCublasComplexNDArray.wrap(new JCublasComplexNDArray(dim)));
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
                JCublasComplexDouble reduce = op(dimension, offset, op);
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

        JCublasComplexNDArray r = new JCublasComplexNDArray(dim);
        JCublasComplexNDArray wrapped = JCublasComplexNDArray.wrap(r);
        JCublasComplexDouble r2 = reduceVector(op,wrapped);
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
        int[] ret = new int[slices()];
        int currOffset = offset;
        for(int i = 0; i < slices(); i++) {
            ret[i] = (currOffset );
            currOffset += stride[0];
        }
        return ret;
    }

    @Override
    public IComplexNDArray assign(Number value) {
        IComplexNDArray one = reshape(new int[]{1,length});
        for(int i = 0; i < one.length(); i++)
            one.put(i,NDArrays.complexScalar(value));
        return one;    }

    @Override
    public int linearIndex(int i) {
        int realStride = getRealStrideForLinearIndex();
        int idx = offset + i * realStride;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }
    private int getRealStrideForLinearIndex() {
        if(stride.length != shape.length)
            throw new IllegalStateException("Stride and shape not equal length");
        if(shape.length == 1)
            return stride[0];
        if(shape.length == 2) {
            if(shape[0] == 1)
                return stride[1];
            if(shape[1] == 1)
                return stride[0];
        }
        return stride[0];
    }
    @Override
    public void iterateOverAllRows(SliceOp op) {

        if(isVector())
            op.operate(new DimensionSlice(false,this,null));

        else {
            for(int i = 0; i < slices(); i++) {
                IComplexNDArray slice = slice(i);
                slice.iterateOverAllRows(op);
            }
        }

    }

    @Override
    public IComplexNDArray rdiv(INDArray other) {
        return dup().rdivi(other);    }

    @Override
    public IComplexNDArray rdivi(INDArray other) {
        return rdivi(other,this);
    }

    @Override
    public IComplexNDArray rdiv(INDArray other, INDArray result) {
        return dup().rdivi(other,result);
    }

    @Override
    public IComplexNDArray rdivi(INDArray other, INDArray result) {
        return (IComplexNDArray) other.divi(this, result);
    }

    @Override
    public IComplexNDArray rsub(INDArray other, INDArray result) {
        return dup().rsubi(other,result);
    }

    @Override
    public IComplexNDArray rsub(INDArray other) {
        return dup().rsubi(other);
    }

    @Override
    public IComplexNDArray rsubi(INDArray other) {
        return rsubi(other,this);
    }

    @Override
    public IComplexNDArray rsubi(INDArray other, INDArray result) {
        return (IComplexNDArray) other.subi(this, result);
    }

    //@Override
    public JCublasComplexNDArray put(int rowIndex, int columnIndex, JCublasComplexDouble value) {
        int i =  index(rowIndex, columnIndex);
        data[i] = value.real();
        data[i+1] = value.imag();
        return this;
    }

    @Override
    public IComplexNDArray hermitian() {
        JCublasComplexNDArray result = new JCublasComplexNDArray(shape());

        JCublasComplexDouble c = new JCublasComplexDouble(0);

        for (int i = 0; i < slices(); i++)
            for (int j = 0; j < columns; j++)
                result.put(j, i, get(i, j, c).conji());
        return result;    }

    @Override
    public IComplexNDArray conj() {
        return dup().conji();
    }

    public JCublasComplexDouble get(int i, JCublasComplexDouble result) {
        return result.set(data[i * 2], data[i*2+1]);
    }


    public JCublasComplexDouble get(int rowIndex, int columnIndex, JCublasComplexDouble result) {
        return get(index(rowIndex, columnIndex), result);
    }


    public int index(int rowIndex, int columnIndex) {
        //System.out.printf("Index for (%d, %d) -> %d\n", rowIndex, columnIndex, (rows * columnIndex + rowIndex) * 2);
        return rows * columnIndex + rowIndex;
    }



    @Override
    public IComplexNDArray conji() {
        JCublasComplexNDArray reshaped = reshape(1,length);
        JCublasComplexDouble c = new JCublasComplexDouble(0.0);
        for (int i = 0; i < length; i++)
            reshaped.put(i, reshaped.get(i, c).conji());
        return this;    }

    @Override
    public JCublasNDArray getReal() {
        int[] stride = ArrayUtil.copy(stride());
        for(int i = 0; i < stride.length; i++)
            stride[i] /= 2;
        JCublasNDArray result = new JCublasNDArray(shape(),stride);
        SimpleJCublas.dcopy(length, data, offset, 2, result.data, 0, 1);
        return result;
    }

    @Override
    public IComplexNDArray repmat(int[] shape) {
        int[] newShape = ArrayUtil.copy(shape());
        assert shape.length <= newShape.length : "Illegal shape: The passed in shape must be <= the current shape length";
        for(int i = 0; i < shape.length; i++)
            newShape[i] *= shape[i];
        IComplexNDArray result = NDArrays.createComplex(newShape);
        //nd copy
        if(isScalar()) {
            for(int i = 0; i < result.length(); i++) {
                result.put(i,getScalar(0));

            }
        }

        else if(isMatrix()) {

            for (int c = 0; c < shape()[1]; c++) {
                for (int r = 0; r < shape()[0]; r++) {
                    for (int i = 0; i < rows(); i++) {
                        for (int j = 0; j < columns(); j++) {
                            result.put(r * rows() + i, c * columns() + j, getScalar(i, j));
                        }
                    }
                }
            }

        }

        else {
            int[] sliceRepmat = ArrayUtil.removeIndex(shape,0);
            for(int i = 0; i < result.slices(); i++) {
                result.putSlice(i,repmat(sliceRepmat));
            }
        }

        return  result;    }

    @Override
    public IComplexNDArray putRow(int row, INDArray toPut) {
        JCublasComplexNDArray n = (JCublasComplexNDArray) toPut;
        JCublasComplexNDArray n2 = n;
        putRow(row,n2);
        return this;    }

    @Override
    public IComplexNDArray putColumn(int column, INDArray toPut) {
        JCublasComplexNDArray n = (JCublasComplexNDArray) toPut;
        JCublasComplexNDArray n2 = n;
        putColumn(column,n2);
        return this;
    }
    //@Override
    public JCublasComplexDouble get(int rowIndex, int columnIndex) {
        int index = offset +  index(rowIndex,columnIndex);
        return new JCublasComplexDouble(data[index],data[index + 1]);
    }

    @Override
    public IComplexNDArray getScalar(int row, int column) {
        return JCublasComplexNDArray.scalar(get(row,column));
    }

    @Override
    public IComplexNDArray getScalar(int i) {
        return JCublasComplexNDArray.scalar(get(i));
    }

    @Override
    public double squaredDistance(INDArray other) {
        double sd = 0.0;
        for (int i = 0; i < length; i++) {
            IComplexNumber diff = (IComplexNumber) getScalar(i).sub(other.getScalar(i)).element();
            double d = (double) diff.absoluteValue();
            sd += d * d;
        }
        return sd;    }

    @Override
    public double distance2(INDArray other) {
        return  Math.sqrt(squaredDistance(other));
    }

    @Override
    public double distance1(INDArray other) {
        double d = 0.0;
        for (int i = 0; i < length; i++) {
            IComplexNumber n = (IComplexNumber) getScalar(i).sub(other.getScalar(i)).element();
            d += n.absoluteValue().doubleValue();
        }
        return d;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, INDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray put(int i, INDArray element) {
        if(element == null)
            throw new IllegalArgumentException("Unable to insert null element");
        assert element.isScalar() : "Unable to insert non scalar element";
        if(element instanceof  IComplexNDArray) {
            put(i,(JCublasComplexDouble) element.element());
        }
        else
            put(i,(double) element.element());
        return this;
    }

    public IComplexNDArray put(int i, double v) {
        if(!isVector() && !isScalar())
            throw new IllegalArgumentException("Unable to do linear indexing with dimensions greater than 1");

        data[linearIndex(i)] = v;
        return this;
    }

    @Override
    public IComplexNDArray diviColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).divi(columnVector.getScalar(i));
        }
        return this;    }

    @Override
    public IComplexNDArray divColumnVector(INDArray columnVector) {
        return dup().diviColumnVector(columnVector);
    }

    @Override
    public IComplexNDArray diviRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).divi(rowVector.getScalar(i));
        }
        return this;
    }

    @Override
    public IComplexNDArray divRowVector(INDArray rowVector) {
        return dup().diviRowVector(rowVector);
    }

    @Override
    public IComplexNDArray muliColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).muli(columnVector.getScalar(i));
        }
        return this;
    }

    @Override
    public IComplexNDArray mulColumnVector(INDArray columnVector) {
        return dup().muliColumnVector(columnVector);
    }

    @Override
    public IComplexNDArray muliRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).muli(rowVector.getScalar(i));
        }
        return this;
    }

    @Override
    public IComplexNDArray mulRowVector(INDArray rowVector) {
        return dup().muliRowVector(rowVector);
    }

    @Override
    public IComplexNDArray subiColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).subi(columnVector.getScalar(i));
        }
        return this;
    }

    @Override
    public IComplexNDArray subColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).subi(columnVector.getScalar(i));
        }
        return this;
    }

    @Override
    public IComplexNDArray subiRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).subi(rowVector.getScalar(i));
        }
        return this;
    }

    @Override
    public IComplexNDArray subRowVector(INDArray rowVector) {
        return dup().subiRowVector(rowVector);
    }

    @Override
    public IComplexNDArray addiColumnVector(INDArray columnVector) {
        for(int i = 0; i < columns(); i++) {
            getColumn(i).addi(columnVector.getScalar(i));
        }
        return this;
    }

    @Override
    public IComplexNDArray addColumnVector(INDArray columnVector) {
        return dup().addiColumnVector(columnVector);
    }

    @Override
    public IComplexNDArray addiRowVector(INDArray rowVector) {
        for(int i = 0; i < rows(); i++) {
            getRow(i).addi(rowVector.getScalar(i));
        }
        return this;
    }

    @Override
    public IComplexNDArray addRowVector(INDArray rowVector) {
        return dup().addiRowVector(rowVector);
    }

    @Override
    public IComplexNDArray mmul(INDArray other) {
        int[] shape = {rows(),other.columns()};
        return mmuli(other,NDArrays.create(shape));
    }

    @Override
    public IComplexNDArray mmul(INDArray other, INDArray result) {
        return dup().mmuli(other,result);
    }

    @Override
    public IComplexNDArray div(INDArray other) {
        return dup().divi(other);
    }

    @Override
    public IComplexNDArray div(INDArray other, INDArray result) {
        return dup().divi(other,result);
    }

    @Override
    public IComplexNDArray mul(INDArray other) {
        return dup().muli(other);
    }

    @Override
    public IComplexNDArray mul(INDArray other, INDArray result) {
        return dup().muli(other,result);
    }

    @Override
    public IComplexNDArray sub(INDArray other) {
        return dup().subi(other);
    }

    @Override
    public IComplexNDArray sub(INDArray other, INDArray result) {
        return dup().subi(other,result);
    }

    @Override
    public IComplexNDArray add(INDArray other) {
        return dup().addi(other);
    }

    @Override
    public IComplexNDArray add(INDArray other, INDArray result) {
        return dup().addi(other,result);
    }

    public boolean multipliesWith(INDArray a) {
        return columns() == a.rows();
    }

    @Override
    public IComplexNDArray mmuli(INDArray other) {
        return mmuli(other,this);
    }
    public void assertMultipliesWith(JCublasComplexNDArray a) {
        if (!multipliesWith(a))
            throw new SizeException("Number of columns of left matrix must be equal to number of rows of right matrix.");
    }

    public void assertMultipliesWith(INDArray a) {
        if (!multipliesWith(a))
            throw new SizeException("Number of columns of left matrix must be equal to number of rows of right matrix.");
    }
    public JCublasComplexNDArray(int[] shape,int[] stride){
        this(shape,stride,0);
    }
    public JCublasComplexNDArray copy(JCublasComplexNDArray a) {
        if (!sameSize(a))
            resize(a.rows, a.columns);

        SimpleJCublas.copy(a, this);
        return a;
    }
    public boolean sameSize(JCublasComplexNDArray a) {
        return rows == a.rows && columns == a.columns;
    }
    public void resize(int newRows, int newColumns) {
        rows = newRows;
        columns = newColumns;
        length = newRows * newColumns;
        data = new double[2 * rows * columns];
    }
    @Override
    public IComplexNDArray mmuli(INDArray other, INDArray result) {
        if (other.isScalar())
            return muli(other.getScalar(0), result);


        JCublasComplexNDArray otherArray = new JCublasComplexNDArray(other);
        JCublasComplexNDArray resultArray = new JCublasComplexNDArray(result);


		/* check sizes and resize if necessary */
        assertMultipliesWith(other);


        if (result == this || result == other) {
			/* actually, blas cannot do multiplications in-place. Therefore, we will fake by
			 * allocating a temporary object on the side and copy the result later.
			 */
            otherArray = otherArray.ravel().reshape(otherArray.shape);

            JCublasComplexNDArray temp = new JCublasComplexNDArray(resultArray.shape(),ArrayUtil.calcStridesFortran(resultArray.shape()));
            temp = SimpleJCublas.gemm(this, otherArray, 1, 0);

            temp.copy(resultArray);

        }
        else {
            otherArray = otherArray.ravel().reshape(otherArray.shape);
            IComplexNDArray thisInput =  this.ravel().reshape(shape());
            resultArray = SimpleJCublas.gemm(thisInput, otherArray, 1, 0);
        }





        return resultArray;    }

    @Override
    public JCublasComplexNDArray divi(INDArray other) {
        return divi(other,this);
    }

    @Override
    public JCublasComplexNDArray divi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(DivideOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(DivideOp.class)
                    .to(result).build().exec();
        return (JCublasComplexNDArray) result;
    }

    @Override
    public IComplexNDArray muli(INDArray other) {
        return muli(other,this);
    }

    @Override
    public IComplexNDArray muli(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(MultiplyOp.class)
                    .to(result).build().exec();

        else
            new TwoArrayOps().from(this).other(other).op(MultiplyOp.class)
                    .to(result).build().exec();
        return (IComplexNDArray) result;
    }

    @Override
    public IComplexNDArray subi(INDArray other) {
        return subi(other,this);
    }

    @Override
    public IComplexNDArray subi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(SubtractOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(SubtractOp.class)
                    .to(result).build().exec();
        return (IComplexNDArray) result;
    }

    @Override
    public IComplexNDArray addi(INDArray other) {
        return addi(other,this);
    }

    @Override
    public IComplexNDArray addi(INDArray other, INDArray result) {
        if(other.isScalar())
            new TwoArrayOps().from(this).scalar(other).op(AddOp.class)
                    .to(result).build().exec();
        else
            new TwoArrayOps().from(this).other(other).op(AddOp.class)
                    .to(result).build().exec();
        return (IComplexNDArray) result;
    }


    @Override
    public IComplexNDArray normmax(int dimension) {
        if(isVector()) {
            return JCublasComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                    arr.put(i.get(),arr2.normmax(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.normmax(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }
    }

    @Override
    public IComplexNDArray norm2(int dimension) {
        if(dimension == Integer.MAX_VALUE)
            return JCublasComplexNDArray.scalar(norm2());
        if(isVector()) {
            return JCublasComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                    arr.put(i.get(),arr2.norm1(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.norm1(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }
    }

    @Override
    public IComplexNDArray norm1(int dimension) {
        if(dimension == Integer.MAX_VALUE)
            return JCublasComplexNDArray.scalar(norm1());

        else if(isVector()) {
            return JCublasComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                    arr.put(i.get(),arr2.norm1(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.norm1(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }    }

    public JCublasComplexDouble std() {
        StandardDeviation dev = new StandardDeviation();
        INDArray real = getReal();
        JCublasNDArray imag = imag();
        double std = dev.evaluate(real.data());
        double std2 = dev.evaluate(imag.data());
        return new JCublasComplexDouble(std,std2);
    }
    @Override
    public INDArray std(int dimension) {
        if(dimension == Integer.MAX_VALUE)
            return JCublasComplexNDArray.scalar(std());
        if(isVector()) {
            return JCublasComplexNDArray.scalar(sum().divi(length()));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                    arr.put(i.get(),arr2.std(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.std(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);
        }
    }
    public JCublasComplexNDArray prod() {
        JCublasComplexNDArray d = new JCublasComplexNDArray(1);

        if(isVector()) {
            for(int i = 0; i < length(); i++) {
                JCublasComplexNDArray d2 = (JCublasComplexNDArray) getScalar(i).element();
                d.muli(d2);
            }
        }
        else {
            JCublasComplexNDArray reshape = reshape(new int[]{1,length()});
            for(int i = 0; i < reshape.length(); i++) {
                JCublasComplexNDArray d2 = (JCublasComplexNDArray) reshape.getScalar(i).element();
                d.muli(d2);
            }
        }

        return d;

    }
    @Override
    public IComplexNDArray prod(int dimension) {
        if(dimension == Integer.MAX_VALUE) {
            return JCublasComplexNDArray.scalar(reshape(new int[]{1,length}).prod());
        }

        else if(isVector()) {
            return JCublasComplexNDArray.scalar(sum().divi(length));
        }
        else {
            int[] shape = ArrayUtil.removeIndex(shape(),dimension);
            final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
            final AtomicInteger i = new AtomicInteger(0);
            iterateOverDimension(dimension, new SliceOp() {
                @Override
                public void operate(DimensionSlice nd) {
                    IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                    arr.put(i.get(),arr2.prod(0));
                    i.incrementAndGet();
                }

                /**
                 * Operates on an ndarray slice
                 *
                 * @param nd the result to operate on
                 */
                @Override
                public void operate(INDArray nd) {
                    arr.put(i.get(),nd.prod(0));
                    i.incrementAndGet();
                }
            }, false);

            return arr.reshape(shape);

        }
    }


    //@Override
    public IComplexNDArray mean(int dimension_) {
            if(isVector()) {
                return JCublasComplexNDArray.scalar(sum().divi(length()));
            }
            else {
                int[] shape = ArrayUtil.removeIndex(shape(),dimension_);
                final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
                final AtomicInteger i = new AtomicInteger(0);
                iterateOverDimension(dimension_, new SliceOp() {
                    @Override
                    public void operate(DimensionSlice nd) {
                        IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                        arr.put(i.get(),arr2.mean(0));
                        i.incrementAndGet();
                    }

                    /**
                     * Operates on an ndarray slice
                     *
                     * @param nd the result to operate on
                     */
                    @Override
                    public void operate(INDArray nd) {
                        arr.put(i.get(),nd.mean(0));
                        i.incrementAndGet();
                    }
                }, false);

                return arr.reshape(shape);
            }
    }

    //@Override


    public double var() {
        double mean = (double) mean(Integer.MAX_VALUE).element();
        return StatUtils.variance(data(), mean);
    }

    public INDArray var(int dimension) {
            if(dimension == Integer.MAX_VALUE) {
                return JCublasNDArray.scalar(reshape(new int[]{1,length}).var());
            }
            else if(isVector()) {
                return JCublasNDArray.scalar(var());
            }
            else {
                int[] shape = ArrayUtil.removeIndex(shape(),dimension);
                final INDArray arr = NDArrays.create(new int[]{ArrayUtil.prod(shape)});
                final AtomicInteger i = new AtomicInteger(0);
                iterateOverDimension(dimension, new SliceOp() {
                    @Override
                    public void operate(DimensionSlice nd) {
                        INDArray arr2 = (INDArray) nd.getResult();
                        arr.put(i.get(),arr2.var(0));
                        i.incrementAndGet();
                    }

                    /**
                     * Operates on an ndarray slice
                     *
                     * @param nd the result to operate on
                     */
                    @Override
                    public void operate(INDArray nd) {
                        arr.put(i.get(),nd.var(0));
                        i.incrementAndGet();
                    }
                }, false);

                return arr.reshape(shape);
            }
        }

    public JCublasComplexDouble sum() {
        JCublasComplexDouble d = new JCublasComplexDouble(0);

        if(isVector()) {
            for(int i = 0; i < length(); i++) {
                JCublasComplexDouble d2 = (JCublasComplexDouble) getScalar(i).element();
                d.addi(d2);
            }
        }
        else {
            JCublasComplexNDArray reshape = reshape(new int[]{1,length()});
            for(int i = 0; i < reshape.length(); i++) {
                JCublasComplexDouble d2 = (JCublasComplexDouble) reshape.getScalar(i).element();
                d.addi(d2);
            }
        }

        return d;
    }


    @Override
    public IComplexNDArray sum(int dimension_) {
            if(dimension_ == Integer.MAX_VALUE)
                return JCublasComplexNDArray.scalar(sum());
            if(isVector()) {
                return JCublasComplexNDArray.scalar(sum().divi(length()));
            }
            else {
                int[] shape = ArrayUtil.removeIndex(shape(),dimension_);
                final IComplexNDArray arr = NDArrays.createComplex(new int[]{ArrayUtil.prod(shape)});
                final AtomicInteger i = new AtomicInteger(0);
                iterateOverDimension(dimension_, new SliceOp() {
                    @Override
                    public void operate(DimensionSlice nd) {
                        IComplexNDArray arr2 = (IComplexNDArray) nd.getResult();
                        arr.put(i.get(),arr2.sum(0));
                        i.incrementAndGet();
                    }

                    /**
                     * Operates on an ndarray slice
                     *
                     * @param nd the result to operate on
                     */
                    @Override
                    public void operate(INDArray nd) {
                        arr.put(i.get(),nd.sum(0));
                        i.incrementAndGet();
                    }
                }, false);

                return arr.reshape(shape);
            }    }

    @Override
    public void setStride(int[] stride) {
            this.stride = stride;

    }

    public int[] offsetsForSlices() {
        int[] ret = new int[slices()];
        int currOffset = offset;
        for(int i = 0; i < slices(); i++) {
            ret[i] = currOffset;
            currOffset += stride[0] ;
        }
        return ret;
    }

    @Override
    public INDArray subArray(int[] offsets, int[] shape, int[] stride) {
        int n = shape.length;
        if (offsets.length != n)
            throw new IllegalArgumentException("Invalid offset " + Arrays.toString(offsets));
        if (shape.length != n)
            throw new IllegalArgumentException("Invalid shape " + Arrays.toString(shape));

        if (Arrays.equals(shape, this.shape)) {
            if (ArrayUtil.isZero(offsets)) {
                return this;
            } else {
                throw new IllegalArgumentException("Invalid subArray offsets");
            }
        }

        return new JCublasComplexNDArray(
                data
                , Arrays.copyOf(shape,shape.length)
                , stride
                ,offset + ArrayUtil.dotProduct(offsets, stride)
        );    }

    @Override
    public IComplexNDArray get(int[] indices) {
        JCublasComplexNDArray result = new JCublasComplexNDArray(data,new int[]{1,indices.length},stride,offset);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, get(indices[i]));
        }

        return result;
    }

    @Override
    public IComplexNDArray dup() {
        double[] dupData = new double[data.length];
        System.arraycopy(data,0,dupData,0,dupData.length);
        JCublasComplexNDArray ret = new JCublasComplexNDArray(dupData,shape,stride,offset,ordering);
        return ret;
    }

    private void sliceVectors(List<JCublasComplexNDArray> list) {
        if(isVector())
            list.add(this);
        else {
            for(int i = 0; i < slices(); i++) {
                slice(i).sliceVectors(list);
            }
        }
    }



    @Override
    public JCublasComplexNDArray ravel() {
        JCublasComplexNDArray ret = new JCublasComplexNDArray(new int[]{1,length});
        List<JCublasComplexNDArray> list = new ArrayList<>();
        sliceVectors(list);
        int count = 0;
        for(int i = 0; i < list.size(); i++) {
            for(int j = 0; j < list.get(i).length; j++)
                ret.put(count++,list.get(i).get(j));
        }
        return ret;
    }

    @Override
    public int slices() {
        if(shape.length < 1)
            return 0;
        return shape[0];
    }

    @Override
    public IComplexNDArray slice(int i, int dimension) {
        int offset = this.offset + (i * stride[0]  );
        JCublasComplexNDArray ret;
        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d ComplexNDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            ret = new JCublasComplexNDArray(
                    data,
                    ArrayUtil.empty(),
                    ArrayUtil.empty(),
                    offset);


            //slice of a matrix is a vector
        else if (shape.length == 2) {
            ret = new JCublasComplexNDArray(
                    data,
                    ArrayUtil.of(shape[1]),
                    Arrays.copyOfRange(stride,1,stride.length),
                    offset

            );

        }

        else {
            if(offset >= data.length)
                throw new IllegalArgumentException("Offset index is > data.length");
            ret = new JCublasComplexNDArray(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset);
        }

        ret.ordering = ordering;
        return ret;
    }

    @Override
    public JCublasComplexNDArray slice(int i) {
        int offset = this.offset + (i * stride[0]  );
        JCublasComplexNDArray ret;
        if (shape.length == 0)
            throw new IllegalArgumentException("Can't slice a 0-d ComplexNDArray");

            //slice of a vector is a scalar
        else if (shape.length == 1)
            ret = new JCublasComplexNDArray(
                    data,
                    ArrayUtil.empty(),
                    ArrayUtil.empty(),
                    offset);


            //slice of a matrix is a vector
        else if (shape.length == 2) {
            ret = new JCublasComplexNDArray(
                    data,
                    ArrayUtil.of(shape[1]),
                    Arrays.copyOfRange(stride,1,stride.length),
                    offset

            );

        }

        else {
            if(offset >= data.length)
                throw new IllegalArgumentException("Offset index is > data.length");
            ret = new JCublasComplexNDArray(data,
                    Arrays.copyOfRange(shape, 1, shape.length),
                    Arrays.copyOfRange(stride, 1, stride.length),
                    offset);
        }

        ret.ordering = ordering;
        return ret;
    }

    @Override
    public int offset() {
        return offset;
    }

    @Override
    public JCublasComplexNDArray reshape(int[] newShape) {
        long ec = 1;
        for (int i = 0; i < shape.length; i++) {
            int si = shape[i];
            if (( ec * si ) != (((int) ec ) * si ))
                throw new IllegalArgumentException("Too many elements");
            ec *= shape[i];
        }
        int n= (int) ec;

        if (ec != n)
            throw new IllegalArgumentException("Too many elements");

        JCublasComplexNDArray ndArray = new JCublasComplexNDArray(data,shape,stride,offset);
        ndArray.ordering = ordering;
        return ndArray;

    }

    @Override
    public JCublasComplexNDArray reshape(int rows, int columns) {
        return reshape(new int[]{rows,columns});
    }
    public JCublasComplexNDArray(double[] data,int[] shape,int offset) {
        this(data,shape,offset,NDArrays.order());
    }
    public JCublasComplexNDArray(double[] data,int[] shape,int offset,char ordering) {
        this(data,shape,ordering == NDArrayFactory.C ? calcStrides(shape,2) : calcStridesFortran(shape,2),offset,ordering);
    }
    public JCublasComplexNDArray(double[] data,int[] shape,int[] stride,int offset,char ordering) {
        this(data);
        if(offset >= data.length)
            throw new IllegalArgumentException("Invalid offset: must be < data.length");
        this.ordering = ordering;
        this.stride = stride;
        initShape(shape);



        this.offset = offset;



        if(data != null  && data.length > 0)
            this.data = data;
    }
    @Override
    public IComplexNDArray transpose() {
        //transpose of row vector is column vector
        if(isRowVector())
            return new JCublasComplexNDArray(data,new int[]{shape[0],1},offset);
            //transpose of a column vector is row vector
        else if(isColumnVector())
            return new JCublasComplexNDArray(data,new int[]{shape[0]},offset);

        JCublasComplexNDArray n = new JCublasComplexNDArray(data,reverseCopy(shape),reverseCopy(stride),offset);
        return n;
    }

    @Override
    public IComplexNDArray swapAxes(int dimension, int with) {
        int[] shape = ArrayUtil.range(0,shape().length);
        shape[dimension] = with;
        shape[with] = dimension;
        return permute(shape);
    }

    private int[] doPermuteSwap(int[] shape,int[] rearrange) {
        int[] ret = new int[shape.length];
        for(int i = 0; i < shape.length; i++) {
            ret[i] = shape[rearrange[i]];
        }
        return ret;
    }
    private void checkArrangeArray(int[] arr) {
        assert arr.length == shape.length : "Invalid rearrangement: number of arrangement != shape";
        for(int i = 0; i < arr.length; i++) {
            if (arr[i] >= arr.length)
                throw new IllegalArgumentException("The specified dimensions can't be swapped. Given element " + i + " was >= number of dimensions");
            if (arr[i] < 0)
                throw new IllegalArgumentException("Invalid dimension: " + i + " : negative value");


        }

        for(int i = 0; i < arr.length; i++) {
            for(int j = 0; j < arr.length; j++) {
                if(i != j && arr[i] == arr[j])
                    throw new IllegalArgumentException("Permute array must have unique elements");
            }
        }

    }
    @Override
    public IComplexNDArray permute(int[] rearrange) {
        checkArrangeArray(rearrange);
        int[] newDims = doPermuteSwap(shape,rearrange);
        int[] newStrides = doPermuteSwap(stride,rearrange);

        JCublasComplexNDArray ret = new JCublasComplexNDArray(data,newDims,newStrides,offset);
        ret.ordering = ordering;
        return ret;
    }

    @Override
    public JCublasComplexNDArray getColumn(int c) {
        if(shape.length == 2) {
            int offset = this.offset + c * 2;
            JCublasComplexNDArray ret =  new JCublasComplexNDArray(
                    data,
                    new int[]{shape[0], 1},
                    new int[]{stride[0], 2},
                    offset
            );
            ret.ordering = ordering;
            return ret;
        }

        else
            throw new IllegalArgumentException("Unable to getFromOrigin row of non 2d matrix");
    }

    @Override
    public JCublasComplexNDArray getRow(int r) {
        if(shape.length == 2) {
            JCublasComplexNDArray ret =  new JCublasComplexNDArray(
                    data,
                    new int[]{shape[1]},
                    new int[]{stride[1]},
                    offset + (r * 2) * columns()
            );

            ret.ordering = ordering;
            return ret;
        }
        else
            throw new IllegalArgumentException("Unable to getFromOrigin row of non 2d matrix");
    }

    @Override
    public int columns() {
        if(isMatrix()) {
            if (shape().length > 2)
                return Shape.squeeze(shape)[1];
            else if (shape().length == 2)
                return shape[1];
        }

        if(isVector()) {
            if(isColumnVector())
                return 1;
            else
                return shape[0];
        }
        throw new IllegalStateException("Unable to getFromOrigin number of of rows for a non 2d matrix");
    }

    @Override
    public int rows() {
        if(isMatrix()) {
            if (shape().length > 2)
                return Shape.squeeze(shape)[0];
            else if (shape().length == 2)
                return shape[0];
        }
        else if(isVector()) {
            if(isRowVector())
                return 1;
            else
                return shape[0];
        }
        throw new IllegalStateException("Unable to getFromOrigin number of of rows for a non 2d matrix");
    }

    @Override
    public boolean isColumnVector() {
        return false;
    }

    @Override
    public boolean isRowVector() {
        if(shape().length == 1)
            return false;

        if(isVector())
            return shape()[1] == 1;

        return false;
    }

    @Override
    public boolean isVector() {
        return shape.length == 1
                ||
                shape.length == 1  && shape[0] == 1
                ||
                shape.length == 2 && (shape[0] == 1 || shape[1] == 1) && !isScalar();
    }

    @Override
    public boolean isMatrix() {
        return shape().length == 2 ||
                shape.length == 3
                        && (shape[0] == 1 || shape[1] == 1 || shape[2] == 1);
    }

    @Override
    public boolean isScalar() {
        if(shape.length == 0)
            return true;
        else if(shape.length == 1 && shape[0] == 1)
            return true;
        else if(shape.length >= 2) {
            for(int i = 0; i < shape.length; i++)
                if(shape[i] != 1)
                    return false;
        }

        return length == 1;
    }

    @Override
    public int[] shape() {
        return shape;
    }

    @Override
    public int[] stride() {
        return stride;
    }

    @Override
    public char ordering() {
        return ordering;
    }

    @Override
    public int size(int dimension) {
        if(isScalar()) {
            if(dimension == 0)
                return length;
            else
                throw new IllegalArgumentException("Illegal dimension for scalar " + dimension);
        }

        else if(isVector()) {
            if(dimension == 0 || dimension == 1)
                return length;
            else
                throw new IllegalArgumentException("No dimension for vector " + dimension);
        }


        return shape[dimension];
    }

    @Override
    public int length() {
        return length;
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
        double[] ret = new double[length * 2];
        JCublasComplexNDArray flattened = ravel();
        int count = 0;
        for(int i = 0; i < flattened.length; i++) {
            ret[count++] = flattened.get(i).realComponent();
            ret[count++] = flattened.get(i).imaginaryComponent();
        }

        return ret;
    }

    @Override
    public void setData(double[] data) {
        this.data = data;
    }

    @Override
    public float[] floatData() {
        return ArrayUtil.floatCopyOf(data);
    }

    @Override
    public void setData(float[] data) {
        this.data = ArrayUtil.doubleCopyOf(data);
    }

    public JCublasComplexNDArray get(Range rs, Range cs) {
        rs.init(0, rows());
        cs.init(0, columns());
        JCublasComplexNDArray result = new JCublasComplexNDArray(rs.length(), cs.length());
        result.ordering = ordering;
        for (; rs.hasMore(); rs.next()) {
            cs.init(0, columns());
            for (; cs.hasMore(); cs.next()) {
                result.put(rs.index(), cs.index(), get(rs.value(), cs.value()));
            }
        }

        return result;
    }

    public JCublasComplexDouble unSafeGet(int i) {
        int idx = unSafeLinearIndex(i);
        return new JCublasComplexDouble(data[idx],data[idx + 1]);
    }


    public int unSafeLinearIndex(int i) {
        int realStride = stride[0];
        int idx = offset + i;
        if(idx >= data.length)
            throw new IllegalArgumentException("Illegal index " + idx + " derived from " + i + " with offset of " + offset + " and stride of " + realStride);
        return idx;
    }
}
