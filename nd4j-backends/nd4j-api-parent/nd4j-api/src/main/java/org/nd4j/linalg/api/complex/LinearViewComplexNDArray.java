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

package org.nd4j.linalg.api.complex;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.Indices;
import org.nd4j.linalg.api.shape.Shape;

import java.util.ArrayList;
import java.util.List;


/**
 * Row vector view of an ndarray
 * @author Adam Gibson
 */
@Deprecated
public class LinearViewComplexNDArray extends BaseComplexNDArray {
    private IComplexNDArray wrapped;
    private List<INDArray> vectors;

    /**
     * Create the view based on
     * the given ndarray
     * @param wrapped the ndarray to
     *                base the linear view on
     */
    public LinearViewComplexNDArray(IComplexNDArray wrapped) {
        if(wrapped.getLeadingOnes() > 0 || wrapped.getTrailingOnes() > 0) {
            wrapped = Nd4j.createComplex(wrapped.data(), Shape.squeeze(wrapped.shape()));
        }

     /*   this.wrapped = wrapped;
        this.shape = new int[] {1,wrapped.length()};
        this.data = wrapped.data();
        this.offset = wrapped.offset();
        this.ordering = wrapped.ordering();
        this.length = wrapped.length();
        vectors = new ArrayList<>();
        collectRows(wrapped);*/

    }


    protected void collectRows(INDArray slice) {
        if(slice.isRowVector()) {
            vectors.add(slice);
        }
        else if(slice.isMatrix()) {
            for(int i = 0; i < slice.rows(); i++)
                vectors.add(slice.getRow(i));
        }
        else
            for(int i = 0; i < slice.slices(); i++)
                collectRows(slice.slice(i));
    }



    @Override
    public boolean isCleanedUp() {
        return wrapped.isCleanedUp();
    }

    @Override
    public void cleanup() {
        wrapped.cleanup();
    }

    @Override
    public void resetLinearView() {

    }

    @Override
    public int secondaryStride() {
        return wrapped.secondaryStride();
    }

    @Override
    public int majorStride() {
        return wrapped.majorStride();
    }

    @Override
    public IComplexNDArray linearView() {
        return this;
    }

    @Override
    public IComplexNDArray linearViewColumnOrder() {
        return this;
    }

    @Override
    public int vectorsAlongDimension(int dimension) {
        if(dimension > 1)
            throw new IllegalArgumentException("Linear view does not have dimensions greater than 1");
        return 1;
    }

    @Override
    public IComplexNDArray vectorAlongDimension(int index, int dimension) {
        if(dimension == 0 || dimension == 1 && index == 0)
            return this;
        throw new IllegalArgumentException("Index must be 0 and dimension must be 0 or 1");
    }

    @Override
    public IComplexNumber getComplex(int i) {
        if(wrapped.isVector())
            return wrapped.getComplex(i);
        int vectorSize = wrapped.size(-1);
        int vectorIdx = Indices.rowNumber(i,wrapped);

        IComplexNDArray currVector = (IComplexNDArray) vectors.get(vectorIdx);
        int offset = vectorSize * vectorIdx;


        int idx =  i - offset;
        return currVector.getComplex(idx);
    }



    @Override
    public IComplexNDArray putScalar(int i, double value) {
        int vectorSize = wrapped.size(-1);
        int vectorIdx = Indices.rowNumber(i, wrapped);

        INDArray currVector = vectors.get(vectorIdx);
        int offset = vectorSize * vectorIdx;


        int idx =  i - offset;
        currVector.putScalar(idx, value);
        return this;
    }

    @Override
    public IComplexNDArray putScalar(int i, IComplexNumber value) {
        int vectorSize = wrapped.size(-1);
        int vectorIdx = Indices.rowNumber(i,wrapped);

        IComplexNDArray currVector = (IComplexNDArray) vectors.get(vectorIdx);
        int offset = vectorSize * vectorIdx;


        int idx =  i - offset;
        currVector.putScalar(idx, value);
        return this;
    }

    @Override
    public int length() {
        return wrapped.length();
    }

    @Override
    public long lengthLong() {
        return wrapped.length();
    }

    @Override
    public double getDouble(int i) {
        if(wrapped.isVector())
            return wrapped.getDouble(i);
        int vectorSize = wrapped.size(-1);
        int vectorIdx = Indices.rowNumber(i,wrapped);

        INDArray currVector = vectors.get(vectorIdx);
        int offset = vectorSize * vectorIdx;


        int idx =  i - offset;

        return currVector.getDouble(idx);
    }


    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("[");
        for(int i = 0; i < wrapped.length(); i++) {
            sb.append(getComplex(i));
            if(i < wrapped.length()  - 1) {
                sb.append(",");
            }
        }

        sb.append("]");
        return sb.toString();
    }
}
