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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.HashSet;
import java.util.Set;

/**
 * Collects the linear indexes
 * for a given vector.
 * This is primarily used for internal use in linear view.
 *
 * @author Adam Gibson
 */
public class LinearIndex extends BaseTransformOp {

    private int internalCount = 0;
    private int[] indices;
    private boolean wholeArray = false;
    private Set<Integer> encountered = new HashSet<>();

    public LinearIndex() {
    }

    public LinearIndex(INDArray x) {
       this(x,true);
    }

    public LinearIndex(INDArray x, INDArray z,boolean wholeArray) {
        super(x, z);
        this.wholeArray = wholeArray;
        initIndexesIfNecessary();
    }

    public LinearIndex(INDArray x, INDArray z, int n,boolean wholeArray) {
        super(x, z, n);
        this.wholeArray = wholeArray;
        initIndexesIfNecessary();

    }

    public LinearIndex(INDArray x, INDArray y, INDArray z, int n,boolean wholeArray) {
        super(x, y, z, n);
        this.wholeArray = wholeArray;
        initIndexesIfNecessary();

    }

    public LinearIndex(INDArray x,boolean wholeArray) {
        super(x);
        this.wholeArray = wholeArray;
        initIndexesIfNecessary();
    }

    @Override
    public String name() {
        return "linearindex";
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        addToIndex();
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        addToIndex();
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        addToIndex();
        return origin;
    }

    @Override
    public float op(float origin, float other) {
        addToIndex();
        return origin;
    }

    @Override
    public double op(double origin, double other) {
        addToIndex();
        return origin;
    }

    @Override
    public double op(double origin) {
        addToIndex();
        return origin;
    }

    @Override
    public float op(float origin) {
        addToIndex();
        return origin;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        addToIndex();
        return origin;
    }


    private void addToIndex() {
        if(!wholeArray)
            return;
        int idx =  getLinearIndex();
        if(!encountered.contains(idx))
            encountered.add(idx);
        else
            throw new IllegalStateException("Please checking striding. Index: " + idx + " already encountered ");
        indices[internalCount] = idx;
        internalCount++;
        numProcessed++;
    }

    //get the linear index for the current number processed
    private int getLinearIndex() {
        return x.linearIndex(numProcessed);
    }

    //initialize the index
    private void initIndexesIfNecessary() {
        if(wholeArray)
            indices = new int[x.length()];
    }

    /**
     * The linear indices collected
     * iterating over the array
     * @return the linear indices
     * collected iterating over the array
     */
    public int[] getIndices() {
        return indices;
    }

    @Override
    public void exec() {
        for(int i = 0; i < x.length(); i++) {
            addToIndex();
        }
    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        INDArray xAlongDimension = x.vectorAlongDimension(index, dimension);

        if (y() != null)
            return new LinearIndex(xAlongDimension, y.vectorAlongDimension(index, dimension), z.vectorAlongDimension(index, dimension), xAlongDimension.length(),false);
        else
            return new LinearIndex(xAlongDimension, z.vectorAlongDimension(index, dimension), x.length(),false);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        INDArray xAlongDimension = x.tensorAlongDimension(index, dimension);

        if (y() != null)
            return new LinearIndex(xAlongDimension, y.tensorAlongDimension(index, dimension), z.tensorAlongDimension(index, dimension), xAlongDimension.length(),false);
        else
            return new LinearIndex(xAlongDimension, z.tensorAlongDimension(index, dimension), x.length(),false);

    }
}

