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

import lombok.Getter;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.dataset.DataSet;

/**
 * SuffleSymmetric is specialOp, suited for shuffling examples within dataset
 *
 * @author raver119@gmail.com
 */
public class ShuffleSymmetric extends BaseTransformOp {

    @Getter private INDArray features;
    @Getter private INDArray featuresMask;
    @Getter private INDArray labels;
    @Getter private INDArray labelsMask;

    public ShuffleSymmetric() {

    }

    public ShuffleSymmetric(DataSet dataSet) {
        this.features = dataSet.getFeatures();
        this.featuresMask = dataSet.getFeaturesMaskArray();
        this.labels = dataSet.getLabels();
        this.labelsMask = dataSet.getLabelsMaskArray();
    }

    public ShuffleSymmetric(INDArray features, INDArray labels) {
        this.features = features;
        this.labels = labels;
    }

    public ShuffleSymmetric(INDArray features, INDArray featuresMask, INDArray labels, INDArray labelsMask ) {
        this.features = features;
        this.featuresMask = featuresMask;
        this.labels = labels;
        this.labelsMask = labelsMask;
    }


    @Override
    public int opNum() {
        return 49;
    }

    @Override
    public String name() {
        return "shuffle_symmetric";
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return null;
    }

    @Override
    public float op(float origin, float other) {
        return 0;
    }

    @Override
    public double op(double origin, double other) {
        return 0;
    }

    @Override
    public double op(double origin) {
        return 0;
    }

    @Override
    public float op(float origin) {
        return 0;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }
}
