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

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class BinomialDistribution extends BaseRandomOp {
    private int trials;
    private double probability;

    public BinomialDistribution(SameDiff sd, int trials, double probability, long[] shape){
        super(sd, shape);
        this.trials = trials;
        this.probability = probability;
        this.extraArgs = new Object[] {(double) this.trials, this.probability};
    }

    public BinomialDistribution(SameDiff sd, int trials, double probability, DataType dataType, long[] shape){
        this(sd, trials, probability, shape);
        super.dataType = dataType;
    }

    public BinomialDistribution(int trials, double probability, DataType dt, long[] shape){
        this(Nd4j.createUninitialized(dt, shape), trials, probability);
    }

    public BinomialDistribution() {
        super();
    }

    /**
     * This op fills Z with binomial distribution over given trials with single given probability for all trials
     * @param z
     * @param trials
     * @param probability
     */
    public BinomialDistribution(@NonNull INDArray z, int trials, double probability) {
        super(z, z, z);
        this.trials = trials;
        this.probability = probability;
        this.extraArgs = new Object[] {(double) this.trials, this.probability};
    }

    /**
     * This op fills Z with binomial distribution over given trials with probability for each trial given as probabilities INDArray
     * @param z
     * @param trials
     * @param probabilities array with probability value for each trial
     */
    public BinomialDistribution(@NonNull INDArray z, int trials, @NonNull INDArray probabilities) {
        super(z, probabilities, z);
        if (trials > probabilities.length())
            throw new IllegalStateException("Number of trials is > then amount of probabilities provided");

        if (probabilities.elementWiseStride() < 1)
            throw new IllegalStateException("Probabilities array shouldn't have negative elementWiseStride");

        Preconditions.checkArgument(probabilities.dataType() == z.dataType(), "Probabilities and Z operand should have same data type");

        this.trials = trials;
        this.probability = 0.0;
        this.extraArgs = new Object[] {(double) this.trials, this.probability};
    }

    /**
     * This op fills Z with binomial distribution over given trials with probability for each trial given as probabilities INDArray
     *
     * @param z
     * @param probabilities
     */
    public BinomialDistribution(@NonNull INDArray z, @NonNull INDArray probabilities) {
        this(z, (int) probabilities.length(), probabilities);
    }

    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String opName() {
        return "distribution_binomial";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public List<DataBuffer> calculateOutputShape(OpContext oc) {
        return calculateOutputShape();
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        LongShapeDescriptor longShapeDescriptor = LongShapeDescriptor.fromShape(shape,dataType);
        return Arrays.asList(Nd4j.createBuffer(longShapeDescriptor.toShapeInfo()));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.emptyList();
    }

    @Override
    public void setZ(INDArray z) {
        //We want all 3 args set to z for this op
        this.x = z;
        this.y = z;
        this.z = z;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes == null || inputDataTypes.isEmpty(), "Expected no input datatypes (no args) for %s, got %s", getClass(), inputDataTypes);
        //Input data type specifies the shape; output data type should be any float
        //TODO MAKE CONFIGUREABLE - https://github.com/eclipse/deeplearning4j/issues/6854
        return Collections.singletonList(DataType.DOUBLE);
    }

    @Override
    public boolean isTripleArgRngOp() {
        return true;
    }
}
