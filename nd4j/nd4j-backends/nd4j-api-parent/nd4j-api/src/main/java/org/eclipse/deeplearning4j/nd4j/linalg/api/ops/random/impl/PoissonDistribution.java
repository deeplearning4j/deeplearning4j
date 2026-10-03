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
package org.eclipse.deeplearning4j.nd4j.linalg.api.ops.random.impl;

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

/**
 * Poisson samples (legacy random op 15). Every element of z is a Poisson(lambda) sample: one rate for all of them, or
 * one rate per element. Each element draws from its own stream of the generator (Knuth's method below lambda 10,
 * transformed rejection with squeeze from 10 on), in float, or in double for a DOUBLE z. A rate of 0 gives 0, a
 * negative rate NaN.
 */
public class PoissonDistribution extends BaseRandomOp {
    private double lambda;

    public PoissonDistribution() {
        super();
    }

    public PoissonDistribution(SameDiff sd, double lambda, DataType dataType, long[] shape) {
        super(sd, shape);
        this.lambda = lambda;
        this.dataType = dataType;
        this.extraArgs = new Object[] {this.lambda};
    }

    public PoissonDistribution(double lambda, DataType dataType, long... shape) {
        this(Nd4j.createUninitialized(dataType, shape), lambda);
    }

    /** Fills z with Poisson(lambda) samples. */
    public PoissonDistribution(@NonNull INDArray z, double lambda) {
        super(null, null, z);
        this.lambda = lambda;
        this.extraArgs = new Object[] {this.lambda};
    }

    /**
     * Fills z with Poisson samples, element i with the rate lambdas[i]. lambdas has z's shape; the op reads it in z's
     * data type, so other types are cast.
     */
    public PoissonDistribution(@NonNull INDArray z, @NonNull INDArray lambdas) {
        super(inType(lambdas, z), null, z);
        Preconditions.checkArgument(Arrays.equals(lambdas.shape(), z.shape()),
                "Rates of shape %s for an output of shape %s: they must have the same shape", lambdas.shape(), z.shape());
        this.lambda = 0.0;
        this.extraArgs = new Object[] {this.lambda};
    }

    /** The array in the output's data type. */
    static INDArray inType(INDArray parameters, INDArray z) {
        return parameters.dataType() == z.dataType() ? parameters : parameters.castTo(z.dataType());
    }

    @Override
    public int opNum() {
        return 15;
    }

    @Override
    public String opName() {
        return "distribution_poisson";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }

    @Override
    public List<DataBuffer> calculateOutputShape(OpContext oc) {
        return calculateOutputShape();
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        LongShapeDescriptor longShapeDescriptor = LongShapeDescriptor.fromShape(shape, dataType);
        return Arrays.asList(Nd4j.createBuffer(longShapeDescriptor.toShapeInfo()));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.emptyList();
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes == null || inputDataTypes.isEmpty(),
                "Expected no input datatypes (no args) for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(dataType);
    }
}
