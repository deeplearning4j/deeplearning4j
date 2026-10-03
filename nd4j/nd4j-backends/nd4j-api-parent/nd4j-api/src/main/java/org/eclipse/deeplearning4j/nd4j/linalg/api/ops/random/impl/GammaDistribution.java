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
 * Gamma samples (legacy random op 16). Every element of z is a Gamma(alpha, beta) sample, with shape alpha and rate
 * beta (mean alpha / beta, variance alpha / beta^2): one shape and rate for all elements, one shape per element, or
 * one shape and one rate per element. Each element draws from its own stream of the generator (Marsaglia and Tsang's
 * method), in float, or in double for a DOUBLE z. A shape or rate that is not positive gives NaN.
 */
public class GammaDistribution extends BaseRandomOp {
    private double alpha;
    private double beta;

    public GammaDistribution() {
        super();
    }

    public GammaDistribution(SameDiff sd, double alpha, double beta, DataType dataType, long[] shape) {
        super(sd, shape);
        this.alpha = alpha;
        this.beta = beta;
        this.dataType = dataType;
        this.extraArgs = new Object[] {this.alpha, this.beta};
    }

    public GammaDistribution(double alpha, double beta, DataType dataType, long... shape) {
        this(Nd4j.createUninitialized(dataType, shape), alpha, beta);
    }

    /** Fills z with Gamma(alpha, beta) samples. */
    public GammaDistribution(@NonNull INDArray z, double alpha, double beta) {
        super(null, null, z);
        this.alpha = alpha;
        this.beta = beta;
        this.extraArgs = new Object[] {this.alpha, this.beta};
    }

    /**
     * Fills z with Gamma samples of rate beta, element i with the shape alphas[i]. alphas has z's shape; the op reads
     * it in z's data type, so other types are cast.
     */
    public GammaDistribution(@NonNull INDArray z, @NonNull INDArray alphas, double beta) {
        super(PoissonDistribution.inType(alphas, z), null, z);
        checkShape("Alphas", alphas, z);
        this.alpha = 0.0;
        this.beta = beta;
        this.extraArgs = new Object[] {this.alpha, this.beta};
    }

    /**
     * Fills z with Gamma samples, element i with the shape alphas[i] and the rate betas[i]. Both have z's shape; the op
     * reads them in z's data type, so other types are cast.
     */
    public GammaDistribution(@NonNull INDArray z, @NonNull INDArray alphas, @NonNull INDArray betas) {
        super(PoissonDistribution.inType(alphas, z), PoissonDistribution.inType(betas, z), z);
        checkShape("Alphas", alphas, z);
        checkShape("Betas", betas, z);
        this.alpha = 0.0;
        this.beta = 0.0;
        this.extraArgs = new Object[] {this.alpha, this.beta};
    }

    private static void checkShape(String what, INDArray parameters, INDArray z) {
        Preconditions.checkArgument(Arrays.equals(parameters.shape(), z.shape()),
                "%s of shape %s for an output of shape %s: they must have the same shape", what, parameters.shape(),
                z.shape());
    }

    @Override
    public int opNum() {
        return 16;
    }

    @Override
    public String opName() {
        return "distribution_gamma";
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
