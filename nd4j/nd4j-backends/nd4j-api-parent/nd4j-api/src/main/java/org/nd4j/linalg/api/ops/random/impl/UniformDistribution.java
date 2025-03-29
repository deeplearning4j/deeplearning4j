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

public class UniformDistribution extends BaseRandomOp {
    private double from;
    private double to;

    public UniformDistribution() {
        super();
    }

    public UniformDistribution(SameDiff sd, double from, double to, long[] shape) {
        super(sd, shape);
        this.from = from;
        this.to = to;
        this.extraArgs = new Object[] {this.from, this.to};
    }

    public UniformDistribution(SameDiff sd, double from, double to, DataType dataType, long[] shape) {
        this(sd, from, to, shape);
        this.dataType = dataType;
    }

    public UniformDistribution(double min, double max, DataType datatype, long... shape){
        this(Nd4j.createUninitialized(datatype, shape), min, max);
        this.shape = shape;
    }

    /**
     * This op fills Z with random values within from...to boundaries
     * @param z
     * @param from
     * @param to
     */
    public UniformDistribution(@NonNull INDArray z, double from, double to) {
        super(null, null, z);
        this.from = from;
        this.to = to;
        this.extraArgs = new Object[] {this.from, this.to};
        this.shape = z.shape();
    }

    /**
     * This op fills Z with random values within 0...1
     * @param z
     */
    public UniformDistribution(@NonNull INDArray z) {
        this(z, 0.0, 1.0);
    }

    /**
     * This op fills Z with random values within 0...to
     * @param z
     */
    public UniformDistribution(@NonNull INDArray z, double to) {
        this(z, 0.0, to);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "distribution_uniform";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.emptyList();
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
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes == null || inputDataTypes.isEmpty(), "Expected no input datatypes (no args) for %s, got %s", getClass(), inputDataTypes);
        //Input data type specifies the shape; output data type should be any float
        //TODO MAKE CONFIGUREABLE - https://github.com/eclipse/deeplearning4j/issues/6854
        return Collections.singletonList(dataType);
    }
}
