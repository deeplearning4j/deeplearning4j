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
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class AlphaDropOut extends BaseRandomOp {

    private double p;
    private double a;
    private double alphaPrime;
    private double b;

    public AlphaDropOut() {

    }

    public AlphaDropOut(@NonNull INDArray x, double p, double alpha, double alphaPrime, double beta) {
        this(x, x, p, alpha, alphaPrime, beta);
    }

    public AlphaDropOut(@NonNull INDArray x, @NonNull INDArray z, double p, double alpha, double alphaPrime, double beta) {
        super(x,null,z);
        this.p = p;
        this.a = alpha;
        this.b = beta;
        this.alphaPrime = alphaPrime;
        this.extraArgs = new Object[] {p, a, b, alphaPrime};
    }

    @Override
    public int opNum() {
        return 12;
    }

    @Override
    public String opName() {
        return "alpha_dropout";
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
        return null;
    }
}
