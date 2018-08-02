/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

import java.util.List;

/**
 * This Op implements numpy.choice method
 * It fills Z from source, following probabilities for each source element
 *
 * @author raver119@gmail.com
 */
public class Choice extends BaseRandomOp {

    public Choice() {
        // no-op
    }

    public Choice(@NonNull INDArray source, @NonNull INDArray probabilities, @NonNull INDArray z) {
        if (source.lengthLong() != probabilities.lengthLong())
            throw new IllegalStateException("From & probabilities length mismatch: " + source.lengthLong() + "/"
                            + probabilities.lengthLong());

        if (probabilities.elementWiseStride() < 1 || source.elementWiseStride() < 1)
            throw new IllegalStateException("Source and probabilities should have element-wise stride >= 1");

        init(source, probabilities, z, z.lengthLong());
        this.extraArgs = new Object[] {0.0};
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public int opNum() {
        return 5;
    }

    @Override
    public String opName() {
        return "choice";
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
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
