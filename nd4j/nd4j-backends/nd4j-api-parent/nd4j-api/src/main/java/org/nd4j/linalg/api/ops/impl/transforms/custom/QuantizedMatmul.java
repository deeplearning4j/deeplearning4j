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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Quantized matrix multiplication.
 *
 * Performs matrix multiplication where inputs may be quantized (integer) types
 * and the output is a floating point type. Supports mixed-precision computation
 * where A and B can be integer or floating point types.
 *
 * Inputs:
 *   0: A tensor (float or integer type)
 *   1: B tensor (float or integer type)
 *
 * Output:
 *   0: result of A x B (floating point type)
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class QuantizedMatmul extends DynamicCustomOp {

    public QuantizedMatmul(SameDiff sameDiff, SDVariable a, SDVariable b) {
        super(null, sameDiff, new SDVariable[]{a, b}, false);
    }

    public QuantizedMatmul(INDArray a, INDArray b) {
        super(new INDArray[]{a, b}, null);
    }

    public QuantizedMatmul(INDArray a, INDArray b, INDArray output) {
        super(new INDArray[]{a, b}, output != null ? new INDArray[]{output} : null);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public String opName() {
        return "quantized_matmul";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes.size() == 2,
                "Expected exactly 2 input datatypes, got %s", dataTypes.size());
        // If either input is floating point, use that type for output.
        // Otherwise default to FLOAT.
        DataType a = dataTypes.get(0);
        DataType b = dataTypes.get(1);
        if (a.isFPType()) {
            return Collections.singletonList(a);
        } else if (b.isFPType()) {
            return Collections.singletonList(b);
        }
        return Collections.singletonList(DataType.FLOAT);
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
