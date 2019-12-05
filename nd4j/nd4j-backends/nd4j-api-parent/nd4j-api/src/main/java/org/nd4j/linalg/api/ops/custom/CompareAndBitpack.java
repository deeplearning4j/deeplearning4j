/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class CompareAndBitpack extends DynamicCustomOp {
    public CompareAndBitpack() {}

    public CompareAndBitpack(INDArray in, double threshold) {
        inputArguments.add(in);
        inputArguments.add(Nd4j.scalar(threshold));
    }

    public CompareAndBitpack(INDArray in, double threshold, INDArray out) {
        this(in, threshold);
        outputArguments.add(out);
    }

    public CompareAndBitpack(SameDiff sameDiff, SDVariable threshold) {
        super("", sameDiff, new SDVariable[]{threshold});
    }

    @Override
    public String opName() {
        return "compare_and_bitpack";
    }

    @Override
    public String tensorflowName() {
        return "CompareAndBitpack";
    }
}