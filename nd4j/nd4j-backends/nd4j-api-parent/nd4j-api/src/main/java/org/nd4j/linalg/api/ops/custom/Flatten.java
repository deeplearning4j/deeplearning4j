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

package org.nd4j.linalg.api.ops.custom;

import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 * This op takes arbitrary number of arrays as input, and returns single "flattened" vector
 *
 * @author raver119@gmail.com
 */
public class Flatten extends DynamicCustomOp {
    private char order;

    public Flatten() {
        //
    }

    public Flatten(char order, INDArray... inputs) {
        this.order = order;

        for (val in:inputs)
            inputArguments.add(in);

        iArguments.add(Long.valueOf((int) this.order));
    }

    public Flatten(INDArray output, INDArray... inputs) {
        this(output.ordering(), inputs);

        outputArguments.add(output);
    }

    @Override
    public String opName() {
        return "flatten";
    }


}
