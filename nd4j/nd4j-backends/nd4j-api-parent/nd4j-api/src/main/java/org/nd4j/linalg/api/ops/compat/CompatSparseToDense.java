/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.compat;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 * This is a wrapper for SparseToDense op that impelements corresponding TF operation
 *
 * @author raver119@gmail.com
 */
public class CompatSparseToDense extends DynamicCustomOp {

    public CompatSparseToDense() {
        //
    }

    public CompatSparseToDense(INDArray indices, INDArray shape, INDArray values) {
        Preconditions.checkArgument(shape.isZ() && indices.isZ(), "Shape & indices arrays must have one integer data types");
        inputArguments.add(indices);
        inputArguments.add(shape);
        inputArguments.add(values);
    }

    public CompatSparseToDense(INDArray indices, INDArray shape, INDArray values, INDArray defaultVaule) {
        this(indices, shape, values);
        Preconditions.checkArgument(defaultVaule.dataType() == values.dataType(), "Values array must have the same data type as defaultValue array");
        inputArguments.add(defaultVaule);
    }

    @Override
    public String opName() {
        return "compat_sparse_to_dense";
    }
}
