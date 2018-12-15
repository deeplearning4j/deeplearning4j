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

package org.nd4j.linalg.api.ops.impl.scatter;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;


/**
 * Scatter ND add operation
 * @author Alex Black
 */
public class ScatterNdUpdate extends DynamicCustomOp {

    public ScatterNdUpdate(SameDiff sameDiff, SDVariable ref, SDVariable indices, SDVariable updates) {
        super(null, sameDiff, new SDVariable[]{ref, indices, updates}, false);
    }

    public ScatterNdUpdate(){}

    @Override
    public String opName() {
        return "scatter_nd_update";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ScatterNdUpdate";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradOut){
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 input datatypes, got %s", inputDataTypes);
        Preconditions.checkState(inputDataTypes.get(0) == inputDataTypes.get(2), "Reference (input 0) and updates (input 2) must have exactly same data types, got %s and %s",
                inputDataTypes.get(0), inputDataTypes.get(2));
        return Collections.singletonList(inputDataTypes.get(0));
    }

}
