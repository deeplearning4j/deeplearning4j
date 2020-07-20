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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Histogram fixed with op
 *
 * @author Alex Black
 */
public class HistogramFixedWidth extends DynamicCustomOp {

    public HistogramFixedWidth(SameDiff sameDiff, SDVariable values, SDVariable valuesRange, SDVariable numBins) {
        super(sameDiff, numBins == null ? new SDVariable[]{values, valuesRange} : new SDVariable[]{values, valuesRange, numBins}, false);
    }

    public HistogramFixedWidth() {
        //no-op
    }

    @Override
    public String opName() {
        return "histogram_fixed_width";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "HistogramFixedWidth";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        //No op - just need the inputs
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //1 or 2 possible: 2 for TF import (fill with specified value
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 2 || dataTypes.size() == 3),
                "Expected 2 or 3 input datatypes for %s, got %s", getClass(), dataTypes);
        //TODO MAKE CONFIGURABLE
        return Collections.singletonList(DataType.INT);
    }
}
