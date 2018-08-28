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

package org.nd4j.linalg.api.ops.impl.accum;

import lombok.NonNull;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.descriptors.properties.adapters.BooleanAdapter;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Sufficient statistics: returns 3 or 4 output arrays:
 * If shift is not provided: count, sum of elements, sum of squares
 * If shift is provided: count, sum of elements, sum of squares, shift
 *
 * @author Alex Black
 */
public class SufficientStatistics extends DynamicCustomOp {

    public SufficientStatistics() {
    }


    public SufficientStatistics(SameDiff sameDiff, @NonNull SDVariable x, @NonNull SDVariable axis, SDVariable shift) {
        super(null, sameDiff, argsNoNull(x, axis, shift), false);
    }

    private static SDVariable[] argsNoNull(SDVariable x, SDVariable axis, SDVariable shift){
        if(shift == null){
            return new SDVariable[]{x, axis};
        } else {
            return new SDVariable[]{x, axis, shift};
        }
    }


    @Override
    public String opName() {
        return "sufficient_statistics";
    }
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Backprop not yet implemented for op: " + getClass().getSimpleName());
    }

}
