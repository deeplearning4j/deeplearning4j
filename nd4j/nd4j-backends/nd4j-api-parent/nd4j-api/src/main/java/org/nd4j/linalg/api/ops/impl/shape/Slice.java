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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Slice function
 *
 * @author Adam Gibson
 */
@Slf4j
public class Slice extends DynamicCustomOp {

    private int[] begin;
    private int[] size;

    public Slice() {}

    public Slice(SameDiff sameDiff, @NonNull SDVariable input, @NonNull int[] begin, @NonNull int[] size){
        super(null, sameDiff, new SDVariable[]{input});
        this.begin = begin;
        this.size = size;
        addIArgument(begin);
        addIArgument(size);
    }


    @Override
    public String opName() {
        return "slice";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Slice";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Collections.singletonList(f().sliceBp(arg(), grad.get(0), begin, size));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null & (dataTypes.size() == 1 || dataTypes.size() == 3),
                "Expected list with 1 or 3 datatypes for %s, got %s", getClass(), dataTypes);
        //Output type is same as input type. 3 inputs for import case
        return Collections.singletonList(dataTypes.get(0));
    }
}
