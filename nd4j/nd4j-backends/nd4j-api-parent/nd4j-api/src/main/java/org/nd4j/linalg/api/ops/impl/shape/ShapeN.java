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

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Returns the shape of N input array as as N output arrays
 *
 * @author Alex Black
 */
public class ShapeN extends DynamicCustomOp {

    public ShapeN() {}

    public ShapeN(SameDiff sameDiff, SDVariable[] inputs, boolean inPlace) {
        super(null, sameDiff, inputs, inPlace);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }


    @Override
    public String opName() {
        return "shapes_of";
    }

    @Override
    public String tensorflowName() {
        return "ShapeN";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        List<SDVariable> out = new ArrayList<>();
        for(SDVariable in : args()){
            out.add(f().zerosLike(in));
        }
        return out;
    }

    @Override
    public int getNumOutputs(){
        return args().length;
    }
}
