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

package org.nd4j.linalg.api.ops.persistence;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

public class SaveV2 extends DynamicCustomOp {


    @Override
    public String opName() {
        return "savev2";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No op found for saving.");
    }

    @Override
    public String tensorflowName() {
        return "SaveV2";
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
         /*
            strided slice typically takes 4 tensor arguments:
            0) input, it's shape determines number of elements in other arguments
            1) begin indices
            2) end indices
            3) strides
         */

    /*    val inputBegin = tNode.getInputs().get(1);
        val inputEnd = tNode.getInputs().get(2);
        val inputStrides = tNode.getInputs().get(3);


        val iArgs = new ArrayList<Integer>();

        // bit masks for this slice
        val bm = nodeDef.getAttrOrThrow("begin_mask");
        val xm = nodeDef.getAttrOrThrow("ellipsis_mask");
        val em = nodeDef.getAttrOrThrow("end_mask");
        val nm = nodeDef.getAttrOrThrow("new_axis_mask");
        val sm = nodeDef.getAttrOrThrow("shrink_axis_mask");

        iArgs.add((int) bm.getI());
        iArgs.add((int) xm.getI());
        iArgs.add((int) em.getI());

        iArgs.add((int) nm.getI());
        iArgs.add((int) sm.getI());

        if (inputBegin.getNode() < 0 && inputEnd.getNode() < 0 && inputStrides.getNode() < 0) {

            // order matters, hehe
            val strides = graph.getVariableSpace().getVariable(tNode.getInputs().remove(3));
            val end = graph.getVariableSpace().getVariable(tNode.getInputs().remove(2));
            val begin = graph.getVariableSpace().getVariable(tNode.getInputs().remove(1));

            for (int e = 0; e < begin.getArray().length(); e++)
                iArgs.add((int) begin.getArray().getInt(e));

            for (int e = 0; e < end.getArray().length(); e++)
                iArgs.add((int) end.getArray().getInt(e));

            for (int e = 0; e < strides.getArray().length(); e++)
                iArgs.add((int) strides.getArray().getInt(e));
        } else {
            // do nothing
        }

        val bits = Ints.toArray(iArgs);*/
    }






}
