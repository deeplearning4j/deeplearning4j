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

    /*    var inputBegin = tNode.getInputs().get(1);
        var inputEnd = tNode.getInputs().get(2);
        var inputStrides = tNode.getInputs().get(3);


        var iArgs = new ArrayList<Integer>();

        // bit masks for this slice
        var bm = nodeDef.getAttrOrThrow("begin_mask");
        var xm = nodeDef.getAttrOrThrow("ellipsis_mask");
        var em = nodeDef.getAttrOrThrow("end_mask");
        var nm = nodeDef.getAttrOrThrow("new_axis_mask");
        var sm = nodeDef.getAttrOrThrow("shrink_axis_mask");

        iArgs.add((int) bm.getI());
        iArgs.add((int) xm.getI());
        iArgs.add((int) em.getI());

        iArgs.add((int) nm.getI());
        iArgs.add((int) sm.getI());

        if (inputBegin.getNode() < 0 && inputEnd.getNode() < 0 && inputStrides.getNode() < 0) {

            // order matters, hehe
            var strides = graph.getVariableSpace().getVariable(tNode.getInputs().remove(3));
            var end = graph.getVariableSpace().getVariable(tNode.getInputs().remove(2));
            var begin = graph.getVariableSpace().getVariable(tNode.getInputs().remove(1));

            for (int e = 0; e < begin.getArray().length(); e++)
                iArgs.add((int) begin.getArray().getInt(e));

            for (int e = 0; e < end.getArray().length(); e++)
                iArgs.add((int) end.getArray().getInt(e));

            for (int e = 0; e < strides.getArray().length(); e++)
                iArgs.add((int) strides.getArray().getInt(e));
        } else {
            // do nothing
        }

        var bits = Ints.toArray(iArgs);*/
    }






}
