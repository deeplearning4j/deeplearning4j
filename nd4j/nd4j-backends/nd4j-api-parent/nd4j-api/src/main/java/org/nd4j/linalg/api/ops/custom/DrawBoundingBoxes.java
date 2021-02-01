/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class DrawBoundingBoxes extends DynamicCustomOp {
    public DrawBoundingBoxes() {}

    public DrawBoundingBoxes(INDArray images, INDArray boxes, INDArray colors) {
        inputArguments.add(images);
        inputArguments.add(boxes);
        inputArguments.add(colors);
    }

    public DrawBoundingBoxes(INDArray images, INDArray boxes, INDArray colors,
                             INDArray output) {
        this(images, boxes, colors);
        outputArguments.add(output);
    }

    public DrawBoundingBoxes(SameDiff sameDiff, SDVariable boxes, SDVariable colors) {
        super("", sameDiff, new SDVariable[]{boxes, colors});
    }

    @Override
    public String opName() {
        return "draw_bounding_boxes";
    }

    @Override
    public String[] tensorflowNames() {
        return new String[]{"DrawBoundingBoxes", "DrawBoundingBoxesV2"};
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}