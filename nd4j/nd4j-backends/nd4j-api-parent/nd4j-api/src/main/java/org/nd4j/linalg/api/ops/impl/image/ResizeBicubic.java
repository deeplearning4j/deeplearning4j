/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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
package org.nd4j.linalg.api.ops.impl.image;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * ResizeBicubic op wrapper
 * @author Alexander Stoyakin
 */
@NoArgsConstructor
public class ResizeBicubic extends DynamicCustomOp {

    protected boolean alignCorners = false;
    protected boolean alignPixelCenters = false;

    public ResizeBicubic(@NonNull INDArray image, INDArray size, boolean alignCorners, boolean alignPixelCenters) {
        addInputArgument(image, size);
        addBArgument(alignCorners, alignPixelCenters);
    }

    public ResizeBicubic(@NonNull SameDiff sameDiff, @NonNull  SDVariable image,
                         SDVariable size, boolean alignCorners, boolean alignPixelCenters) {
        super(sameDiff, new SDVariable[]{image, size});
        addBArgument(alignCorners, alignPixelCenters);
    }

    @Override
    public String opName() {
        return "resize_bicubic";
    }

    @Override
    public String tensorflowName() {
        return "ResizeBicubic";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);

        this.alignCorners = attributesForNode.get("align_corners").getB();
        this.alignPixelCenters = attributesForNode.get("half_pixel_centers").getB();
        addBArgument(alignCorners, alignPixelCenters);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && (inputDataTypes.size() == 1 || inputDataTypes.size() == 2),
                "Expected 1 or 2 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(Nd4j.defaultFloatingPointType());
    }
}
