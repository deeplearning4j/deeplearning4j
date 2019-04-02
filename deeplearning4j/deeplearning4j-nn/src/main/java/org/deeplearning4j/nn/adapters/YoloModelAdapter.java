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

package org.deeplearning4j.nn.adapters;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.val;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.ModelAdapter;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.List;

/**
 * This ModelAdapter implementation is suited for use of Yolo2 model with ParallelInference
 *
 * @author raver119@gmail.com
 */
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class YoloModelAdapter implements ModelAdapter<List<DetectedObject>> {
    @Builder.Default private int outputLayerIndex = 0;
    @Builder.Default private int outputIndex = 0;
    @Builder.Default private double detectionThreshold = 0.5;

    @Override
    public List<DetectedObject> apply(Model model, INDArray[] inputs, INDArray[] masks, INDArray[] labelsMasks) {
        if (model instanceof ComputationGraph) {
            val blindLayer = ((ComputationGraph) model).getOutputLayer(outputLayerIndex);
            if (blindLayer instanceof Yolo2OutputLayer) {
                val output = ((ComputationGraph) model).output(false, inputs, masks);
                return ((Yolo2OutputLayer) blindLayer).getPredictedObjects(output[outputIndex], detectionThreshold);
            } else {
                throw new ND4JIllegalStateException("Output layer with index [" + outputLayerIndex + "] is NOT Yolo2OutputLayer");
            }
        } else
            throw new ND4JIllegalStateException("Yolo2 model must be ComputationGraph");
    }

    @Override
    public List<DetectedObject> apply(INDArray... outputs) {
        throw new UnsupportedOperationException("Please use apply(Model, INDArray[], INDArray[]) signature");
    }
}
