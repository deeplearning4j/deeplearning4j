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
package org.eclipse.deeplearning4j.frameworkimport.keras.e2e;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
public class KerasVGG16ImportTest extends BaseDL4JTest {

    private static final String MODEL_PATH = "/home/agibsonccc/Documents/GitHub/keras-vgg16-import-reproducer/VGG16Model.h5";

    // VGG16 ImageNet mean values in BGR order
    private static final double MEAN_B = 103.939;
    private static final double MEAN_G = 116.779;
    private static final double MEAN_R = 123.68;

    @Override
    public long getTimeoutMilliseconds() {
        return 300000L;
    }

    @Test
    public void testVGG16KerasImportForwardPass() throws Exception {
        File modelFile = new File(MODEL_PATH);
        if (!modelFile.exists()) {
            log.warn("VGG16 model not found at {}, skipping test", MODEL_PATH);
            return;
        }

        // Nd4j.getEnvironment().setDebug(true);
        // Nd4j.getEnvironment().setVerbose(true);

        log.info("Importing Keras VGG16 model from: {}", MODEL_PATH);
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(MODEL_PATH, false);
        // Do NOT call model.init() — it was already called during import, and calling it again
        // would reinitialize all weights, wiping out the imported Keras weights.
        model.getConfiguration().setInferenceWorkspaceMode(WorkspaceMode.NONE);
        model.getConfiguration().setTrainingWorkspaceMode(WorkspaceMode.NONE);

        log.info("Model summary:\n{}", model.summary());

        // Create a preprocessed NHWC input [1, 224, 224, 3]
        // Use random data - we just need to verify forward pass doesn't crash
        INDArray input = Nd4j.rand(new int[]{1, 224, 224, 3}).muli(255.0);

        // Manual VGG mean subtraction on NHWC data (BGR channel order)
        input.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.point(0)).subi(MEAN_B);
        input.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.point(1)).subi(MEAN_G);
        input.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.point(2)).subi(MEAN_R);

        log.info("Input shape: {}", Arrays.toString(input.shape()));

        log.info("Running feedForward to trace layer-by-layer shapes...");
        Map<String, INDArray> activations = model.feedForward(new INDArray[]{input}, false);
        for (Map.Entry<String, INDArray> entry : activations.entrySet()) {
            INDArray act = entry.getValue();
            log.info("Layer: {} shape={} order={} stride={}", entry.getKey(),
                    Arrays.toString(act.shape()), act.ordering(),
                    Arrays.toString(act.stride()));
        }

        log.info("Running forward pass...");
        INDArray output = model.outputSingle(input);

        log.info("Output shape: {}", Arrays.toString(output.shape()));
        log.info("Output argmax: {}", output.argMax(1));
        log.info("Output max: {}", output.maxNumber());

        assertNotNull(output);
        assertArrayEquals(new long[]{1, 1000}, output.shape());
    }
}
