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

package org.eclipse.deeplearning4j.nd4j.autodiff.serialization;

import lombok.extern.slf4j.Slf4j;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Test for SameDiff flatbuffers serialization with models larger than 2GB.
 *
 * This test creates a large model that exceeds the 2GB limit to verify
 * that the 64-bit flatbuffers support works correctly.
 *
 * Note: This test is marked with @Ignore by default as it requires significant
 * memory resources and time to run.
 */
@Slf4j
public class LargeSameDiffSerializationTest extends BaseND4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Override
    public long getTimeoutMilliseconds() {
        return 30 * 60 * 1000L; // 30 minutes timeout
    }

    @Test
    public void testLargeModelSerialization() throws IOException {
        // Parameters to create a model larger than 2GB
        int numLayers = 10;
        int layerSize = 10000;
        Nd4j.getEnvironment().setCudaDeviceLimit(Environment.CUDA_LIMIT_MALLOC_HEAP_SIZE,9999999999L);
        File tempFile = testDir.newFile("large-samediff-model.bin");
        log.info("Will save model to: {}", tempFile.getAbsolutePath());

        // Create the model
        SameDiff sd = SameDiff.create();

        // Input
        long[] inputShape = new long[]{1, layerSize};
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, inputShape);
        SDVariable current = input;

        // Create a large network with many dense layers
        Random r = new Random(42);
        log.info("Creating large neural network...");

        for (int i = 0; i < numLayers; i++) {
            String layerName = "layer_" + i;
            log.info("Creating layer {}", i);

            // Create large weights
            SDVariable weights = sd.var(layerName + "_w", Nd4j.rand(DataType.FLOAT, layerSize, layerSize));
            SDVariable bias = sd.var(layerName + "_b", Nd4j.rand(DataType.FLOAT, 1, layerSize));

            // Dense layer
            current = sd.nn.relu(current.mmul(weights).add(bias),1.0);
        }

        // Output layer
        SDVariable outputWeights = sd.var("output_w", Nd4j.rand(DataType.FLOAT, layerSize, 10));
        SDVariable outputBias = sd.var("output_b", Nd4j.rand(DataType.FLOAT, 1, 10));
        SDVariable output = sd.nn.softmax(current.mmul(outputWeights).add(outputBias));
        sd.setOutputs(output.name());

        // Calculate and log estimated size
        long bytesPerElement = 4; // float32
        long totalElements = 0;
        for (int i = 0; i < numLayers; i++) {
            totalElements += layerSize * layerSize; // weights
            totalElements += layerSize;             // bias
        }
        totalElements += layerSize * 10 + 10;      // output layer

        long estimatedBytes = totalElements * bytesPerElement;
        double estimatedGB = estimatedBytes / (1024.0 * 1024.0 * 1024.0);
        log.info("Estimated model size: {:.2f} GB ({} elements)", estimatedGB, totalElements);

        // Make sure we're creating a model that's larger than 2GB
        assertTrue("Model should be larger than 2GB", estimatedBytes > 2L * 1024 * 1024 * 1024);

        // Save the model
        log.info("Saving model to disk...");
        long startTime = System.currentTimeMillis();
        sd.save(tempFile, true);
        long endTime = System.currentTimeMillis();
        log.info("Model saved in {:.2f} seconds", (endTime - startTime) / 1000.0);

        // Check file size
        long fileSizeBytes = tempFile.length();
        double fileSizeGB = fileSizeBytes / (1024.0 * 1024.0 * 1024.0);
        log.info("Actual file size: {:.2f} GB", fileSizeGB);

        // Verify file size is greater than 2GB
        assertTrue("Model file should be larger than 2GB", fileSizeBytes > 2L * 1024 * 1024 * 1024);

        // Try to load the model back
        log.info("Attempting to load model from disk...");
        startTime = System.currentTimeMillis();
        SameDiff loadedModel = SameDiff.load(tempFile, true);
        endTime = System.currentTimeMillis();
        log.info("Model loaded in {:.2f} seconds", (endTime - startTime) / 1000.0);

        // Verify the model loaded correctly
        assertEquals("Variable count mismatch", sd.variables().size(), loadedModel.variables().size());
        assertEquals("Op count mismatch", sd.ops().length, loadedModel.ops().length);

        log.info("Test completed successfully!");
    }
}