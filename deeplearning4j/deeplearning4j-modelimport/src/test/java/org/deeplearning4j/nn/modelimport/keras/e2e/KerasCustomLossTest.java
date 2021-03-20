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
package org.deeplearning4j.nn.modelimport.keras.e2e;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLossUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.SameDiffLoss;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Keras Custom Loss Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@Tag(TagNames.LOSS_FUNCTIONS)
class KerasCustomLossTest extends BaseDL4JTest {

    @TempDir
    public Path testDir;

    @DisplayName("Log Cosh")
    class LogCosh extends SameDiffLoss {

        @Override
        public SDVariable defineLoss(SameDiff sd, SDVariable layerInput, SDVariable labels) {
            return sd.math.log(sd.math.cosh(labels.sub(layerInput)));
        }
    }

    @Test
    @DisplayName("Test Sequential Lambda Layer Import")
    void testSequentialLambdaLayerImport() throws Exception {
        KerasLossUtils.registerCustomLoss("logcosh", new LogCosh());
        String modelPath = "modelimport/keras/examples/custom_loss.h5";
        try (InputStream is = Resources.asStream(modelPath)) {
            File modelFile = testDir.resolve("tempModel" + System.currentTimeMillis() + ".h5").toFile();
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            MultiLayerNetwork model = new KerasSequentialModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath()).enforceTrainingConfig(true).buildSequential().getMultiLayerNetwork();
            System.out.println(model.summary());
            INDArray input = Nd4j.create(new int[] { 10, 3 });
            model.output(input);
        } finally {
            KerasLossUtils.clearCustomLoss();
        }
    }
}
