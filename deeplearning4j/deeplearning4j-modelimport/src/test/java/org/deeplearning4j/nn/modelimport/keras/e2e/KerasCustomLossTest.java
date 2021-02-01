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
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.SameDiffLoss;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;


public class KerasCustomLossTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    public class LogCosh extends SameDiffLoss {
        @Override
        public SDVariable defineLoss(SameDiff sd, SDVariable layerInput, SDVariable labels) {
            return sd.math.log(sd.math.cosh(labels.sub(layerInput)));
        }
    }

    @Test
    public void testSequentialLambdaLayerImport() throws Exception {
        KerasLossUtils.registerCustomLoss("logcosh", new LogCosh());

        String modelPath = "modelimport/keras/examples/custom_loss.h5";

        try(InputStream is = Resources.asStream(modelPath)) {
            File modelFile = testDir.newFile("tempModel" + System.currentTimeMillis() + ".h5");
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            MultiLayerNetwork model = new KerasSequentialModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                    .enforceTrainingConfig(true).buildSequential().getMultiLayerNetwork();

            System.out.println(model.summary());
            INDArray input = Nd4j.create(new int[]{10, 3});

            model.output(input);
        } finally {
            KerasLossUtils.clearCustomLoss();
        }
    }


}
