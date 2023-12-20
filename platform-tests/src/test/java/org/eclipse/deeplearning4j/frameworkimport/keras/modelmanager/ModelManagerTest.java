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
package org.eclipse.deeplearning4j.frameworkimport.keras.modelmanager;

import org.eclipse.deeplearning4j.modelimport.keras.ModelManager;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

public class ModelManagerTest {

    @Test
    public void testBidirectional() throws Exception {
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);
        ModelManager modelManager = new ModelManager("/home/agibsonccc/Documents/GitHub/deeplearning4j/contrib/keras-tests-reproducers/keras-reproducer-baselines/dl4j/baseline/bidirectional_tests");
        modelManager.loadModels();
        System.out.println("Running models");
        modelManager.runModels();
     /*   modelManager.runModels();
        modelManager.compareOutputs();
        modelManager.compareGradients();*/
    }

}
