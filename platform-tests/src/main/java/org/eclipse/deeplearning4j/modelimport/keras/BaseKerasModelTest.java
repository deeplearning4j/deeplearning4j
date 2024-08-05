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
package org.eclipse.deeplearning4j.modelimport.keras;

import org.junit.Before;

import java.io.File;

public abstract class BaseKerasModelTest {
    //derived from the python contrib module. Note in order to run these tests you should ensure that
    //contrib/keras-tests-reproducers/keras-reproducer-baselines/dl4j/baseline/keras/save_models.py is ran first
     public final static String BASE_DIR = new File(System.getProperty("java.io.tmpdir"), "keras-dl4j-verification-models").getAbsolutePath();
    protected KerasTestModelManager kerasTestModelManager;

    @Before
    public void init() throws Exception {
        File baseDirCheck = new File(BASE_DIR);
        if(!baseDirCheck.exists()) {
            throw new IllegalStateException("Run save_models.py first to generate models. See comment above BASE_DIR for more information.");
        }
        kerasTestModelManager = new KerasTestModelManager(BASE_DIR,modelTestNames());
        kerasTestModelManager.loadModels();
    }

    public void runTests() throws Exception {
        System.out.println("Running models");
        kerasTestModelManager.runModels();
        kerasTestModelManager.compareOutputs();
    }


    public abstract String[] modelTestNames();
}
