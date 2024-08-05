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

import lombok.SneakyThrows;

import java.io.File;
import java.util.*;

/**
 * This class is used to load keras h5 models for testing
 * with deeplearning4j.
 *
 * @author Adam Gibson
 */
public class KerasTestModelManager {
    protected String directoryPath;
    protected File dir;
    protected Map<String,ModelTestCase> testCases = new HashMap<>();

    protected Set<String> testsToRun = new HashSet<>();
    /**
     * Takes in the top level directory of the tests.
     * @param directoryPath
     */
    @SneakyThrows
    public KerasTestModelManager(String directoryPath, String...testsToRun) {
        this.directoryPath = directoryPath;
        dir = new File(directoryPath);
        this.testsToRun = new HashSet<>(Arrays.asList(testsToRun));
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IllegalArgumentException("Invalid directoryPath: " + directoryPath);
        }

        File[] testDirs = dir.listFiles(File::isDirectory);
        for(File testDir : testDirs) {
            if(!this.testsToRun.isEmpty() && !this.testsToRun.contains(testDir.getName()))
                continue;
            ModelTestCase modelTestCase = new ModelTestCase(testDir,testDir.getName());
            testCases.put(testDir.getName(),modelTestCase);

        }

    }


    public void loadModels() throws Exception {
        for (ModelTestCase modelTestCase : testCases.values()) {
            modelTestCase.loadModels();
        }
    }

    public void runModels() throws Exception {
        for (ModelTestCase modelTestCase : testCases.values()) {
            modelTestCase.runModels();
        }
    }



    public void compareOutputs() throws Exception {
        for (ModelTestCase modelTestCase : testCases.values()) {
            modelTestCase.compareOutputs();
        }
    }






}