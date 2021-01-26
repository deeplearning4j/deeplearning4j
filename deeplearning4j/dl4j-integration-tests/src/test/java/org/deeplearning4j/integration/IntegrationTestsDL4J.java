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

package org.deeplearning4j.integration;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.integration.testcases.dl4j.*;
import org.junit.AfterClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

//@Ignore("AB - 2019/05/27 - Integration tests need to be updated")
public class IntegrationTestsDL4J extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 300_000L;
    }

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @AfterClass
    public static void afterClass(){
        IntegrationTestRunner.printCoverageInformation();
    }

    // ***** MLPTestCases *****
    @Test
    public void testMLPMnist() throws Exception {
        IntegrationTestRunner.runTest(MLPTestCases.getMLPMnist(), testDir);
    }

    @Test
    public void testMlpMoon() throws Exception {
        IntegrationTestRunner.runTest(MLPTestCases.getMLPMoon(), testDir);
    }

    // ***** RNNTestCases *****
    @Test
    public void testRnnSeqClassification1() throws Exception {
        IntegrationTestRunner.runTest(RNNTestCases.getRnnCsvSequenceClassificationTestCase1(), testDir);
    }

    @Test
    public void testRnnSeqClassification2() throws Exception {
        IntegrationTestRunner.runTest(RNNTestCases.getRnnCsvSequenceClassificationTestCase2(), testDir);
    }

    @Test
    public void testRnnCharacter() throws Exception {
        IntegrationTestRunner.runTest(RNNTestCases.getRnnCharacterTestCase(), testDir);
    }


    // ***** CNN1DTestCases *****
    @Test
    public void testCnn1dCharacter() throws Exception {
        IntegrationTestRunner.runTest(CNN1DTestCases.getCnn1dTestCaseCharRNN(), testDir);
    }


    // ***** CNN2DTestCases *****
    @Test
    public void testLenetMnist() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.getLenetMnist(), testDir);
    }

    @Test
    public void testYoloHouseNumbers() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.getYoloHouseNumbers(), testDir);
    }

    @Test
    public void testCnn2DLenetTransferDropoutRepeatability() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.testLenetTransferDropoutRepeatability(), testDir);
    }


    // ***** CNN3DTestCases *****
    @Test
    public void testCnn3dSynthetic() throws Exception {
        IntegrationTestRunner.runTest(CNN3DTestCases.getCnn3dTestCaseSynthetic(), testDir);
    }

    // ***** UnsupervisedTestCases *****
    @Test
    public void testVAEMnistAnomaly() throws Exception {
        IntegrationTestRunner.runTest(UnsupervisedTestCases.getVAEMnistAnomaly(), testDir);
    }

    // ***** TransferLearningTestCases *****
    @Test
    public void testVgg16Transfer() throws Exception {
        IntegrationTestRunner.runTest(CNN2DTestCases.getVGG16TransferTinyImagenet(), testDir);
    }
}
