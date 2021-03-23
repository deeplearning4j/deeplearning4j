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
package org.deeplearning4j.integration;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.integration.testcases.samediff.SameDiffCNNCases;
import org.deeplearning4j.integration.testcases.samediff.SameDiffMLPTestCases;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.nio.file.Path;

@Tag(TagNames.FILE_IO)
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class IntegrationTestsSameDiff extends BaseDL4JTest {

    @TempDir
    static Path testDir;

    @Override
    public long getTimeoutMilliseconds() {
        return 300_000L;
    }




    @Test
    public void testMLPMnist() throws Exception {
        IntegrationTestRunner.runTest(SameDiffMLPTestCases.getMLPMnist(), testDir);
    }

    @Test
    public void testMLPMoon() throws Exception {
        IntegrationTestRunner.runTest(SameDiffMLPTestCases.getMLPMoon(), testDir);
    }

    @Test
    public void testLenetMnist() throws Exception {
        IntegrationTestRunner.runTest(SameDiffCNNCases.getLenetMnist(), testDir);
    }

    @Test
    public void testCnn3dSynthetic() throws Exception {
        IntegrationTestRunner.runTest(SameDiffCNNCases.getCnn3dSynthetic(), testDir);
    }


}
