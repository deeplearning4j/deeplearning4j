/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.integration;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.integration.testcases.samediff.SameDiffMLPTestCases;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class IntegrationTestsSameDiff extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 300_000L;
    }

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();


    @Test
    public void testMLPMnist() throws Exception {
        IntegrationTestRunner.runTest(SameDiffMLPTestCases.getMLPMnist(), testDir);
    }

}
