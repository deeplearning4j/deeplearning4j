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
package org.deeplearning4j.perf.listener;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.core.listener.HardwareMetric;
import org.deeplearning4j.core.listener.SystemPolling;
import org.junit.jupiter.api.Disabled;
import org.junit.Rule;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.ExtendWith;

@Disabled("AB 2019/05/24 - Failing on CI - \"Could not initialize class oshi.jna.platform.linux.Libc\" - Issue #7657")
@DisplayName("System Polling Test")
class SystemPollingTest extends BaseDL4JTest {

    @TempDir
    public Path tempDir;

    @Test
    @DisplayName("Test Polling")
    void testPolling() throws Exception {
        Nd4j.create(1);
        File tmpDir = tempDir.toFile();
        SystemPolling systemPolling = new SystemPolling.Builder().outputDirectory(tmpDir).pollEveryMillis(1000).build();
        systemPolling.run();
        Thread.sleep(8000);
        systemPolling.stopPolling();
        File[] files = tmpDir.listFiles();
        assertTrue(files != null && files.length > 0);
        // System.out.println(Arrays.toString(files));
        String yaml = FileUtils.readFileToString(files[0]);
        HardwareMetric fromYaml = HardwareMetric.fromYaml(yaml);
        System.out.println(fromYaml);
    }
}
