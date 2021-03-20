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
package org.datavec.api.util;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.BaseND4JTest;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.AnyOf.anyOf;
import static org.hamcrest.core.IsEqual.equalTo;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.common.tests.tags.TagNames;

@DisplayName("Class Path Resource Test")
@Tag(TagNames.JAVA_ONLY)
@Tag(TagNames.FILE_IO)
class ClassPathResourceTest extends BaseND4JTest {

    // File sizes are reported slightly different on Linux vs. Windows
    private boolean isWindows = false;

    @BeforeEach
    void setUp() throws Exception {
        String osname = System.getProperty("os.name");
        if (osname != null && osname.toLowerCase().contains("win")) {
            isWindows = true;
        }
    }

    @Test
    @DisplayName("Test Get File 1")
    void testGetFile1() throws Exception {
        File intFile = new ClassPathResource("datavec-api/iris.dat").getFile();
        assertTrue(intFile.exists());
        if (isWindows) {
            assertThat(intFile.length(), anyOf(equalTo(2700L), equalTo(2850L)));
        } else {
            assertEquals(2700, intFile.length());
        }
    }

    @Test
    @DisplayName("Test Get File Slash 1")
    void testGetFileSlash1() throws Exception {
        File intFile = new ClassPathResource("datavec-api/iris.dat").getFile();
        assertTrue(intFile.exists());
        if (isWindows) {
            assertThat(intFile.length(), anyOf(equalTo(2700L), equalTo(2850L)));
        } else {
            assertEquals(2700, intFile.length());
        }
    }

    @Test
    @DisplayName("Test Get File With Space 1")
    void testGetFileWithSpace1() throws Exception {
        File intFile = new ClassPathResource("datavec-api/csvsequence test.txt").getFile();
        assertTrue(intFile.exists());
        if (isWindows) {
            assertThat(intFile.length(), anyOf(equalTo(60L), equalTo(64L)));
        } else {
            assertEquals(60, intFile.length());
        }
    }

    @Test
    @DisplayName("Test Input Stream")
    void testInputStream() throws Exception {
        ClassPathResource resource = new ClassPathResource("datavec-api/csvsequence_1.txt");
        File intFile = resource.getFile();
        if (isWindows) {
            assertThat(intFile.length(), anyOf(equalTo(60L), equalTo(64L)));
        } else {
            assertEquals(60, intFile.length());
        }
        InputStream stream = resource.getInputStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line = "";
        int cnt = 0;
        while ((line = reader.readLine()) != null) {
            cnt++;
        }
        assertEquals(5, cnt);
    }

    @Test
    @DisplayName("Test Input Stream Slash")
    void testInputStreamSlash() throws Exception {
        ClassPathResource resource = new ClassPathResource("datavec-api/csvsequence_1.txt");
        File intFile = resource.getFile();
        if (isWindows) {
            assertThat(intFile.length(), anyOf(equalTo(60L), equalTo(64L)));
        } else {
            assertEquals(60, intFile.length());
        }
        InputStream stream = resource.getInputStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line = "";
        int cnt = 0;
        while ((line = reader.readLine()) != null) {
            cnt++;
        }
        assertEquals(5, cnt);
    }
}
