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
package org.deeplearning4j.ui;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.core.ui.UiConnectionInfo;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

@DisplayName("Ui Connection Info Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
@Tag(TagNames.UI)
class UiConnectionInfoTest extends BaseDL4JTest {

    @BeforeEach
    void setUp() throws Exception {
    }

    @Test
    @DisplayName("Test Get First Part 1")
    void testGetFirstPart1() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setPort(8080).build();
        assertEquals(info.getFirstPart(), "http://localhost:8080");
    }

    @Test
    @DisplayName("Test Get First Part 2")
    void testGetFirstPart2() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().enableHttps(true).setPort(8080).build();
        assertEquals(info.getFirstPart(), "https://localhost:8080");
    }

    @Test
    @DisplayName("Test Get First Part 3")
    void testGetFirstPart3() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082).build();
        assertEquals(info.getFirstPart(), "https://192.168.1.1:8082");
    }

    @Test
    @DisplayName("Test Get Second Part 1")
    void testGetSecondPart1() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082).setPath("www-data").build();
        assertEquals(info.getSecondPart(), "/www-data/");
    }

    @Test
    @DisplayName("Test Get Second Part 2")
    void testGetSecondPart2() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082).setPath("/www-data/tmp/").build();
        assertEquals(info.getSecondPart(), "/www-data/tmp/");
    }

    @Test
    @DisplayName("Test Get Second Part 3")
    void testGetSecondPart3() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082).setPath("/www-data/tmp").build();
        assertEquals(info.getSecondPart(), "/www-data/tmp/");
    }

    @Test
    @DisplayName("Test Get Second Part 4")
    void testGetSecondPart4() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082).setPath("/www-data//tmp").build();
        assertEquals(info.getSecondPart(), "/www-data/tmp/");
    }

    @Test
    @DisplayName("Test Get Second Part 5")
    void testGetSecondPart5() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082).setPath("/www-data//tmp").build();
        assertEquals(info.getSecondPart("alpha"), "/www-data/tmp/alpha/");
    }

    @Test
    @DisplayName("Test Get Second Part 6")
    void testGetSecondPart6() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082).setPath("//www-data//tmp").build();
        assertEquals(info.getSecondPart("/alpha/"), "/www-data/tmp/alpha/");
    }

    @Test
    @DisplayName("Test Get Second Part 7")
    void testGetSecondPart7() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082).setPath("//www-data//tmp").build();
        assertEquals(info.getSecondPart("/alpha//beta/"), "/www-data/tmp/alpha/beta/");
    }

    @Test
    @DisplayName("Test Get Second Part 8")
    void testGetSecondPart8() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(false).setPort(8082).setPath("/www-data//tmp").build();
        assertEquals(info.getFullAddress(), "http://192.168.1.1:8082/www-data/tmp/");
    }
}
