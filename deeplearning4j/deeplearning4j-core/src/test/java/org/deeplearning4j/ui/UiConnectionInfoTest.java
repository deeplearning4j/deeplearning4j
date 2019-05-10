/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.ui;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class UiConnectionInfoTest extends BaseDL4JTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testGetFirstPart1() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setPort(8080).build();

        assertEquals("http://localhost:8080", info.getFirstPart());
    }

    @Test
    public void testGetFirstPart2() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().enableHttps(true).setPort(8080).build();

        assertEquals("https://localhost:8080", info.getFirstPart());
    }

    @Test
    public void testGetFirstPart3() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082)
                        .build();

        assertEquals("https://192.168.1.1:8082", info.getFirstPart());
    }


    @Test
    public void testGetSecondPart1() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082)
                        .setPath("www-data").build();

        assertEquals("/www-data/", info.getSecondPart());
    }

    @Test
    public void testGetSecondPart2() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082)
                        .setPath("/www-data/tmp/").build();

        assertEquals("/www-data/tmp/", info.getSecondPart());
    }

    @Test
    public void testGetSecondPart3() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082)
                        .setPath("/www-data/tmp").build();

        assertEquals("/www-data/tmp/", info.getSecondPart());
    }

    @Test
    public void testGetSecondPart4() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082)
                        .setPath("/www-data//tmp").build();

        assertEquals("/www-data/tmp/", info.getSecondPart());
    }

    @Test
    public void testGetSecondPart5() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082)
                        .setPath("/www-data//tmp").build();

        assertEquals("/www-data/tmp/alpha/", info.getSecondPart("alpha"));
    }

    @Test
    public void testGetSecondPart6() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082)
                        .setPath("//www-data//tmp").build();

        assertEquals("/www-data/tmp/alpha/", info.getSecondPart("/alpha/"));
    }

    @Test
    public void testGetSecondPart7() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(true).setPort(8082)
                        .setPath("//www-data//tmp").build();

        assertEquals("/www-data/tmp/alpha/beta/", info.getSecondPart("/alpha//beta/"));
    }

    @Test
    public void testGetSecondPart8() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder().setAddress("192.168.1.1").enableHttps(false)
                        .setPort(8082).setPath("/www-data//tmp").build();

        assertEquals("http://192.168.1.1:8082/www-data/tmp/", info.getFullAddress());
    }
}
