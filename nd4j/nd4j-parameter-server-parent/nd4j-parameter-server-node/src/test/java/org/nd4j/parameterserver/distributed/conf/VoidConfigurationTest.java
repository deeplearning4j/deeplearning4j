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

package org.nd4j.parameterserver.distributed.conf;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class VoidConfigurationTest {

    @Rule
    public Timeout globalTimeout = Timeout.seconds(30);

    @Test
    public void testNetworkMask1() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.168.1.0/24");

        assertEquals("192.168.1.0/24", configuration.getNetworkMask());
    }


    @Test
    public void testNetworkMask2() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.168.1.12");

        assertEquals("192.168.1.0/24", configuration.getNetworkMask());
    }

    @Test
    public void testNetworkMask5() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.168.0.0/16");

        assertEquals("192.168.0.0/16", configuration.getNetworkMask());
    }

    @Test
    public void testNetworkMask6() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.168.0.0/8");

        assertEquals("192.168.0.0/8", configuration.getNetworkMask());
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testNetworkMask3() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.256.1.1/24");

        assertEquals("192.168.1.0/24", configuration.getNetworkMask());
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testNetworkMask4() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("0.0.0.0/8");

        assertEquals("192.168.1.0/24", configuration.getNetworkMask());
    }
}
