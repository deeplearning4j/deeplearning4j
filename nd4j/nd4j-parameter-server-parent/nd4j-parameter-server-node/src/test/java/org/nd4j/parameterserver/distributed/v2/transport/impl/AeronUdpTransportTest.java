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

package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import static org.junit.Assert.*;

@Slf4j
public class AeronUdpTransportTest extends BaseND4JTest {
    private static final String IP = "127.0.0.1";
    private static final int ROOT_PORT = 40781;

    @Override
    public long getTimeoutMilliseconds() {
        return 240_000L;
    }

    @Test
    @Ignore
    public void testBasic_Connection_1() throws Exception {
        // we definitely want to shutdown all transports after test, to avoid issues with shmem
        try(val transportA = new AeronUdpTransport(IP, ROOT_PORT, IP, ROOT_PORT, VoidConfiguration.builder().build());
            val transportB = new AeronUdpTransport(IP, 40782, IP, ROOT_PORT, VoidConfiguration.builder().build())) {
            transportA.launchAsMaster();

            Thread.sleep(50);

            transportB.launch();

            Thread.sleep(50);

            assertEquals(2, transportA.getMesh().totalNodes());
            assertEquals(transportA.getMesh(), transportB.getMesh());
        }
    }
}