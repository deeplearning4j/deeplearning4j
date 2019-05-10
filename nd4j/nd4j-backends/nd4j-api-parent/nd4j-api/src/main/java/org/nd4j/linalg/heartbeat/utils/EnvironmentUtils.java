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

package org.nd4j.linalg.heartbeat.utils;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.reports.Environment;

import java.net.NetworkInterface;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * @author raver119@gmail.com
 */
public class EnvironmentUtils {

    /**
     * This method build
     * @return
     */
    public static Environment buildEnvironment() {
        Environment environment = new Environment();

        environment.setJavaVersion(System.getProperty("java.specification.version"));
        environment.setNumCores(Runtime.getRuntime().availableProcessors());
        environment.setAvailableMemory(Runtime.getRuntime().maxMemory());
        environment.setOsArch(System.getProperty("os.arch"));
        environment.setOsName(System.getProperty("os.opName"));
        environment.setBackendUsed(Nd4j.getExecutioner().getClass().getSimpleName());

        return environment;
    }

    public static long buildCId() {
        /*
            builds repeatable anonymous value
        */
        long ret = 0;

        try {
            List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());

            for (NetworkInterface networkInterface : interfaces) {
                try {
                    byte[] arr = networkInterface.getHardwareAddress();
                    long seed = 0;
                    for (int i = 0; i < arr.length; i++) {
                        seed += ((long) arr[i] & 0xffL) << (8 * i);
                    }
                    Random random = new Random(seed);

                    return random.nextLong();
                } catch (Exception e) {
                    ; // do nothing, just skip to next interface
                }
            }

        } catch (Exception e) {
            ; // do nothing here
        }

        return ret;
    }
}
