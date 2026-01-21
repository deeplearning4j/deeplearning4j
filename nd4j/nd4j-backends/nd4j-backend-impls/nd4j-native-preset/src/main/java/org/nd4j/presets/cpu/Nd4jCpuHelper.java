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

package org.nd4j.presets.cpu;

import org.nd4j.nativeblas.NativeOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class Nd4jCpuHelper extends Nd4jCpuPresets implements NativeOps {
    private static final Logger log = LoggerFactory.getLogger(Nd4jCpuHelper.class);

    static {
        String openblasThreads = System.getenv("OPENBLAS_NUM_THREADS");
        String sdOpenblasThreads = System.getenv("SD_OPENBLAS_THREADS");

        if (openblasThreads == null && sdOpenblasThreads == null) {
            // Neither environment variable is set - default to single-threaded
            // We can't actually set environment variables in Java, but we can
            // call openblas_set_num_threads(1) after the library loads.
            // For now, log a warning if the user hasn't set these.
            log.debug("OpenBLAS threading: No OPENBLAS_NUM_THREADS or SD_OPENBLAS_THREADS set. " +
                    "OpenBLAS will be configured to single-threaded mode after loading to prevent TLS corruption. " +
                    "Set OPENBLAS_NUM_THREADS=N to enable multi-threaded OpenBLAS.");
        }
    }
}
