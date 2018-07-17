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

package org.nd4j.nativeblas;

import lombok.Getter;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * @author raver119@gmail.com
 * @author saudet
 */
public class NativeOpsHolder {
    private static Logger log = LoggerFactory.getLogger(NativeOpsHolder.class);
    private static final NativeOpsHolder INSTANCE = new NativeOpsHolder();

    @Getter
    private final NativeOps deviceNativeOps;

    private NativeOpsHolder() {
        try {
            Properties props = Nd4jContext.getInstance().getConf();

            String name = System.getProperty(Nd4j.NATIVE_OPS, props.get(Nd4j.NATIVE_OPS).toString());
            Class<? extends NativeOps> nativeOpsClazz = Class.forName(name).asSubclass(NativeOps.class);
            deviceNativeOps = nativeOpsClazz.newInstance();

            deviceNativeOps.initializeDevicesAndFunctions();
            int numThreads;
            String numThreadsString = System.getenv("OMP_NUM_THREADS");
            if (numThreadsString != null && !numThreadsString.isEmpty()) {
                numThreads = Integer.parseInt(numThreadsString);
                deviceNativeOps.setOmpNumThreads(numThreads);
            } else {
                int cores = Loader.totalCores();
                int chips = Loader.totalChips();
                if (chips > 0 && cores > 0) {
                    deviceNativeOps.setOmpNumThreads(Math.max(1, cores / chips));
                } else
                    deviceNativeOps.setOmpNumThreads(
                                    deviceNativeOps.getCores(Runtime.getRuntime().availableProcessors()));
            }
            //deviceNativeOps.setOmpNumThreads(4);

            log.info("Number of threads used for NativeOps: {}", deviceNativeOps.ompGetMaxThreads());
        } catch (Exception | Error e) {
            throw new RuntimeException(
                            "ND4J is probably missing dependencies. For more information, please refer to: http://nd4j.org/getstarted.html",
                            e);
        }
    }

    public static NativeOpsHolder getInstance() {
        return INSTANCE;
    }

}
