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

package org.deeplearning4j.perf.listener;

import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import oshi.json.SystemInfo;

import java.io.File;
import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Poll a system for its local statistics with a specified time.
 * The polling process will output a yaml file
 * in the specified output directory
 *
 * with all the related system diagnostics.
 *
 * @author Adam Gibson
 */
public class SystemPolling {

    private ScheduledExecutorService scheduledExecutorService;
    private long pollEveryMillis;
    private File outputDirectory;
    private NameProvider nameProvider;
    private ObjectMapper objectMapper = new ObjectMapper(new YAMLFactory());

    private SystemPolling(long pollEveryMillis,File outputDirectory,NameProvider nameProvider) {
        this.pollEveryMillis = pollEveryMillis;
        this.outputDirectory = outputDirectory;
        this.nameProvider = nameProvider;
    }


    /**
     * Run the polling in the background as a thread pool
     * running every {@link #pollEveryMillis} milliseconds
     */
    public void run() {
        scheduledExecutorService = Executors.newScheduledThreadPool(1);
        scheduledExecutorService.scheduleAtFixedRate(new Runnable() {
            @Override
            public void run() {
                SystemInfo systemInfo = new SystemInfo();
                HardwareMetric hardwareMetric = HardwareMetric.fromSystem(systemInfo,nameProvider.nextName());
                File hardwareFile = new File(outputDirectory,hardwareMetric.getName() + ".yml");
                try {
                    objectMapper.writeValue(hardwareFile,hardwareMetric);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        },0,pollEveryMillis, TimeUnit.MILLISECONDS);
    }


    /**
     * Shut down the background polling
     */
    public void stopPolling() {
        scheduledExecutorService.shutdownNow();
    }


    /**
     * The naming sequence provider.
     * This is for allowing generation of naming the output
     * according to some semantic sequence (such as a neural net epoch
     * or some time stamp)
     */
    public  interface NameProvider {
        String nextName();
    }

    public static class Builder {
        private long pollEveryMillis;
        private File outputDirectory;

        private NameProvider nameProvider = new NameProvider() {
            @Override
            public String nextName() {
                return UUID.randomUUID().toString();
            }
        };


        /**
         * The name provider for  this {@link SystemPolling}
         * the default value for this is a simple UUID
         * @param nameProvider the name provider to use
         * @return
         */
        public Builder nameProvider(NameProvider nameProvider) {
            this.nameProvider = nameProvider;
            return this;
        }


        /**
         * The interval in milliseconds in which to poll
         * the system for diagnostics
         * @param pollEveryMillis the interval in milliseconds
         * @return
         */
        public Builder pollEveryMillis(long pollEveryMillis) {
            this.pollEveryMillis = pollEveryMillis;
            return this;
        }

        /**
         * The output directory for the files
         * @param outputDirectory the output directory for the logs
         * @return
         */
        public Builder outputDirectory(File outputDirectory) {
            this.outputDirectory = outputDirectory;
            return this;
        }

        public SystemPolling build() {
            return new SystemPolling(pollEveryMillis,outputDirectory,nameProvider);
        }

    }

}
