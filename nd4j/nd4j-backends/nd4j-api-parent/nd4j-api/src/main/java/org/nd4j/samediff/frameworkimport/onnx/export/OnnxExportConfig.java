/*
 *  ******************************************************************************
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

package org.nd4j.samediff.frameworkimport.onnx.export;

/**
 * Configuration for ONNX export.
 * Stub implementation - TODO: implement full functionality.
 */
public class OnnxExportConfig {

    /**
     * Configuration for exporting with training state.
     */
    public static final OnnxExportConfig WITH_TRAINING = new OnnxExportConfig().setIncludeTrainingState(true);

    private int opsetVersion = 13;
    private String producerName = "nd4j";
    private String producerVersion = "1.0.0";
    private boolean includeTrainingState = false;

    public OnnxExportConfig() {
    }

    public int getOpsetVersion() {
        return opsetVersion;
    }

    public OnnxExportConfig setOpsetVersion(int opsetVersion) {
        this.opsetVersion = opsetVersion;
        return this;
    }

    public String getProducerName() {
        return producerName;
    }

    public OnnxExportConfig setProducerName(String producerName) {
        this.producerName = producerName;
        return this;
    }

    public String getProducerVersion() {
        return producerVersion;
    }

    public OnnxExportConfig setProducerVersion(String producerVersion) {
        this.producerVersion = producerVersion;
        return this;
    }

    public boolean isIncludeTrainingState() {
        return includeTrainingState;
    }

    public OnnxExportConfig setIncludeTrainingState(boolean includeTrainingState) {
        this.includeTrainingState = includeTrainingState;
        return this;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private OnnxExportConfig config = new OnnxExportConfig();

        public Builder opsetVersion(int opsetVersion) {
            config.setOpsetVersion(opsetVersion);
            return this;
        }

        public Builder producerName(String producerName) {
            config.setProducerName(producerName);
            return this;
        }

        public Builder producerVersion(String producerVersion) {
            config.setProducerVersion(producerVersion);
            return this;
        }

        public Builder includeTrainingState(boolean includeTrainingState) {
            config.setIncludeTrainingState(includeTrainingState);
            return this;
        }

        public OnnxExportConfig build() {
            return config;
        }
    }
}
