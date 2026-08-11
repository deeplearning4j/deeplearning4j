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

package org.eclipse.deeplearning4j.pipeline;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.config.ND4JInferenceWeightDataType;

import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * Interface for format-specific pipeline loaders.
 */
public interface PipelineLoader {

    boolean supports(ModelFormat format);
    ModelFormat getFormat();
    SameDiff loadModel(ModelManifest manifest, LoadConfig config) throws IOException;
    SameDiff loadModel(File file, LoadConfig config) throws IOException;
    Map<String, SameDiff> loadPipeline(ModelManifest manifest, LoadConfig config) throws IOException;

    default boolean canLoad(ModelManifest manifest) {
        return supports(manifest.getFormat());
    }

    default String getName() {
        return getFormat().getDescription() + " Loader";
    }

    default boolean convertsToSdz() {
        return true;
    }

    interface LoadConfig {
        String getDataType();
        String getDevice();
        boolean useMmap();
        boolean cacheConvertedModel();
        File getCacheDirectory();
        boolean convertToFloat32();
        boolean dequantize();

        static LoadConfig defaults() { return new DefaultLoadConfig(); }
        static LoadConfigBuilder builder() { return new LoadConfigBuilder(); }
    }

    class DefaultLoadConfig implements LoadConfig {
        @Override public String getDataType() {
            return ND4JInferenceWeightDataType.resolve().canonicalName();
        }
        @Override public String getDevice() { return "cpu"; }
        @Override public boolean useMmap() { return true; }
        @Override public boolean cacheConvertedModel() { return true; }
        @Override public File getCacheDirectory() {
            return new File(System.getProperty("user.home"), ".cache/samediff/models");
        }
        @Override public boolean convertToFloat32() {
            return ND4JInferenceWeightDataType.resolve() == ND4JInferenceWeightDataType.FLOAT32;
        }
        @Override public boolean dequantize() { return true; }
    }

    class LoadConfigBuilder {
        private String dataType;
        private String device = "cpu";
        private boolean useMmap = true;
        private boolean cacheConvertedModel = true;
        private File cacheDirectory;
        private boolean convertToFloat32;
        private boolean dequantize = true;

        public LoadConfigBuilder() {
            ND4JInferenceWeightDataType defaultType = ND4JInferenceWeightDataType.resolve();
            this.dataType = defaultType.canonicalName();
            this.convertToFloat32 = defaultType == ND4JInferenceWeightDataType.FLOAT32;
        }

        public LoadConfigBuilder dataType(String dataType) {
            ND4JInferenceWeightDataType parsed = ND4JInferenceWeightDataType.fromString(dataType);
            this.dataType = parsed.canonicalName();
            this.convertToFloat32 = parsed == ND4JInferenceWeightDataType.FLOAT32;
            return this;
        }
        public LoadConfigBuilder device(String device) { this.device = device; return this; }
        public LoadConfigBuilder useMmap(boolean useMmap) { this.useMmap = useMmap; return this; }
        public LoadConfigBuilder cacheConvertedModel(boolean cache) { this.cacheConvertedModel = cache; return this; }
        public LoadConfigBuilder cacheDirectory(File dir) { this.cacheDirectory = dir; return this; }
        public LoadConfigBuilder convertToFloat32(boolean convert) {
            this.convertToFloat32 = convert;
            if (convert) {
                this.dataType = ND4JInferenceWeightDataType.FLOAT32.canonicalName();
            } else if (ND4JInferenceWeightDataType.fromString(this.dataType)
                    == ND4JInferenceWeightDataType.FLOAT32) {
                this.dataType = ND4JInferenceWeightDataType.FLOAT16.canonicalName();
            }
            return this;
        }
        public LoadConfigBuilder dequantize(boolean dequantize) { this.dequantize = dequantize; return this; }

        public LoadConfig build() {
            final boolean finalConvertToFloat32 = convertToFloat32;
            final boolean finalDequantize = dequantize;
            return new LoadConfig() {
                @Override public String getDataType() { return dataType; }
                @Override public String getDevice() { return device; }
                @Override public boolean useMmap() { return useMmap; }
                @Override public boolean cacheConvertedModel() { return cacheConvertedModel; }
                @Override public File getCacheDirectory() {
                    return cacheDirectory != null ? cacheDirectory :
                            new File(System.getProperty("user.home"), ".cache/samediff/models");
                }
                @Override public boolean convertToFloat32() { return finalConvertToFloat32; }
                @Override public boolean dequantize() { return finalDequantize; }
            };
        }
    }
}
