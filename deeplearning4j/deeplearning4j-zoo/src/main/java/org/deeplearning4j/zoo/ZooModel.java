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

package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

/**
 * A zoo model is instantiable, returns information about itself, and can download
 * pretrained models if available.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public abstract class ZooModel<T> implements InstantiableModel {

    public boolean pretrainedAvailable(PretrainedType pretrainedType) {
        return pretrainedUrl(pretrainedType) != null;
    }

    /**
     * By default, will return a pretrained ImageNet if available.
     *
     * @return
     * @throws IOException
     */
    public Model initPretrained() throws IOException {
        return initPretrained(PretrainedType.IMAGENET);
    }

    /**
     * Returns a pretrained model for the given dataset, if available.
     *
     * @param pretrainedType
     * @return
     * @throws IOException
     */
    public <M extends Model> M initPretrained(PretrainedType pretrainedType) throws IOException {
        String remoteUrl = pretrainedUrl(pretrainedType);
        if (remoteUrl == null)
            throw new UnsupportedOperationException(
                            "Pretrained " + pretrainedType + " weights are not available for this model.");

        String localFilename = new File(remoteUrl).getName();

        File rootCacheDir = DL4JResources.getDirectory(ResourceType.ZOO_MODEL, modelName());
        File cachedFile = new File(rootCacheDir, localFilename);

        if (!cachedFile.exists()) {
            log.info("Downloading model to " + cachedFile.toString());
            FileUtils.copyURLToFile(new URL(remoteUrl), cachedFile);
        } else {
            log.info("Using cached model at " + cachedFile.toString());
        }

        long expectedChecksum = pretrainedChecksum(pretrainedType);
        if (expectedChecksum != 0L) {
            log.info("Verifying download...");
            Checksum adler = new Adler32();
            FileUtils.checksum(cachedFile, adler);
            long localChecksum = adler.getValue();
            log.info("Checksum local is " + localChecksum + ", expecting " + expectedChecksum);

            if (expectedChecksum != localChecksum) {
                log.error("Checksums do not match. Cleaning up files and failing...");
                cachedFile.delete();
                throw new IllegalStateException(
                                "Pretrained model file failed checksum. If this error persists, please open an issue at https://github.com/deeplearning4j/deeplearning4j.");
            }
        }

        if (modelType() == MultiLayerNetwork.class) {
            return (M) ModelSerializer.restoreMultiLayerNetwork(cachedFile);
        } else if (modelType() == ComputationGraph.class) {
            return (M) ModelSerializer.restoreComputationGraph(cachedFile);
        } else {
            throw new UnsupportedOperationException(
                            "Pretrained models are only supported for MultiLayerNetwork and ComputationGraph.");
        }
    }

    @Override
    public String modelName() {
        return getClass().getSimpleName().toLowerCase();
    }
}
