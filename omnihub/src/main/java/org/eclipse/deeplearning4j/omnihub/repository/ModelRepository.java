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
package org.eclipse.deeplearning4j.omnihub.repository;

import java.io.File;
import java.io.IOException;

/**
 * Abstraction for a model repository backend.
 * Implementations provide download/caching for models from different sources
 * (GitHub raw files, HuggingFace Hub, etc.).
 *
 * @author Adam Gibson
 */
public interface ModelRepository {

    /**
     * Human-readable name for this repository (e.g. "github", "huggingface").
     */
    String name();

    /**
     * Priority for fallback ordering. Lower values are tried first.
     * Default repos: HuggingFace = 10, GitHub = 20.
     */
    int priority();

    /**
     * Whether this repository can potentially resolve a model by name and framework.
     * A return value of true does not guarantee the model exists — the actual
     * {@link #resolve} call may still throw IOException.
     */
    boolean canResolve(String modelName, String framework);

    /**
     * Download/cache a model by name and framework, returning the local file or directory.
     *
     * @param modelName     the model file name (e.g. "resnet18.fb")
     * @param framework     the framework identifier (e.g. "samediff", "dl4j")
     * @param forceDownload if true, re-download even if cached
     * @return the local file containing the downloaded model
     * @throws IOException if the model cannot be resolved from this repository
     */
    File resolve(String modelName, String framework, boolean forceDownload) throws IOException;

    /**
     * Whether this repository supports HuggingFace-style repo ID resolution.
     */
    boolean supportsHuggingFace();

    /**
     * Download/cache a HuggingFace model repository by repo ID.
     *
     * @param huggingFaceId the HuggingFace repo ID (e.g. "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
     * @param filePattern   glob pattern to filter files (e.g. "*.gguf"), or null for all
     * @param forceDownload if true, re-download even if cached
     * @return the local directory containing the downloaded model files
     * @throws IOException if download fails
     * @throws UnsupportedOperationException if this repository does not support HuggingFace resolution
     */
    File resolveHuggingFace(String huggingFaceId, String filePattern, boolean forceDownload) throws IOException;
}
