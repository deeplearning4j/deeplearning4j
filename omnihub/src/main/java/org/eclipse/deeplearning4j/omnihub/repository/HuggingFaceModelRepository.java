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

import org.deeplearning4j.omnihub.OmnihubConfig;
import org.eclipse.deeplearning4j.omnihub.HuggingFaceHubDownloader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;

/**
 * Model repository backed by HuggingFace Hub.
 * Can resolve named models from a configured HF organization (default: KompileAI)
 * and also supports direct HuggingFace repo ID resolution.
 *
 * @author Adam Gibson
 */
public class HuggingFaceModelRepository implements ModelRepository {

    private static final Logger log = LoggerFactory.getLogger(HuggingFaceModelRepository.class);
    private static final String HF_API_BASE = "https://huggingface.co/api/models";

    private final String org;
    private final int priority;

    public HuggingFaceModelRepository() {
        this(OmnihubConfig.getHuggingFaceOrg(), 10);
    }

    public HuggingFaceModelRepository(String org, int priority) {
        this.org = org;
        this.priority = priority;
    }

    @Override
    public String name() {
        return "huggingface";
    }

    @Override
    public int priority() {
        return priority;
    }

    @Override
    public boolean canResolve(String modelName, String framework) {
        String repoId = org + "/" + stripExtension(modelName);
        try {
            return repoExists(repoId);
        } catch (IOException e) {
            log.debug("Failed to check HuggingFace repo existence for {}: {}", repoId, e.getMessage());
            return false;
        }
    }

    @Override
    public File resolve(String modelName, String framework, boolean forceDownload) throws IOException {
        String repoId = org + "/" + stripExtension(modelName);
        return HuggingFaceHubDownloader.downloadRepo(repoId, null, forceDownload, "main");
    }

    @Override
    public boolean supportsHuggingFace() {
        return true;
    }

    @Override
    public File resolveHuggingFace(String huggingFaceId, String filePattern, boolean forceDownload) throws IOException {
        return HuggingFaceHubDownloader.downloadRepo(huggingFaceId, filePattern, forceDownload, "main");
    }

    private boolean repoExists(String repoId) throws IOException {
        URL url = URI.create(HF_API_BASE + "/" + repoId).toURL();
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        try {
            String token = System.getenv("HF_TOKEN");
            if (token != null && !token.isEmpty()) {
                conn.setRequestProperty("Authorization", "Bearer " + token);
            }
            conn.setRequestMethod("HEAD");
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);
            return conn.getResponseCode() == 200;
        } finally {
            conn.disconnect();
        }
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }
}
