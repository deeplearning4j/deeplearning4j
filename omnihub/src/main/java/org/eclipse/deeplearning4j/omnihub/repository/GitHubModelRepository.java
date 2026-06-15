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

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.omnihub.OmnihubConfig;
import org.eclipse.deeplearning4j.omnihub.ProgressInputStream;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;

/**
 * Model repository backed by a GitHub raw-file URL
 * (default: KonduitAI/omnihub-zoo on GitHub).
 *
 * @author Adam Gibson
 */
public class GitHubModelRepository implements ModelRepository {

    private final String baseUrl;
    private final int priority;

    public GitHubModelRepository() {
        this(OmnihubConfig.getOmnihubUrl(), 20);
    }

    public GitHubModelRepository(String baseUrl, int priority) {
        this.baseUrl = baseUrl;
        this.priority = priority;
    }

    @Override
    public String name() {
        return "github";
    }

    @Override
    public int priority() {
        return priority;
    }

    @Override
    public boolean canResolve(String modelName, String framework) {
        return true;
    }

    @Override
    public File resolve(String modelName, String framework, boolean forceDownload) throws IOException {
        File destination = new File(OmnihubConfig.getOmnihubHome(), framework);
        File destFile = new File(destination, modelName);
        if (forceDownload && destFile.exists()) {
            destFile.delete();
        }
        if (!destFile.exists()) {
            destination.mkdirs();
            String url = baseUrl + "/" + framework + "/" + modelName;
            URL remoteUrl = URI.create(url).toURL();
            long size = getFileSize(remoteUrl);
            try (InputStream is = new ProgressInputStream(new BufferedInputStream(remoteUrl.openStream()), size)) {
                FileUtils.copyInputStreamToFile(is, destFile);
            }
        }
        return destFile;
    }

    @Override
    public boolean supportsHuggingFace() {
        return false;
    }

    @Override
    public File resolveHuggingFace(String huggingFaceId, String filePattern, boolean forceDownload) throws IOException {
        throw new UnsupportedOperationException("GitHub repository does not support HuggingFace resolution");
    }

    private static long getFileSize(URL url) throws IOException {
        URLConnection conn = null;
        try {
            conn = url.openConnection();
            if (conn instanceof HttpURLConnection) {
                ((HttpURLConnection) conn).setRequestMethod("HEAD");
            }
            conn.getInputStream();
            return conn.getContentLength();
        } finally {
            if (conn instanceof HttpURLConnection) {
                ((HttpURLConnection) conn).disconnect();
            }
        }
    }
}
