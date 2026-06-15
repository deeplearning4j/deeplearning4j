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
package org.eclipse.deeplearning4j.omnihub;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.omnihub.OmnihubConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Downloads model files from HuggingFace Hub repositories.
 * Files are cached to {@code $OMNIHUB_HOME/huggingface/{modelId}/}.
 *
 * <p>Supports:
 * <ul>
 *   <li>HF Hub API for listing repo files</li>
 *   <li>{@code HF_TOKEN} env var for gated/private models</li>
 *   <li>Selective download with glob-style file patterns</li>
 *   <li>Caching — skips existing files unless forceDownload</li>
 * </ul>
 *
 * @author Adam Gibson
 */
public class HuggingFaceHubDownloader {

    private static final Logger log = LoggerFactory.getLogger(HuggingFaceHubDownloader.class);

    private static final String HF_API_BASE = "https://huggingface.co/api/models";
    private static final String HF_RESOLVE_BASE = "https://huggingface.co";

    /**
     * Downloads a HuggingFace repository to the local omnihub cache.
     *
     * @param huggingFaceId the model repo ID (e.g. "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
     * @return the local directory containing the downloaded files
     * @throws IOException if download fails
     */
    public static File downloadRepo(String huggingFaceId) throws IOException {
        return downloadRepo(huggingFaceId, null, false, "main");
    }

    /**
     * Downloads a HuggingFace repository to the local omnihub cache with selective file filtering.
     *
     * @param huggingFaceId the model repo ID
     * @param filePattern   glob-style pattern to filter files (e.g. "*.gguf", "*.safetensors"), or null for all
     * @param forceDownload if true, re-downloads existing files
     * @param revision      the git revision (branch/tag/commit), typically "main"
     * @return the local directory containing the downloaded files
     * @throws IOException if download fails
     */
    public static File downloadRepo(String huggingFaceId, String filePattern, boolean forceDownload, String revision) throws IOException {
        File destDir = new File(new File(OmnihubConfig.getOmnihubHome(), "huggingface"),
                huggingFaceId.replace("/", File.separator));
        if (!destDir.exists()) {
            destDir.mkdirs();
        }

        List<String> filePaths = listRepoFiles(huggingFaceId, revision);

        Pattern pattern = null;
        if (filePattern != null && !filePattern.isEmpty()) {
            String regex = globToRegex(filePattern);
            pattern = Pattern.compile(regex);
        }

        for (String filePath : filePaths) {
            if (pattern != null && !pattern.matcher(filePath).matches()) {
                continue;
            }

            File destFile = new File(destDir, filePath.replace("/", File.separator));
            if (!forceDownload && destFile.exists()) {
                log.debug("Skipping existing file: {}", filePath);
                continue;
            }

            File parentDir = destFile.getParentFile();
            if (!parentDir.exists()) {
                parentDir.mkdirs();
            }

            downloadFile(huggingFaceId, filePath, revision, destFile);
        }

        return destDir;
    }

    /**
     * Lists all files in a HuggingFace repo using the Hub API.
     */
    private static List<String> listRepoFiles(String huggingFaceId, String revision) throws IOException {
        String apiUrl = HF_API_BASE + "/" + huggingFaceId + "/tree/" + revision;
        List<String> files = new ArrayList<>();
        collectFiles(apiUrl, files);
        return files;
    }

    private static void collectFiles(String apiUrl, List<String> files) throws IOException {
        URL url = URI.create(apiUrl).toURL();
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        addAuthHeader(conn);
        conn.setRequestMethod("GET");
        conn.setRequestProperty("Accept", "application/json");

        int responseCode = conn.getResponseCode();
        if (responseCode != 200) {
            throw new IOException("HuggingFace API returned " + responseCode + " for " + apiUrl);
        }

        Gson gson = new Gson();
        JsonArray entries;
        try (InputStreamReader reader = new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8)) {
            entries = gson.fromJson(reader, JsonArray.class);
        } finally {
            conn.disconnect();
        }

        for (JsonElement element : entries) {
            JsonObject entry = element.getAsJsonObject();
            String type = entry.get("type").getAsString();
            String path = entry.get("path").getAsString();
            if ("file".equals(type)) {
                files.add(path);
            }
            // Directories are not recursed into via tree API — HF tree API returns flat file list
        }
    }

    /**
     * Downloads a single file from a HuggingFace repo.
     */
    private static void downloadFile(String huggingFaceId, String filePath, String revision, File destFile) throws IOException {
        String fileUrl = HF_RESOLVE_BASE + "/" + huggingFaceId + "/resolve/" + revision + "/" + filePath;
        log.info("Downloading {} -> {}", fileUrl, destFile.getAbsolutePath());

        URL url = URI.create(fileUrl).toURL();
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        addAuthHeader(conn);
        conn.setInstanceFollowRedirects(true);

        int responseCode = conn.getResponseCode();
        if (responseCode == 302 || responseCode == 301) {
            String redirectUrl = conn.getHeaderField("Location");
            conn.disconnect();
            url = URI.create(redirectUrl).toURL();
            conn = (HttpURLConnection) url.openConnection();
            conn.setInstanceFollowRedirects(true);
        } else if (responseCode != 200) {
            conn.disconnect();
            throw new IOException("Failed to download " + fileUrl + ": HTTP " + responseCode);
        }

        long contentLength = conn.getContentLengthLong();
        if (contentLength <= 0) {
            contentLength = 1; // avoid division by zero in progress bar
        }

        try (InputStream is = new ProgressInputStream(new BufferedInputStream(conn.getInputStream()), contentLength)) {
            FileUtils.copyInputStreamToFile(is, destFile);
        } finally {
            conn.disconnect();
        }
    }

    /**
     * Adds HF_TOKEN authorization header if the env var is set.
     */
    private static void addAuthHeader(HttpURLConnection conn) {
        String token = System.getenv("HF_TOKEN");
        if (token != null && !token.isEmpty()) {
            conn.setRequestProperty("Authorization", "Bearer " + token);
        }
    }

    /**
     * Converts a simple glob pattern (e.g. "*.gguf") to a regex.
     * Supports * and ? wildcards.
     */
    private static String globToRegex(String glob) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < glob.length(); i++) {
            char c = glob.charAt(i);
            switch (c) {
                case '*':
                    sb.append(".*");
                    break;
                case '?':
                    sb.append('.');
                    break;
                case '.':
                case '(':
                case ')':
                case '[':
                case ']':
                case '{':
                case '}':
                case '\\':
                case '^':
                case '$':
                case '|':
                case '+':
                    sb.append('\\').append(c);
                    break;
                default:
                    sb.append(c);
            }
        }
        return sb.toString();
    }
}
