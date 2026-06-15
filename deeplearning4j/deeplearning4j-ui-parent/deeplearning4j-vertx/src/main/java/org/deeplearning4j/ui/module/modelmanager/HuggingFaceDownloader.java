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

package org.deeplearning4j.ui.module.modelmanager;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;
import java.util.function.Consumer;

/**
 * Downloads model files from HuggingFace Hub.
 * Supports single-file downloads, multi-file downloads, and automatic
 * tokenizer artifact fetching (tokenizer.json, tokenizer_config.json, etc.).
 *
 * Borrows patterns from kompile-model-staging's HuggingFaceDownloader:
 * - Default tokenizer file set for LLM downloads
 * - Non-fatal fetch for tokenizer files (404 → skip)
 * - Progress callback with speed calculation
 */
@Slf4j
public class HuggingFaceDownloader {

    private static final String BASE_URL = "https://huggingface.co";
    private static final int CONNECT_TIMEOUT = 30_000;
    private static final int READ_TIMEOUT = 600_000;
    private static final int BUFFER_SIZE = 8192;

    /** Tokenizer files to auto-fetch for LLM models (best-effort, non-fatal) */
    private static final List<String> DEFAULT_TOKENIZER_FILES = Arrays.asList(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
    );

    private final String authToken;

    public HuggingFaceDownloader() {
        this(null);
    }

    public HuggingFaceDownloader(String authToken) {
        this.authToken = authToken;
    }

    @Data
    @AllArgsConstructor
    public static class DownloadProgress {
        private String fileName;
        private long bytesDownloaded;
        private long totalBytes;
        private double bytesPerSecond;

        public int getPercent() {
            if (totalBytes <= 0) return 0;
            return (int) Math.min(100, (bytesDownloaded * 100) / totalBytes);
        }
    }

    @Data
    @AllArgsConstructor
    public static class DownloadResult {
        private Path filePath;
        private long fileSize;
        private String checksum;
    }

    /**
     * Result of a multi-file download operation.
     */
    @Data
    public static class MultiDownloadResult {
        private final Map<String, Path> downloadedFiles = new LinkedHashMap<>();
        private long totalBytes;
        private String modelChecksum;

        public void addFile(String key, Path path) {
            downloadedFiles.put(key, path);
        }
    }

    /**
     * Download a single file from HuggingFace.
     */
    public DownloadResult download(String repo, String filePath, Path destDir,
                                   Consumer<DownloadProgress> progress) throws IOException {
        return download(repo, filePath, "main", destDir, progress);
    }

    public DownloadResult download(String repo, String filePath, String revision, Path destDir,
                                   Consumer<DownloadProgress> progress) throws IOException {
        Files.createDirectories(destDir);
        String url = buildDownloadUrl(repo, filePath, revision);
        String fileName = filePath.contains("/") ? filePath.substring(filePath.lastIndexOf('/') + 1) : filePath;
        Path destFile = destDir.resolve(fileName);

        long totalBytes = downloadFile(url, destFile, fileName, progress);
        String checksum = calculateSha256(destFile);

        return new DownloadResult(destFile, totalBytes, checksum);
    }

    /**
     * Download multiple files from a HuggingFace repo.
     * The "model" key file gets a SHA-256 checksum; other files are best-effort.
     *
     * @param repo     repository name (e.g. "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
     * @param files    key → repo-relative path map (e.g. "model" → "model-q4_k_m.gguf")
     * @param revision branch/tag (default "main")
     * @param destDir  local directory to save to
     * @param progress progress callback (may be null)
     * @return result with all downloaded file paths
     */
    public MultiDownloadResult downloadMultiple(String repo, Map<String, String> files,
                                                String revision, Path destDir,
                                                Consumer<DownloadProgress> progress) throws IOException {
        Files.createDirectories(destDir);
        MultiDownloadResult result = new MultiDownloadResult();
        long totalBytes = 0;

        for (Map.Entry<String, String> entry : files.entrySet()) {
            String key = entry.getKey();
            String filePath = entry.getValue();
            try {
                DownloadResult dr = download(repo, filePath, revision, destDir, progress);
                result.addFile(key, dr.getFilePath());
                totalBytes += dr.getFileSize();
                if ("model".equals(key)) {
                    result.setModelChecksum(dr.getChecksum());
                }
            } catch (IOException e) {
                // Model file is required; other files are best-effort
                if ("model".equals(key)) {
                    throw e;
                }
                log.warn("Failed to download optional file {} from {}/{}: {}",
                        filePath, repo, revision, e.getMessage());
            }
        }

        result.setTotalBytes(totalBytes);
        return result;
    }

    /**
     * Auto-fetch tokenizer files for a HuggingFace repo.
     * Downloads tokenizer.json, tokenizer_config.json, and special_tokens_map.json.
     * All files are best-effort — 404 or other errors are logged and skipped.
     *
     * @param repo     repository name
     * @param revision branch/tag (default "main")
     * @param destDir  local directory to save to
     * @return map of successfully downloaded files (key → local path)
     */
    public Map<String, Path> fetchTokenizerFiles(String repo, String revision, Path destDir) {
        Map<String, Path> fetched = new LinkedHashMap<>();
        String rev = (revision != null && !revision.isEmpty()) ? revision : "main";

        for (String fileName : DEFAULT_TOKENIZER_FILES) {
            try {
                String url = buildDownloadUrl(repo, fileName, rev);
                Path destFile = destDir.resolve(fileName);
                downloadFile(url, destFile, fileName, null);
                String key = fileName.replace(".json", "").replace("_", "");
                fetched.put(fileName, destFile);
                log.info("Fetched tokenizer file: {}/{} → {}", repo, fileName, destFile);
            } catch (IOException e) {
                log.debug("Tokenizer file not available: {}/{}/{} ({})", repo, rev, fileName, e.getMessage());
            }
        }

        return fetched;
    }

    /**
     * Check if a model repository is accessible.
     */
    public boolean isAvailable(String repo) {
        try {
            String url = BASE_URL + "/" + repo + "/resolve/main/config.json";
            HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
            conn.setRequestMethod("HEAD");
            conn.setConnectTimeout(CONNECT_TIMEOUT);
            conn.setReadTimeout(10_000);
            if (authToken != null) {
                conn.setRequestProperty("Authorization", "Bearer " + authToken);
            }
            int code = conn.getResponseCode();
            conn.disconnect();
            return code == 200;
        } catch (Exception e) {
            return false;
        }
    }

    String buildDownloadUrl(String repo, String filePath, String revision) {
        String rev = (revision != null && !revision.isEmpty()) ? revision : "main";
        return String.format("%s/%s/resolve/%s/%s", BASE_URL, repo, rev, filePath);
    }

    private long downloadFile(String urlStr, Path dest, String fileName,
                              Consumer<DownloadProgress> progress) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL(urlStr).openConnection();
        conn.setConnectTimeout(CONNECT_TIMEOUT);
        conn.setReadTimeout(READ_TIMEOUT);
        conn.setInstanceFollowRedirects(false);
        if (authToken != null) {
            conn.setRequestProperty("Authorization", "Bearer " + authToken);
        }

        int responseCode = conn.getResponseCode();

        // Handle redirects manually
        if (responseCode == 301 || responseCode == 302 || responseCode == 307 || responseCode == 308) {
            String location = conn.getHeaderField("Location");
            conn.disconnect();
            if (location != null) {
                return downloadFile(location, dest, fileName, progress);
            }
            throw new IOException("Redirect without Location header");
        }

        if (responseCode != 200) {
            conn.disconnect();
            throw new IOException("Download failed with HTTP " + responseCode + " for " + urlStr);
        }

        long contentLength = conn.getContentLengthLong();
        long totalRead = 0;
        long startTime = System.currentTimeMillis();
        long lastProgressTime = 0;

        try (InputStream in = conn.getInputStream();
             OutputStream out = new BufferedOutputStream(Files.newOutputStream(dest))) {
            byte[] buf = new byte[BUFFER_SIZE];
            int n;
            while ((n = in.read(buf)) != -1) {
                out.write(buf, 0, n);
                totalRead += n;

                if (progress != null) {
                    long now = System.currentTimeMillis();
                    if (now - lastProgressTime >= 500) {
                        lastProgressTime = now;
                        double elapsed = (now - startTime) / 1000.0;
                        double bps = elapsed > 0 ? totalRead / elapsed : 0;
                        progress.accept(new DownloadProgress(fileName, totalRead, contentLength, bps));
                    }
                }
            }
        } finally {
            conn.disconnect();
        }

        // Final progress update
        if (progress != null) {
            double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
            double bps = elapsed > 0 ? totalRead / elapsed : 0;
            progress.accept(new DownloadProgress(fileName, totalRead, contentLength, bps));
        }

        return totalRead;
    }

    static String calculateSha256(Path file) throws IOException {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream in = new BufferedInputStream(Files.newInputStream(file))) {
                byte[] buf = new byte[BUFFER_SIZE];
                int n;
                while ((n = in.read(buf)) != -1) {
                    md.update(buf, 0, n);
                }
            }
            byte[] hash = md.digest();
            StringBuilder sb = new StringBuilder("sha256:");
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IOException("SHA-256 not available", e);
        }
    }
}
