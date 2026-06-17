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

package org.eclipse.deeplearning4j.model.download;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Shared HTTP download infrastructure for model files.
 *
 * Handles downloading with progress bars, caching, redirect following,
 * and atomic temp-file writes. Used by both LLMModelDownloader and VLMModelDownloader.
 */
@Slf4j
public class ModelDownloader {

    /**
     * Download a file from a URL to a cache directory, skipping if already cached.
     *
     * @param url       the URL to download from
     * @param fileName  the file name to save as
     * @param cacheDir  the directory to cache in
     * @return download result with file path and metadata
     * @throws IOException if download fails
     */
    public static DownloadResult download(String url, String fileName, File cacheDir) throws IOException {
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }

        File outputFile = new File(cacheDir, fileName);
        boolean downloadedNow = false;
        long startTime = System.currentTimeMillis();

        if (!outputFile.exists()) {
            log.info("Downloading {} from {}", fileName, url);
            downloadFile(url, outputFile);
            downloadedNow = true;
            log.info("Downloaded {} to {}", fileName, outputFile.getAbsolutePath());
        } else {
            log.info("Using cached file: {}", outputFile.getAbsolutePath());
        }

        return DownloadResult.builder()
                .modelFile(outputFile)
                .downloadedNow(downloadedNow)
                .fileSizeBytes(outputFile.length())
                .downloadTimeMs(downloadedNow ? System.currentTimeMillis() - startTime : 0)
                .build();
    }

    /**
     * Check if a file is already cached.
     */
    public static boolean isCached(String fileName, File cacheDir) {
        return new File(cacheDir, fileName).exists();
    }

    /**
     * Resolve and create a cache directory from a system property or default.
     */
    public static File getCacheDir(String propertyName, String defaultDir) {
        String cacheDir = System.getProperty(propertyName, defaultDir);
        File dir = new File(cacheDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return dir;
    }

    /**
     * Clear all files in a cache directory.
     */
    public static void clearCache(File cacheDir) throws IOException {
        if (cacheDir.exists()) {
            File[] files = cacheDir.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) {
                        Files.delete(file.toPath());
                        log.info("Deleted cached file: {}", file.getName());
                    }
                }
            }
        }
    }

    /**
     * List cached files matching given extensions.
     */
    public static File[] listCachedFiles(File cacheDir, String... extensions) {
        if (cacheDir.exists()) {
            return cacheDir.listFiles((dir, name) -> {
                for (String ext : extensions) {
                    if (name.endsWith("." + ext)) return true;
                }
                return false;
            });
        }
        return new File[0];
    }

    // ==================== Internal Methods ====================

    private static void downloadFile(String urlString, File outputFile) throws IOException {
        outputFile.getParentFile().mkdirs();

        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.setConnectTimeout(30000);
        connection.setReadTimeout(60000);
        connection.setRequestProperty("User-Agent", "DL4J-ModelDownloader/1.0");

        // Authenticate with HuggingFace for gated models (e.g. google/gemma)
        String hfToken = System.getenv("HF_TOKEN");
        if (hfToken == null || hfToken.isEmpty()) {
            hfToken = System.getProperty("hf.token");
        }
        if (hfToken != null && !hfToken.isEmpty()) {
            connection.setRequestProperty("Authorization", "Bearer " + hfToken);
        }

        int responseCode = connection.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_MOVED_TEMP ||
                responseCode == HttpURLConnection.HTTP_MOVED_PERM ||
                responseCode == HttpURLConnection.HTTP_SEE_OTHER ||
                responseCode == 307 || responseCode == 308) {
            String newUrl = connection.getHeaderField("Location");
            log.debug("Redirecting to: {}", newUrl);
            connection.disconnect();
            downloadFile(newUrl, outputFile);
            return;
        }

        if (responseCode != HttpURLConnection.HTTP_OK) {
            throw new IOException("Failed to download: HTTP " + responseCode + " from " + urlString);
        }

        long contentLength = connection.getContentLengthLong();
        String sizeStr = contentLength > 0 ? formatBytes(contentLength) : "unknown size";
        log.info("Downloading {} ...", sizeStr);

        Path tempFile = Files.createTempFile("dl4j-download-", ".tmp");
        try (InputStream in = new BufferedInputStream(connection.getInputStream());
             OutputStream out = new BufferedOutputStream(Files.newOutputStream(tempFile))) {

            byte[] buffer = new byte[8192];
            long totalRead = 0;
            int bytesRead;
            long lastUpdateTime = System.currentTimeMillis();
            int lastPercent = -1;

            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
                totalRead += bytesRead;

                long now = System.currentTimeMillis();
                if (now - lastUpdateTime > 100) {
                    int percent = contentLength > 0 ? (int) ((totalRead * 100) / contentLength) : -1;
                    if (percent != lastPercent) {
                        printProgressBar(totalRead, contentLength, outputFile.getName());
                        lastPercent = percent;
                    }
                    lastUpdateTime = now;
                }
            }

            printProgressBar(totalRead, contentLength, outputFile.getName());
            log.debug("");
        }

        Files.move(tempFile, outputFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        connection.disconnect();
        log.info("Download complete: {}", outputFile.getName());
    }

    static void printProgressBar(long current, long total, String fileName) {
        int barWidth = 40;
        String downloadedStr = formatBytes(current);

        if (total > 0) {
            int percent = (int) ((current * 100) / total);
            int filled = (int) ((current * barWidth) / total);
            int empty = barWidth - filled;

            StringBuilder bar = new StringBuilder();
            bar.append(String.format("%-30s ", truncateFileName(fileName, 30)));
            bar.append("[");
            for (int i = 0; i < filled; i++) bar.append("=");
            if (filled < barWidth) bar.append(">");
            for (int i = 0; i < empty - 1; i++) bar.append(" ");
            bar.append("] ");
            bar.append(String.format("%3d%% ", percent));
            bar.append(String.format("%s / %s", downloadedStr, formatBytes(total)));

            log.debug("{}", bar);
        } else {
            char[] spinner = {'|', '/', '-', '\\'};
            int spinIdx = (int) ((current / 10000) % 4);

            StringBuilder bar = new StringBuilder();
            bar.append(String.format("%-30s ", truncateFileName(fileName, 30)));
            bar.append("[");
            bar.append(spinner[spinIdx]);
            bar.append("] ");
            bar.append(downloadedStr);

            log.debug("{}", bar);
        }
    }

    static String formatBytes(long bytes) {
        return org.nd4j.common.util.ND4JFileUtils.formatBytes(bytes);
    }

    static String truncateFileName(String fileName, int maxLen) {
        if (fileName.length() <= maxLen) {
            return fileName;
        }
        return "..." + fileName.substring(fileName.length() - maxLen + 3);
    }

    /**
     * Result of a model download operation.
     */
    @Data
    @Builder
    public static class DownloadResult {
        private final File modelFile;
        private final boolean downloadedNow;
        private final long fileSizeBytes;
        private final long downloadTimeMs;
    }
}
