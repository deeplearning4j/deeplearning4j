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
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.util.ND4JFileUtils;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.ProtocolException;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Shared HTTP download infrastructure for model files.
 *
 * Handles verified streaming downloads, resumable destination-local partial files,
 * caching and bounded redirects. Used by both LLMModelDownloader and VLMModelDownloader.
 */
@Slf4j
public class ModelDownloader {

    private static final int MAX_ATTEMPTS = 4;
    private static final int MAX_REDIRECTS = 10;
    private static final Pattern CONTENT_RANGE = Pattern.compile("bytes ([0-9]+)-([0-9]+)/([0-9]+)");
    // Bounded lock storage; canonical destinations also share an OS lock across JVMs.
    private static final ReentrantLock[] DOWNLOAD_LOCKS = new ReentrantLock[64];
    static {
        for (int i = 0; i < DOWNLOAD_LOCKS.length; i++) {
            DOWNLOAD_LOCKS[i] = new ReentrantLock();
        }
    }

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
        File outputFile = new File(cacheDir, fileName).getCanonicalFile();
        ReentrantLock lock = DOWNLOAD_LOCKS[(outputFile.hashCode() & Integer.MAX_VALUE) % DOWNLOAD_LOCKS.length];
        try {
            lock.lockInterruptibly();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new InterruptedIOException("Interrupted waiting for download lock");
        }
        try {
            // Existing caches remain usable even in read-only directories, without new sidecars.
            if (outputFile.exists()) {
                return downloadLocked(url, outputFile);
            }
            Files.createDirectories(outputFile.toPath().getParent());
            // Never unlink the lock file: a waiter may already have its inode open.
            try (FileChannel channel = FileChannel.open(sidecar(outputFile.toPath(), ".download.lock"),
                    StandardOpenOption.CREATE, StandardOpenOption.WRITE);
                 FileLock ignored = channel.lock()) {
                return downloadLocked(url, outputFile);
            }
        } finally {
            lock.unlock();
        }
    }

    private static DownloadResult downloadLocked(String url, File outputFile) throws IOException {
        boolean downloadedNow = false;
        long startTime = System.currentTimeMillis();

        if (!outputFile.exists()) {
            log.info("Downloading {}", outputFile.getName());
            downloadFile(url, outputFile);
            downloadedNow = true;
            log.info("Downloaded {} to {}", outputFile.getName(), outputFile.getAbsolutePath());
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

    /**
     * Sidecar contract (all siblings of the destination): .part contains a contiguous prefix;
     * .part.properties stores version=1, source (exact original URL), etag (strong or empty),
     * and length (expected complete byte count). .part.properties.tmp is an atomic-write staging
     * file. Redirect targets and HF bearer credentials are never persisted. On successful promotion
     * the partial/metadata are removed; .download.lock remains to keep cross-JVM locking safe.
     * Metadata is also invalidated if an untrusted response cannot be rolled back to its original
     * prefix. An existing completed destination retains the legacy cache behavior.
     * A prefix without a matching source, strong validator and valid length is never appended to.
     */
    private static void downloadFile(String urlString, File outputFile) throws IOException {
        Path target = outputFile.toPath();
        Path partial = sidecar(target, ".part");
        Path metadata = sidecar(target, ".part.properties");
        URL original = new URL(urlString);
        long[] lastProgress = {System.nanoTime()};
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            checkInterrupted();
            ResumeState state = readState(metadata, urlString);
            long offset = Files.exists(partial) ? Files.size(partial) : 0;
            if (state == null || !strongEtag(state.etag) || offset >= state.length) {
                // Even a full-length prefix is not proof that the previous response finished cleanly.
                offset = 0;
            }
            try {
                transfer(original, urlString, target, partial, metadata, state, offset, lastProgress);
                checkInterrupted();
                moveIntoPlace(partial, target);
                Files.deleteIfExists(metadata);
                Files.deleteIfExists(sidecar(metadata, ".tmp"));
                log.info("Download complete: {} ({} bytes)", outputFile.getName(), Files.size(target));
                return;
            } catch (TransientDownloadException e) {
                // A failed output flush/close may be suppressed behind the network failure.
                // Do not retry local I/O errors or trust bytes after an unsuccessful flush.
                if (e.getSuppressed().length > 0 || attempt == MAX_ATTEMPTS
                        || Thread.currentThread().isInterrupted()) {
                    throw e;
                }
                log.warn("Download {} interrupted; retry {}/{} from original URL, preserving partial bytes",
                        outputFile.getName(), attempt + 1, MAX_ATTEMPTS);
                try {
                    Thread.sleep(1000L << (attempt - 1));
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new InterruptedIOException("Interrupted during download retry");
                }
            }
        }
    }

    private static void transfer(URL original, String source, Path target, Path partial, Path metadata,
                                 ResumeState state, long offset, long[] lastProgress) throws IOException {
        HttpURLConnection connection = openFollowingRedirects(original, offset, state);
        try {
            int status = connection.getResponseCode();
            long length = contentLength(connection);
            String encoding = connection.getHeaderField("Content-Encoding");
            if (encoding != null && !"identity".equalsIgnoreCase(encoding)) {
                throw new IOException("Encoded download response cannot be byte-verified");
            }
            String etag = connection.getHeaderField("ETag");
            long expected;
            if (status == HttpURLConnection.HTTP_PARTIAL) {
                if (offset == 0 || state == null || !state.etag.equals(etag)) {
                    throw new IOException("Unexpected partial response or changed/missing ETag");
                }
                String range = connection.getHeaderField("Content-Range");
                Matcher matcher = CONTENT_RANGE.matcher(range == null ? "" : range);
                if (!matcher.matches()) {
                    throw new IOException("Invalid Content-Range");
                }
                long start = parseLength(matcher.group(1));
                long end = parseLength(matcher.group(2));
                expected = parseLength(matcher.group(3));
                if (start != offset || expected != state.length || end != expected - 1 || end < start
                        || (length >= 0 && length != expected - offset)) {
                    throw new IOException("Content-Range/Content-Length does not match requested remainder");
                }
                log.info("Resuming {} at {} / {} bytes", target.getFileName(), offset, expected);
            } else if (status == HttpURLConnection.HTTP_OK) {
                if (connection.getHeaderField("Content-Range") != null || length < 0) {
                    throw new IOException("Full response requires Content-Length and no Content-Range");
                }
                expected = length;
                if (Files.exists(partial) && Files.size(partial) > 0) {
                    log.warn("Restarting {} from zero: {}", target.getFileName(), offset > 0
                            ? "server ignored Range or If-Range validator changed"
                            : "partial has no matching resumable metadata");
                }
                offset = 0;
                // Truncate BEFORE publishing new metadata: a crash must not label old bytes as new.
                try (OutputStream ignored = Files.newOutputStream(partial)) {
                    // Create/truncate only after validating the full response headers.
                }
                writeState(metadata, source, strongEtag(etag) ? etag : "", expected);
            } else {
                // Includes 416: never promote bytes merely because a server rejects the range.
                throw new IOException("Unexpected download HTTP status " + status);
            }

            InputStream body;
            try {
                body = connection.getInputStream();
            } catch (IOException e) {
                throw networkFailure(e);
            }
            try (InputStream in = new BufferedInputStream(body);
                 OutputStream out = new BufferedOutputStream(Files.newOutputStream(partial,
                         StandardOpenOption.WRITE, StandardOpenOption.APPEND))) {
                byte[] buffer = new byte[64 * 1024];
                long received = offset;
                while (true) {
                    checkInterrupted();
                    int count;
                    try {
                        count = in.read(buffer);
                    } catch (IOException e) {
                        throw networkFailure(e);
                    }
                    if (count == -1) {
                        if (received != expected) {
                            throw new TransientDownloadException("Truncated response: received " + received
                                    + " of " + expected + " bytes", null);
                        }
                        break;
                    }
                    if (count > expected - received) {
                        throw new ProtocolException("Response exceeds expected byte count");
                    }
                    out.write(buffer, 0, count);
                    received += count;
                    long now = System.nanoTime();
                    if (now - lastProgress[0] >= 30_000_000_000L) {
                        log.info("Downloading {}: {} / {} bytes", target.getFileName(), received, expected);
                        lastProgress[0] = now;
                    }
                }
            } catch (ProtocolException invalidBody) {
                // A malformed body cannot extend the trusted prefix, even if some chunks fitted.
                try (FileChannel file = FileChannel.open(partial, StandardOpenOption.WRITE)) {
                    file.truncate(offset);
                } catch (IOException rollbackFailure) {
                    invalidBody.addSuppressed(rollbackFailure);
                    // Prevent resuming bytes whose rollback failed.
                    Files.deleteIfExists(metadata);
                }
                throw invalidBody;
            }
            if (Files.size(partial) != expected) {
                throw new IOException("Partial file size differs from verified response length");
            }
        } finally {
            connection.disconnect();
        }
    }

    private static HttpURLConnection openFollowingRedirects(URL original, long offset, ResumeState state)
            throws IOException {
        URL current = original;
        Set<String> visited = new HashSet<>();
        for (int redirects = 0; redirects <= MAX_REDIRECTS; redirects++) {
            checkInterrupted();
            if (!("https".equalsIgnoreCase(current.getProtocol()) || "http".equalsIgnoreCase(current.getProtocol()))
                    || current.getUserInfo() != null || !visited.add(current.toExternalForm())) {
                throw new IOException("Unsupported URL or redirect loop");
            }
            HttpURLConnection connection = (HttpURLConnection) current.openConnection();
            boolean keep = false;
            try {
                connection.setInstanceFollowRedirects(false);
                connection.setConnectTimeout(30000);
                connection.setReadTimeout(60000);
                connection.setUseCaches(false);
                connection.setRequestProperty("User-Agent", "DL4J-ModelDownloader/1.0");
                connection.setRequestProperty("Accept-Encoding", "identity");
                if (offset > 0) {
                    connection.setRequestProperty("Range", "bytes=" + offset + "-");
                    connection.setRequestProperty("If-Range", state.etag);
                }
                // HF credentials belong ONLY to the HTTPS Hub origin, never CDN/signed URLs.
                if (isHuggingFaceOrigin(original) && isHuggingFaceOrigin(current)) {
                    String token = System.getenv("HF_TOKEN");
                    if (token == null || token.isEmpty()) {
                        token = System.getProperty(ND4JSystemProperties.HF_TOKEN);
                    }
                    if (token != null && !token.isEmpty()) {
                        connection.setRequestProperty("Authorization", "Bearer " + token);
                    }
                }
                int status;
                try {
                    status = connection.getResponseCode();
                } catch (IOException e) {
                    throw networkFailure(e);
                }
                if (status == 301 || status == 302 || status == 303 || status == 307 || status == 308) {
                    String location = connection.getHeaderField("Location");
                    if (location == null || location.trim().isEmpty() || redirects == MAX_REDIRECTS) {
                        throw new IOException("Missing redirect Location or redirect limit exceeded");
                    }
                    URL next = new URL(current, location);
                    if ("https".equalsIgnoreCase(current.getProtocol())
                            && !"https".equalsIgnoreCase(next.getProtocol())) {
                        throw new IOException("Refusing HTTPS downgrade redirect");
                    }
                    current = next;
                    continue;
                }
                if (status == 408 || status == 429 || status == 500 || status == 502 || status == 503
                        || status == 504 || (redirects > 0 && (status == 401 || status == 403))) {
                    // A signed redirect can expire. Re-resolve the ORIGINAL (possibly pinned) URL.
                    throw new TransientDownloadException("Transient download HTTP status " + status, null);
                }
                if (status != HttpURLConnection.HTTP_OK && status != HttpURLConnection.HTTP_PARTIAL) {
                    throw new IOException("Download failed with HTTP status " + status);
                }
                keep = true;
                return connection;
            } finally {
                if (!keep) {
                    connection.disconnect();
                }
            }
        }
        throw new IOException("Redirect limit exceeded");
    }

    private static boolean isHuggingFaceOrigin(URL url) {
        return "https".equalsIgnoreCase(url.getProtocol()) && "huggingface.co".equalsIgnoreCase(url.getHost())
                && (url.getPort() == -1 || url.getPort() == 443) && url.getUserInfo() == null;
    }

    private static IOException networkFailure(IOException e) {
        if (e instanceof SocketTimeoutException || e instanceof SocketException || e instanceof EOFException) {
            return new TransientDownloadException("Transient network failure during download", e);
        }
        return e; // Disk errors, TLS/protocol errors and cancellation are not retryable.
    }

    private static void checkInterrupted() throws InterruptedIOException {
        if (Thread.currentThread().isInterrupted()) {
            throw new InterruptedIOException("Download interrupted; partial retained");
        }
    }

    private static Path sidecar(Path target, String suffix) {
        return target.resolveSibling(target.getFileName().toString() + suffix);
    }

    private static boolean strongEtag(String etag) {
        return etag != null && etag.matches("\"[!#-~\\x80-\\xFF]*\"");
    }

    private static long contentLength(HttpURLConnection connection) throws IOException {
        String value = connection.getHeaderField("Content-Length");
        if (value != null && connection.getHeaderField("Transfer-Encoding") != null) {
            throw new IOException("Ambiguous Content-Length and Transfer-Encoding");
        }
        return value == null ? -1 : parseLength(value);
    }

    private static long parseLength(String value) throws IOException {
        try {
            if (!value.matches("[0-9]+")) {
                throw new NumberFormatException();
            }
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            throw new IOException("Invalid byte count in response/metadata", e);
        }
    }

    private static ResumeState readState(Path metadata, String source) throws IOException {
        if (!Files.exists(metadata)) {
            return null;
        }
        if (Files.size(metadata) > 64 * 1024) {
            return null;
        }
        Properties properties = new Properties();
        try (InputStream in = Files.newInputStream(metadata)) {
            properties.load(in);
        } catch (IllegalArgumentException malformed) {
            return null;
        }
        if (!"1".equals(properties.getProperty("version")) || !source.equals(properties.getProperty("source"))) {
            return null;
        }
        try {
            return new ResumeState(properties.getProperty("etag", ""),
                    parseLength(properties.getProperty("length", "")));
        } catch (IOException malformed) {
            return null;
        }
    }

    private static void writeState(Path metadata, String source, String etag, long length) throws IOException {
        Properties properties = new Properties();
        properties.setProperty("version", "1");
        properties.setProperty("source", source);
        properties.setProperty("etag", etag);
        properties.setProperty("length", Long.toString(length));
        Path staging = sidecar(metadata, ".tmp");
        try (OutputStream out = Files.newOutputStream(staging)) {
            properties.store(out, "DL4J resumable download");
        }
        moveIntoPlace(staging, metadata);
    }

    private static void moveIntoPlace(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            // Same-directory rename on filesystems without the optional ATOMIC_MOVE facility.
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static final class ResumeState {
        private final String etag;
        private final long length;

        private ResumeState(String etag, long length) {
            this.etag = etag;
            this.length = length;
        }
    }

    private static final class TransientDownloadException extends IOException {
        private TransientDownloadException(String message, IOException cause) {
            super(message, cause);
        }
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
        return ND4JFileUtils.formatBytes(bytes);
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
