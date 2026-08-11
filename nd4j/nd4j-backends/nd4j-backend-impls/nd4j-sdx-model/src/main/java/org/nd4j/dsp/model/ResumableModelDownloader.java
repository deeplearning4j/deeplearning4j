/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.UnknownHostException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Backend-neutral, resumable HTTP downloader for large immutable model artifacts.
 *
 * <p>The implementation uses only the Java standard library and APIs available on Android. A
 * stable {@code .partial} file and adjacent metadata file survive cancellation and transient
 * failures. They are published only after length and optional SHA-256 validation.</p>
 */
public final class ResumableModelDownloader {
    private static final int BUFFER_SIZE = 64 * 1024;
    private static final String META_VERSION = "1";

    public interface ConnectionFactory {
        HttpURLConnection open(URI uri) throws IOException;
    }

    public interface MonotonicClock {
        long nanoTime();

        long currentTimeMillis();
    }

    public interface Sleeper {
        void sleep(long millis, CancellationHandle cancellation) throws InterruptedException;
    }

    public interface Jitter {
        long apply(long delayMillis, int completedAttempts);
    }

    public interface UriPolicy {
        void check(URI uri, URI previousUri, int redirectCount) throws IOException;
    }

    public interface ProgressListener {
        void onProgress(ProgressEvent event);
    }

    public enum EventType {
        ATTEMPT,
        RESUME,
        PROGRESS,
        RETRY,
        VERIFY,
        COMPLETE
    }

    public static final class ProgressEvent {
        private final EventType type;
        private final int attempt;
        private final long bytesDownloaded;
        private final long totalBytes;
        private final double bytesPerSecond;
        private final long estimatedRemainingMillis;
        private final long delayMillis;
        private final String message;

        private ProgressEvent(EventType type, int attempt, long bytesDownloaded, long totalBytes,
                              double bytesPerSecond, long estimatedRemainingMillis,
                              long delayMillis, String message) {
            this.type = type;
            this.attempt = attempt;
            this.bytesDownloaded = bytesDownloaded;
            this.totalBytes = totalBytes;
            this.bytesPerSecond = bytesPerSecond;
            this.estimatedRemainingMillis = estimatedRemainingMillis;
            this.delayMillis = delayMillis;
            this.message = message;
        }

        public EventType getType() { return type; }
        public int getAttempt() { return attempt; }
        public long getBytesDownloaded() { return bytesDownloaded; }
        public long getTotalBytes() { return totalBytes; }
        public double getBytesPerSecond() { return bytesPerSecond; }
        public long getEstimatedRemainingMillis() { return estimatedRemainingMillis; }
        public long getDelayMillis() { return delayMillis; }
        public String getMessage() { return message; }
    }

    public static final class DownloadRequest {
        private final URI uri;
        private final Path destination;
        private final long maxBytes;
        private final long expectedLength;
        private final String expectedSha256;
        private final Map<String, String> headers;

        private DownloadRequest(Builder builder) {
            uri = Objects.requireNonNull(builder.uri, "uri").normalize();
            destination = Objects.requireNonNull(builder.destination, "destination").toAbsolutePath().normalize();
            if (!uri.isAbsolute() || uri.getFragment() != null) {
                throw new IllegalArgumentException("Download URI must be absolute and have no fragment");
            }
            if (builder.maxBytes < 0) {
                throw new IllegalArgumentException("maxBytes must be non-negative");
            }
            if (builder.expectedLength < -1) {
                throw new IllegalArgumentException("expectedLength must be -1 or non-negative");
            }
            if (builder.expectedLength > builder.maxBytes) {
                throw new IllegalArgumentException("expectedLength exceeds maxBytes");
            }
            maxBytes = builder.maxBytes;
            expectedLength = builder.expectedLength;
            expectedSha256 = normalizeSha256(builder.expectedSha256);
            LinkedHashMap<String, String> copy = new LinkedHashMap<>();
            for (Map.Entry<String, String> entry : builder.headers.entrySet()) {
                String name = requireHeader(entry.getKey(), "header name");
                String value = requireHeader(entry.getValue(), "header value");
                if ("range".equalsIgnoreCase(name) || "if-range".equalsIgnoreCase(name)) {
                    throw new IllegalArgumentException("Range and If-Range are managed by the downloader");
                }
                copy.put(name, value);
            }
            headers = Collections.unmodifiableMap(copy);
        }

        public static Builder builder(URI uri, Path destination) { return new Builder(uri, destination); }
        public URI getUri() { return uri; }
        public Path getDestination() { return destination; }
        public long getMaxBytes() { return maxBytes; }
        public long getExpectedLength() { return expectedLength; }
        public String getExpectedSha256() { return expectedSha256; }
        public Map<String, String> getHeaders() { return headers; }

        public static final class Builder {
            private final URI uri;
            private final Path destination;
            private long maxBytes = Long.MAX_VALUE;
            private long expectedLength = -1;
            private String expectedSha256;
            private final Map<String, String> headers = new LinkedHashMap<>();

            private Builder(URI uri, Path destination) { this.uri = uri; this.destination = destination; }
            public Builder maxBytes(long value) { maxBytes = value; return this; }
            public Builder expectedLength(long value) { expectedLength = value; return this; }
            public Builder expectedSha256(String value) { expectedSha256 = value; return this; }
            public Builder header(String name, String value) { headers.put(name, value); return this; }
            public DownloadRequest build() { return new DownloadRequest(this); }
        }
    }

    public static final class DownloadPolicy {
        private final int maxAttempts;
        private final Duration connectTimeout;
        private final Duration readIdleTimeout;
        private final Duration attemptTimeout;
        private final Duration initialBackoff;
        private final Duration maxBackoff;
        private final Duration maxRetryAfter;
        private final Duration progressInterval;
        private final int maxRedirects;
        private final UriPolicy uriPolicy;

        private DownloadPolicy(Builder builder) {
            maxAttempts = positive(builder.maxAttempts, "maxAttempts");
            connectTimeout = positive(builder.connectTimeout, "connectTimeout");
            readIdleTimeout = positive(builder.readIdleTimeout, "readIdleTimeout");
            // Zero deliberately disables only the whole-attempt wall clock. Connect and
            // read-idle timeouts still bound a dead connection while allowing a very large,
            // continuously progressing model transfer to finish.
            attemptTimeout = nonNegative(builder.attemptTimeout, "attemptTimeout");
            initialBackoff = nonNegative(builder.initialBackoff, "initialBackoff");
            maxBackoff = nonNegative(builder.maxBackoff, "maxBackoff");
            maxRetryAfter = nonNegative(builder.maxRetryAfter, "maxRetryAfter");
            progressInterval = nonNegative(builder.progressInterval, "progressInterval");
            if (maxBackoff.compareTo(initialBackoff) < 0) {
                throw new IllegalArgumentException("maxBackoff must be at least initialBackoff");
            }
            if (builder.maxRedirects < 0) {
                throw new IllegalArgumentException("maxRedirects must be non-negative");
            }
            maxRedirects = builder.maxRedirects;
            uriPolicy = Objects.requireNonNull(builder.uriPolicy, "uriPolicy");
        }

        public static Builder builder() { return new Builder(); }
        public static DownloadPolicy defaults() { return builder().build(); }
        public int getMaxAttempts() { return maxAttempts; }
        public Duration getConnectTimeout() { return connectTimeout; }
        public Duration getReadIdleTimeout() { return readIdleTimeout; }
        public Duration getAttemptTimeout() { return attemptTimeout; }
        public Duration getInitialBackoff() { return initialBackoff; }
        public Duration getMaxBackoff() { return maxBackoff; }
        public Duration getMaxRetryAfter() { return maxRetryAfter; }
        public Duration getProgressInterval() { return progressInterval; }
        public int getMaxRedirects() { return maxRedirects; }
        public UriPolicy getUriPolicy() { return uriPolicy; }

        public static final class Builder {
            private int maxAttempts = 4;
            private Duration connectTimeout = Duration.ofSeconds(30);
            private Duration readIdleTimeout = Duration.ofMinutes(10);
            private Duration attemptTimeout = Duration.ofHours(12);
            private Duration initialBackoff = Duration.ofSeconds(1);
            private Duration maxBackoff = Duration.ofSeconds(30);
            private Duration maxRetryAfter = Duration.ofMinutes(2);
            private Duration progressInterval = Duration.ofMillis(500);
            private int maxRedirects = 5;
            private UriPolicy uriPolicy = ResumableModelDownloader::defaultUriPolicy;

            public Builder maxAttempts(int value) { maxAttempts = value; return this; }
            public Builder connectTimeout(Duration value) { connectTimeout = value; return this; }
            public Builder readIdleTimeout(Duration value) { readIdleTimeout = value; return this; }
            public Builder attemptTimeout(Duration value) { attemptTimeout = value; return this; }
            public Builder initialBackoff(Duration value) { initialBackoff = value; return this; }
            public Builder maxBackoff(Duration value) { maxBackoff = value; return this; }
            public Builder maxRetryAfter(Duration value) { maxRetryAfter = value; return this; }
            public Builder progressInterval(Duration value) { progressInterval = value; return this; }
            public Builder maxRedirects(int value) { maxRedirects = value; return this; }
            public Builder uriPolicy(UriPolicy value) { uriPolicy = value; return this; }
            public DownloadPolicy build() { return new DownloadPolicy(this); }
        }
    }

    public static final class DownloadResult {
        private final Path path;
        private final long bytes;
        private final String sha256;
        private final int attempts;
        private final boolean resumed;

        private DownloadResult(Path path, long bytes, String sha256, int attempts, boolean resumed) {
            this.path = path;
            this.bytes = bytes;
            this.sha256 = sha256;
            this.attempts = attempts;
            this.resumed = resumed;
        }

        public Path getPath() { return path; }
        public long getBytes() { return bytes; }
        public String getSha256() { return sha256; }
        public int getAttempts() { return attempts; }
        public boolean isResumed() { return resumed; }
    }

    public static final class CancellationHandle {
        private final AtomicBoolean cancelled = new AtomicBoolean();
        private final AtomicReference<HttpURLConnection> activeConnection = new AtomicReference<>();
        private final Object backoffMonitor = new Object();

        public void cancel() {
            cancelled.set(true);
            HttpURLConnection connection = activeConnection.getAndSet(null);
            if (connection != null) {
                connection.disconnect();
            }
            synchronized (backoffMonitor) {
                backoffMonitor.notifyAll();
            }
        }

        public boolean isCancelled() { return cancelled.get(); }
    }

    public static final class DownloadCancelledException extends IOException {
        public DownloadCancelledException() { super("Model download was cancelled"); }
    }

    public static final class HttpStatusException extends IOException {
        private final int statusCode;
        private final URI uri;
        private final long retryAfterMillis;

        private HttpStatusException(int statusCode, URI uri, long retryAfterMillis) {
            super("HTTP " + statusCode + " while downloading " + uri);
            this.statusCode = statusCode;
            this.uri = uri;
            this.retryAfterMillis = retryAfterMillis;
        }

        public int getStatusCode() { return statusCode; }
        public URI getUri() { return uri; }
    }

    private final ConnectionFactory connectionFactory;
    private final MonotonicClock clock;
    private final Sleeper sleeper;
    private final Jitter jitter;

    public ResumableModelDownloader() {
        this(uri -> (HttpURLConnection) new URL(uri.toASCIIString()).openConnection(),
                new MonotonicClock() {
                    public long nanoTime() { return System.nanoTime(); }
                    public long currentTimeMillis() { return System.currentTimeMillis(); }
                }, ResumableModelDownloader::interruptibleSleep, (delay, attempt) -> delay);
    }

    public ResumableModelDownloader(ConnectionFactory connectionFactory, MonotonicClock clock,
                                    Sleeper sleeper, Jitter jitter) {
        this.connectionFactory = Objects.requireNonNull(connectionFactory, "connectionFactory");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.sleeper = Objects.requireNonNull(sleeper, "sleeper");
        this.jitter = Objects.requireNonNull(jitter, "jitter");
    }

    public DownloadResult download(DownloadRequest request) throws IOException {
        return download(request, DownloadPolicy.defaults(), event -> { }, new CancellationHandle());
    }

    public DownloadResult download(DownloadRequest request, DownloadPolicy policy,
                                   ProgressListener listener, CancellationHandle cancellation) throws IOException {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(policy, "policy");
        Objects.requireNonNull(listener, "listener");
        Objects.requireNonNull(cancellation, "cancellation");
        Path destination = request.getDestination();
        Path partial = sibling(destination, ".partial");
        // Keep the sidecar name compatible with the Android downloader this utility replaces.
        Path metadataPath = sibling(destination, ".partial.metadata");
        Path parent = destination.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        policy.getUriPolicy().check(request.getUri(), null, 0);
        prepareLocalState(request, partial, metadataPath);

        IOException last = null;
        AttemptResult completedTransfer = null;
        int completedAttempt = 0;
        boolean everResumed = false;
        String verificationMessage = "verifying length and SHA-256";
        for (int attempt = 1; attempt <= policy.getMaxAttempts(); attempt++) {
            checkCancelled(cancellation);
            long localBytes = size(partial);
            Metadata localMetadata = readMetadata(metadataPath);
            long localTotal = knownTotal(request, localMetadata);
            if (localBytes > 0 && localTotal >= 0 && localBytes == localTotal) {
                completedTransfer = new AttemptResult(localTotal, true, 0);
                completedAttempt = attempt;
                everResumed = true;
                verificationMessage = "verifying completed partial locally";
                break;
            }

            emit(listener, new ProgressEvent(EventType.ATTEMPT, attempt, localBytes,
                    localTotal, 0, -1, 0,
                    "attempt " + attempt + " of " + policy.getMaxAttempts()));
            long attemptDeadline = deadline(clock.nanoTime(), policy.getAttemptTimeout());
            try {
                completedTransfer = executeAttempt(request, policy, listener, cancellation,
                        partial, metadataPath, attempt, attemptDeadline);
                completedAttempt = attempt;
                everResumed |= completedTransfer.resumed;
                break;
            } catch (DownloadCancelledException cancelled) {
                throw cancelled;
            } catch (PoisonedPartialException poisoned) {
                discardPartial(partial, metadataPath);
                throw poisoned;
            } catch (StalePartialException stale) {
                discardPartial(partial, metadataPath);
                last = stale;
                if (attempt == policy.getMaxAttempts()) {
                    throw stale;
                }
                backoff(policy, listener, cancellation, attempt, 0, stale.getMessage(),
                        size(partial), knownTotal(request, readMetadata(metadataPath)));
            } catch (HttpStatusException status) {
                last = status;
                if (!isRetryableStatus(status.getStatusCode()) || attempt == policy.getMaxAttempts()) {
                    throw status;
                }
                long retryAfter = Math.min(status.retryAfterMillis, millis(policy.getMaxRetryAfter()));
                backoff(policy, listener, cancellation, attempt, retryAfter, status.getMessage(),
                        size(partial), knownTotal(request, readMetadata(metadataPath)));
            } catch (IOException failure) {
                last = failure;
                if (cancellation.isCancelled()) {
                    throw new DownloadCancelledException();
                }
                if (!isTransient(failure) || attempt == policy.getMaxAttempts()) {
                    throw failure;
                }
                backoff(policy, listener, cancellation, attempt, -1, failure.getMessage(),
                        size(partial), knownTotal(request, readMetadata(metadataPath)));
            }
        }
        if (completedTransfer == null) {
            throw last == null ? new IOException("Model download failed") : last;
        }
        return verifyAndPublish(request, policy, listener, cancellation, partial, metadataPath,
                destination, completedTransfer, completedAttempt, everResumed, verificationMessage);
    }

    /**
     * Return the exact validator-backed prefix reusable by {@link #download}, deleting stale
     * or incomplete local state first. This is intended for storage preflight calculations.
     */
    public long resumableBytes(DownloadRequest request) throws IOException {
        Objects.requireNonNull(request, "request");
        Path partial = sibling(request.getDestination(), ".partial");
        Path metadataPath = sibling(request.getDestination(), ".partial.metadata");
        prepareLocalState(request, partial, metadataPath);
        Metadata metadata = readMetadata(metadataPath);
        return metadata != null && metadata.validator() != null ? size(partial) : 0;
    }

    private AttemptResult executeAttempt(DownloadRequest request, DownloadPolicy policy,
                                         ProgressListener listener, CancellationHandle cancellation,
                                         Path partial, Path metadataPath, int attempt,
                                         long attemptDeadline) throws IOException {
        Metadata metadata = readMetadata(metadataPath);
        long offset = size(partial);
        boolean resume = offset > 0 && metadata != null;
        if (offset > 0 && !resume) {
            // A previous response without a strong validator is not safe to append to.
            discardPartial(partial, metadataPath);
            offset = 0;
        }
        if (resume) {
            emit(listener, new ProgressEvent(EventType.RESUME, attempt, offset,
                    knownTotal(request, metadata), 0, -1, 0, "requesting byte range at " + offset));
        }
        URI uri = request.getUri();
        int redirects = 0;
        while (true) {
            checkCancelled(cancellation);
            ensureTimeRemaining(attemptDeadline, "whole-attempt timeout");
            HttpURLConnection connection = connectionFactory.open(uri);
            cancellation.activeConnection.set(connection);
            try {
                configure(connection, request, policy, resume, offset, metadata, attemptDeadline);
                int status = connection.getResponseCode();
                if (isRedirect(status)) {
                    if (redirects >= policy.getMaxRedirects()) {
                        throw new IOException("HTTP redirect limit exceeded");
                    }
                    String location = connection.getHeaderField("Location");
                    if (location == null || location.trim().isEmpty()) {
                        throw new IOException("HTTP redirect omitted Location");
                    }
                    URI next = uri.resolve(location).normalize();
                    try {
                        policy.getUriPolicy().check(next, uri, redirects + 1);
                    } catch (IOException rejected) {
                        throw new NonRetryableDownloadException(rejected.getMessage(), rejected);
                    }
                    uri = next;
                    redirects++;
                    continue;
                }
                if (status < 200 || status >= 300) {
                    throw new HttpStatusException(status, uri, retryAfterMillis(connection));
                }
                if (resume && status == HttpURLConnection.HTTP_OK) {
                    discardPartial(partial, metadataPath);
                    offset = 0;
                    resume = false;
                    metadata = null;
                } else if (resume) {
                    validateResumeResponse(connection, offset, metadata, request);
                } else if (status == HttpURLConnection.HTTP_PARTIAL) {
                    validateRangeFromZero(connection, request);
                }
                ResponseInfo response = responseInfo(connection, request, metadata, offset, resume);
                Metadata responseMetadata = new Metadata(request.getUri().toASCIIString(),
                        response.etag, response.lastModified, response.totalBytes,
                        request.getExpectedSha256());
                if (responseMetadata.validator() != null) {
                    writeMetadata(metadataPath, responseMetadata);
                } else {
                    // A prefix without ETag or Last-Modified cannot be resumed safely.
                    Files.deleteIfExists(metadataPath);
                }
                return transfer(connection, request, policy, listener, cancellation, partial,
                        attempt, attemptDeadline, offset, response.totalBytes, resume);
            } finally {
                cancellation.activeConnection.compareAndSet(connection, null);
                connection.disconnect();
            }
        }
    }

    private AttemptResult transfer(HttpURLConnection connection, DownloadRequest request,
                                   DownloadPolicy policy, ProgressListener listener,
                                   CancellationHandle cancellation, Path partial, int attempt,
                                   long deadline, long offset, long total, boolean resumed) throws IOException {
        long bytes = offset;
        long completionLength = total >= 0 ? total : request.getExpectedLength();
        if (completionLength >= 0 && bytes > completionLength) {
            throw new PoisonedPartialException("Partial download has " + bytes
                    + " bytes; expected at most " + completionLength);
        }
        long startBytes = offset;
        long startNanos = clock.nanoTime();
        long lastEvent = startNanos;
        StandardOpenOption[] options = offset == 0
                ? new StandardOpenOption[] { StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                                             StandardOpenOption.WRITE }
                : new StandardOpenOption[] { StandardOpenOption.CREATE, StandardOpenOption.APPEND,
                                             StandardOpenOption.WRITE };
        updateReadTimeout(connection, policy, deadline);
        try (InputStream input = new BufferedInputStream(connection.getInputStream());
             OutputStream output = new BufferedOutputStream(Files.newOutputStream(partial, options))) {
            byte[] buffer = new byte[BUFFER_SIZE];
            while (true) {
                checkCancelled(cancellation);
                // A successful HTTP body is not required to close immediately. In particular,
                // Hugging Face's CDN can keep the connection alive after the declared model
                // bytes have arrived. Do not turn a complete download into an idle timeout by
                // waiting for a redundant EOF read when the representation length is known.
                if (completionLength >= 0 && bytes == completionLength) {
                    break;
                }
                updateReadTimeout(connection, policy, deadline);
                int count;
                try {
                    count = input.read(buffer);
                } catch (SocketTimeoutException timeout) {
                    throw new SocketTimeoutException("Read-idle or whole-attempt timeout");
                }
                if (count < 0) {
                    break;
                }
                if (bytes > request.getMaxBytes() - count) {
                    throw new PoisonedPartialException("Download exceeds maxBytes");
                }
                if (completionLength >= 0 && bytes > completionLength - count) {
                    throw new PoisonedPartialException("Response exceeds expected length "
                            + completionLength);
                }
                output.write(buffer, 0, count);
                bytes += count;
                long now = clock.nanoTime();
                if (now - lastEvent >= policy.getProgressInterval().toNanos()) {
                    double rate = rate(bytes - startBytes, now - startNanos);
                    emit(listener, progress(attempt, bytes, total, rate));
                    lastEvent = now;
                }
            }
        }
        long elapsed = clock.nanoTime() - startNanos;
        double finalRate = rate(bytes - startBytes, elapsed);
        if (total >= 0 && bytes != total) {
            throw new EOFException("Response ended at " + bytes + " bytes; expected " + total);
        }
        if (request.getExpectedLength() >= 0 && bytes != request.getExpectedLength()) {
            throw new EOFException("Download ended at " + bytes + " bytes; expected " + request.getExpectedLength());
        }
        return new AttemptResult(bytes, resumed, finalRate);
    }

    private void configure(HttpURLConnection connection, DownloadRequest request,
                           DownloadPolicy policy, boolean resume, long offset,
                           Metadata metadata, long deadline) throws IOException {
        connection.setInstanceFollowRedirects(false);
        connection.setRequestMethod("GET");
        connection.setUseCaches(false);
        for (Map.Entry<String, String> header : request.getHeaders().entrySet()) {
            connection.setRequestProperty(header.getKey(), header.getValue());
        }
        if (resume) {
            connection.setRequestProperty("Range", "bytes=" + offset + "-");
            String validator = metadata.validator();
            if (validator != null) {
                connection.setRequestProperty("If-Range", validator);
            }
        }
        long remainingMillis = remainingMillis(deadline, clock.nanoTime());
        connection.setConnectTimeout(toTimeoutMillis(Math.min(millis(policy.getConnectTimeout()), remainingMillis)));
        connection.setReadTimeout(toTimeoutMillis(Math.min(millis(policy.getReadIdleTimeout()), remainingMillis)));
    }

    private void updateReadTimeout(HttpURLConnection connection, DownloadPolicy policy,
                                   long deadline) throws SocketTimeoutException {
        long remaining = remainingMillis(deadline, clock.nanoTime());
        if (remaining <= 0) {
            throw new SocketTimeoutException("Whole-attempt timeout");
        }
        connection.setReadTimeout(toTimeoutMillis(Math.min(millis(policy.getReadIdleTimeout()), remaining)));
    }

    private static void validateResumeResponse(HttpURLConnection connection, long offset,
                                               Metadata metadata, DownloadRequest request) throws IOException {
        if (connection.getResponseCode() != HttpURLConnection.HTTP_PARTIAL) {
            throw new StalePartialException("Range resume did not return HTTP 206");
        }
        ContentRange range;
        try {
            range = parseContentRange(connection.getHeaderField("Content-Range"));
        } catch (PoisonedPartialException invalidRange) {
            throw new StalePartialException(invalidRange.getMessage(), invalidRange);
        }
        if (range.start != offset) {
            throw new StalePartialException("Content-Range starts at " + range.start + ", expected " + offset);
        }
        if (range.total >= 0 && metadata.totalBytes >= 0 && range.total != metadata.totalBytes) {
            throw new StalePartialException("Resumed representation length changed");
        }
        if (request.getExpectedLength() >= 0 && range.total >= 0 && range.total != request.getExpectedLength()) {
            throw new StalePartialException("Resumed representation has unexpected length");
        }
        String etag = strongEtag(connection.getHeaderField("ETag"));
        String modified = trim(connection.getHeaderField("Last-Modified"));
        if (metadata.etag != null && etag != null && !metadata.etag.equals(etag)) {
            throw new StalePartialException("Resumed representation ETag changed");
        }
        if (metadata.etag == null && metadata.lastModified != null && modified != null
                && !metadata.lastModified.equals(modified)) {
            throw new StalePartialException("Resumed representation Last-Modified changed");
        }
    }

    private static void validateRangeFromZero(HttpURLConnection connection,
                                              DownloadRequest request) throws IOException {
        ContentRange range = parseContentRange(connection.getHeaderField("Content-Range"));
        if (range.start != 0) {
            throw new PoisonedPartialException("Unexpected partial response starts at " + range.start);
        }
        if (request.getExpectedLength() >= 0 && range.total >= 0 && range.total != request.getExpectedLength()) {
            throw new PoisonedPartialException("Response has unexpected total length");
        }
    }

    private static ResponseInfo responseInfo(HttpURLConnection connection, DownloadRequest request,
                                             Metadata old, long offset, boolean resumed) throws IOException {
        long contentLength = connection.getContentLengthLong();
        long total;
        if (connection.getResponseCode() == HttpURLConnection.HTTP_PARTIAL) {
            ContentRange range = parseContentRange(connection.getHeaderField("Content-Range"));
            total = range.total >= 0 ? range.total : safeAdd(offset, contentLength);
        } else {
            total = contentLength;
        }
        if (total < 0 && resumed && old != null) {
            total = old.totalBytes;
        }
        if (total > request.getMaxBytes()) {
            throw new PoisonedPartialException("Response exceeds maxBytes");
        }
        if (request.getExpectedLength() >= 0 && total >= 0 && total != request.getExpectedLength()) {
            throw new PoisonedPartialException("Response length " + total
                    + " does not match expected length " + request.getExpectedLength());
        }
        return new ResponseInfo(total, trim(connection.getHeaderField("ETag")),
                trim(connection.getHeaderField("Last-Modified")));
    }

    private static void prepareLocalState(DownloadRequest request, Path partial,
                                          Path metadataPath) throws IOException {
        boolean partialExists = Files.exists(partial);
        boolean metadataExists = Files.exists(metadataPath);
        if (partialExists != metadataExists) {
            discardPartial(partial, metadataPath);
            return;
        }
        if (!partialExists) {
            return;
        }
        Metadata metadata;
        try {
            metadata = readMetadata(metadataPath);
        } catch (IOException invalid) {
            discardPartial(partial, metadataPath);
            return;
        }
        long bytes = Files.size(partial);
        boolean valid = metadata != null
                && request.getUri().toASCIIString().equals(metadata.uri)
                && metadata.validator() != null
                && bytes <= request.getMaxBytes()
                && Objects.equals(request.getExpectedSha256(), metadata.expectedSha256)
                && (request.getExpectedLength() < 0 || bytes <= request.getExpectedLength())
                && (metadata.totalBytes < 0 || bytes <= metadata.totalBytes)
                && (request.getExpectedLength() < 0 || metadata.totalBytes < 0
                    || request.getExpectedLength() == metadata.totalBytes);
        if (!valid) {
            discardPartial(partial, metadataPath);
        }
    }

    private String validateCompleted(DownloadRequest request, DownloadPolicy policy,
                                     ProgressListener listener, int attempt, Path partial,
                                     long total, CancellationHandle cancellation,
                                     String message) throws IOException {
        long actual = Files.size(partial);
        if (actual != total) {
            throw new PoisonedPartialException("Model verification failed: downloaded file length changed "
                    + "before verification (received " + total + " bytes, found " + actual + " bytes)");
        }
        if (actual > request.getMaxBytes()) {
            throw new PoisonedPartialException("Model verification failed: downloaded file has " + actual
                    + " bytes, exceeding the configured limit of " + request.getMaxBytes() + " bytes");
        }
        if (request.getExpectedLength() >= 0 && actual != request.getExpectedLength()) {
            throw new PoisonedPartialException("Model verification failed: byte length mismatch (expected "
                    + request.getExpectedLength() + " bytes, actual " + actual + " bytes)");
        }
        String digest = sha256(partial, actual, policy, listener, attempt, cancellation, message);
        if (request.getExpectedSha256() != null
                && !request.getExpectedSha256().equals(digest)) {
            throw new PoisonedPartialException("Model verification failed: SHA-256 mismatch for " + actual
                    + " bytes (expected " + request.getExpectedSha256() + ", actual " + digest + ")");
        }
        emit(listener, verificationProgress(attempt, actual, actual, 0,
                request.getExpectedSha256() == null
                        ? "length verified; SHA-256 calculated"
                        : "length and SHA-256 verified against expected values"));
        return digest;
    }

    private DownloadResult verifyAndPublish(DownloadRequest request, DownloadPolicy policy,
                                            ProgressListener listener, CancellationHandle cancellation,
                                            Path partial, Path metadataPath, Path destination,
                                            AttemptResult completedTransfer, int attempt,
                                            boolean everResumed, String verificationMessage) throws IOException {
        String digest;
        try {
            digest = validateCompleted(request, policy, listener, attempt, partial,
                    completedTransfer.totalBytes, cancellation, verificationMessage);
        } catch (PoisonedPartialException poisoned) {
            discardPartial(partial, metadataPath);
            throw poisoned;
        }
        checkCancelled(cancellation);
        publish(partial, destination);
        String completionMessage = cleanupPublishedMetadata(metadataPath);
        emit(listener, new ProgressEvent(EventType.COMPLETE, attempt, completedTransfer.totalBytes,
                completedTransfer.totalBytes, completedTransfer.rate, 0, 0, completionMessage));
        return new DownloadResult(destination, completedTransfer.totalBytes, digest, attempt, everResumed);
    }

    private static String cleanupPublishedMetadata(Path metadataPath) {
        try {
            Files.deleteIfExists(metadataPath);
            return "download complete";
        } catch (IOException failure) {
            String detail = failure.getMessage();
            return "download complete; resume metadata cleanup deferred"
                    + (detail == null || detail.isEmpty() ? "" : ": " + detail);
        }
    }

    private static void publish(Path partial, Path destination) throws IOException {
        try {
            // Both paths are siblings in app-owned storage. The default move is no-replace and
            // maps to the filesystem's rename operation without requiring hard-link support,
            // which Android application filesystems do not consistently expose through NIO.
            Files.move(partial, destination);
        } catch (FileAlreadyExistsException exists) {
            throw new NonRetryableDownloadException(
                    "Destination appeared before verified model publication", exists);
        } catch (UnsupportedOperationException unsupported) {
            throw new NonRetryableDownloadException(
                    "Destination filesystem cannot publish the verified model", unsupported);
        } catch (IOException failure) {
            // Publication is a local storage operation, not another network attempt. Retrying it
            // inside the transfer loop would repeatedly hash the same complete file and surface
            // as an endless Verify -> Download cycle. Preserve the complete partial for an
            // explicit resume after the underlying storage problem is fixed.
            String detail = failure.getMessage();
            throw new NonRetryableDownloadException(
                    "Could not publish the verified model into app storage"
                            + (detail == null || detail.isEmpty() ? "" : ": " + detail),
                    failure);
        }
    }

    private void backoff(DownloadPolicy policy, ProgressListener listener,
                         CancellationHandle cancellation, int attempt,
                         long retryAfterMillis, String reason,
                         long savedBytes, long totalBytes) throws IOException {
        long exponential = saturatedShift(millis(policy.getInitialBackoff()), attempt - 1);
        long bounded = Math.min(exponential, millis(policy.getMaxBackoff()));
        long selected = retryAfterMillis >= 0 ? retryAfterMillis : bounded;
        long delay = Math.max(0, Math.min(jitter.apply(selected, attempt),
                retryAfterMillis >= 0 ? millis(policy.getMaxRetryAfter()) : millis(policy.getMaxBackoff())));
        emit(listener, new ProgressEvent(EventType.RETRY, attempt, savedBytes, totalBytes, 0, -1,
                delay, reason == null ? "retry" : reason));
        try {
            sleeper.sleep(delay, cancellation);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted during retry backoff", interrupted);
        }
        checkCancelled(cancellation);
    }

    private static void interruptibleSleep(long millis, CancellationHandle cancellation)
            throws InterruptedException {
        long deadline = System.nanoTime() + Duration.ofMillis(millis).toNanos();
        synchronized (cancellation.backoffMonitor) {
            while (!cancellation.isCancelled()) {
                long remaining = deadline - System.nanoTime();
                if (remaining <= 0) {
                    return;
                }
                long waitMillis = Math.max(1, Math.min(millis, remaining / 1_000_000L));
                cancellation.backoffMonitor.wait(waitMillis);
            }
        }
    }

    private long retryAfterMillis(HttpURLConnection connection) {
        String value = trim(connection.getHeaderField("Retry-After"));
        if (value == null) {
            return -1;
        }
        try {
            long seconds = Long.parseLong(value);
            return seconds < 0 ? -1 : saturatedMultiply(seconds, 1000);
        } catch (NumberFormatException ignored) {
            try {
                long target = ZonedDateTime.parse(value, DateTimeFormatter.RFC_1123_DATE_TIME)
                        .toInstant().toEpochMilli();
                return Math.max(0, target - clock.currentTimeMillis());
            } catch (DateTimeParseException invalid) {
                return -1;
            }
        }
    }

    private static Metadata readMetadata(Path path) throws IOException {
        if (!Files.exists(path)) {
            return null;
        }
        Properties properties = new Properties();
        try (InputStream input = Files.newInputStream(path)) {
            properties.load(input);
        }
        if (!META_VERSION.equals(properties.getProperty("version"))) {
            throw new IOException("Unsupported partial metadata version");
        }
        // version 1 sidecars written by the former Android implementation used
        // identity/expectedBytes/validatorKind/validator. Read both layouts so an app
        // upgrade does not throw away a safe multi-gigabyte partial.
        String uri = properties.getProperty("uri", properties.getProperty("identity"));
        if (uri == null) {
            throw new IOException("Partial metadata omitted URI");
        }
        long total;
        try {
            String encodedTotal = properties.getProperty(
                    "totalBytes", properties.getProperty("expectedBytes", "-1"));
            total = encodedTotal == null || encodedTotal.isEmpty() ? -1 : Long.parseLong(encodedTotal);
        } catch (NumberFormatException invalid) {
            throw new IOException("Invalid partial metadata length", invalid);
        }
        String etag = emptyToNull(properties.getProperty("etag"));
        String lastModified = emptyToNull(properties.getProperty("lastModified"));
        if (etag == null && lastModified == null) {
            String validator = emptyToNull(properties.getProperty("validator"));
            String validatorKind = emptyToNull(properties.getProperty("validatorKind"));
            if ("etag".equals(validatorKind)) etag = validator;
            else if ("last-modified".equals(validatorKind)) lastModified = validator;
        }
        return new Metadata(uri, etag, lastModified, total,
                emptyToNull(properties.getProperty("expectedSha256")));
    }

    private static void writeMetadata(Path path, Metadata metadata) throws IOException {
        Path temporary = sibling(path, ".tmp");
        StringBuilder value = new StringBuilder();
        appendProperty(value, "version", META_VERSION);
        appendProperty(value, "uri", metadata.uri);
        appendProperty(value, "etag", metadata.etag == null ? "" : metadata.etag);
        appendProperty(value, "lastModified", metadata.lastModified == null ? "" : metadata.lastModified);
        appendProperty(value, "totalBytes", Long.toString(metadata.totalBytes));
        appendProperty(value, "expectedSha256",
                metadata.expectedSha256 == null ? "" : metadata.expectedSha256);
        Files.write(temporary, value.toString().getBytes(StandardCharsets.ISO_8859_1),
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
        try {
            Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.deleteIfExists(temporary);
            throw new NonRetryableDownloadException(
                    "Filesystem does not support atomic partial metadata", unsupported);
        }
    }

    private static void appendProperty(StringBuilder output, String key, String value) {
        output.append(key).append('=');
        for (int i = 0; i < value.length(); i++) {
            char character = value.charAt(i);
            if (character == '\\' || character == '\n' || character == '\r') {
                output.append('\\');
                if (character == '\n') output.append('n');
                else if (character == '\r') output.append('r');
                else output.append('\\');
            } else {
                output.append(character);
            }
        }
        output.append('\n');
    }

    private static ContentRange parseContentRange(String value) throws IOException {
        if (value == null || !value.startsWith("bytes ")) {
            throw new PoisonedPartialException("Missing or invalid Content-Range");
        }
        int dash = value.indexOf('-', 6);
        int slash = value.indexOf('/', dash + 1);
        if (dash < 0 || slash < 0) {
            throw new PoisonedPartialException("Invalid Content-Range");
        }
        try {
            long start = Long.parseLong(value.substring(6, dash));
            long end = Long.parseLong(value.substring(dash + 1, slash));
            long total = "*".equals(value.substring(slash + 1))
                    ? -1 : Long.parseLong(value.substring(slash + 1));
            if (start < 0 || end < start || (total >= 0 && end >= total)) {
                throw new NumberFormatException();
            }
            return new ContentRange(start, end, total);
        } catch (NumberFormatException invalid) {
            throw new PoisonedPartialException("Invalid Content-Range");
        }
    }

    private static void defaultUriPolicy(URI uri, URI previous, int redirects) throws IOException {
        String scheme = uri.getScheme();
        if (!("https".equalsIgnoreCase(scheme) || "http".equalsIgnoreCase(scheme))
                || uri.getHost() == null || uri.getUserInfo() != null || uri.getFragment() != null) {
            throw new IOException("URI policy rejected " + uri);
        }
        if (previous != null && "https".equalsIgnoreCase(previous.getScheme())
                && !"https".equalsIgnoreCase(scheme)) {
            throw new IOException("URI policy rejected HTTPS downgrade");
        }
    }

    private static boolean isTransient(IOException failure) {
        // Automatic retries are only for transport failures before a complete response exists.
        // Local file, metadata, verification, and publication failures must surface directly.
        return failure instanceof EOFException
                || failure instanceof SocketTimeoutException
                || failure instanceof SocketException
                || failure instanceof UnknownHostException;
    }

    private static boolean isRetryableStatus(int status) {
        return status == 408 || status == 425 || status == 429 || status == 500
                || status == 502 || status == 503 || status == 504;
    }

    private static boolean isRedirect(int status) {
        return status == 301 || status == 302 || status == 303 || status == 307 || status == 308;
    }

    private static ProgressEvent progress(int attempt, long bytes, long total, double rate) {
        long eta = total >= bytes && rate > 0
                ? (long) (((total - bytes) / rate) * 1000.0) : -1;
        return new ProgressEvent(EventType.PROGRESS, attempt, bytes, total, rate, eta, 0, "progress");
    }

    private static void emit(ProgressListener listener, ProgressEvent event) {
        listener.onProgress(event);
    }

    private String sha256(Path path, long totalBytes, DownloadPolicy policy,
                          ProgressListener listener, int attempt,
                          CancellationHandle cancellation, String message) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
        long verifiedBytes = 0;
        long startNanos = clock.nanoTime();
        long lastEvent = startNanos;
        emit(listener, verificationProgress(attempt, 0, totalBytes, 0, message));
        try (InputStream input = new BufferedInputStream(Files.newInputStream(path))) {
            byte[] buffer = new byte[BUFFER_SIZE];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                checkCancelled(cancellation);
                if (count > 0) {
                    digest.update(buffer, 0, count);
                    verifiedBytes += count;
                }
                long now = clock.nanoTime();
                if (now - lastEvent >= policy.getProgressInterval().toNanos()) {
                    emit(listener, verificationProgress(
                            attempt, verifiedBytes, totalBytes,
                            rate(verifiedBytes, now - startNanos), message));
                    lastEvent = now;
                }
            }
        }
        checkCancelled(cancellation);
        if (verifiedBytes != totalBytes) {
            throw new PoisonedPartialException("Model verification failed: file length changed while SHA-256 "
                    + "was calculated (expected to verify " + totalBytes + " bytes, verified "
                    + verifiedBytes + " bytes)");
        }
        long completedNanos = clock.nanoTime();
        emit(listener, verificationProgress(
                attempt, verifiedBytes, totalBytes,
                rate(verifiedBytes, completedNanos - startNanos),
                "SHA-256 calculated; comparing verification result"));
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest.digest()) hex.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        return hex.toString();
    }

    private static ProgressEvent verificationProgress(int attempt, long verifiedBytes,
                                                      long totalBytes, double rate,
                                                      String message) {
        long eta = totalBytes >= verifiedBytes && rate > 0
                ? (long) (((totalBytes - verifiedBytes) / rate) * 1000.0) : -1;
        return new ProgressEvent(EventType.VERIFY, attempt, verifiedBytes, totalBytes,
                rate, eta, 0, message);
    }

    private static String normalizeSha256(String value) {
        if (value == null || value.trim().isEmpty()) return null;
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if (!normalized.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("expectedSha256 must contain 64 hexadecimal characters");
        }
        return normalized;
    }

    private static String requireHeader(String value, String label) {
        if (value == null || value.isEmpty() || value.indexOf('\r') >= 0 || value.indexOf('\n') >= 0) {
            throw new IllegalArgumentException(label + " must be non-empty and single-line");
        }
        return value;
    }

    private static Duration positive(Duration value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isZero() || value.isNegative()) throw new IllegalArgumentException(name + " must be positive");
        return value;
    }

    private static int positive(int value, String name) {
        if (value <= 0) throw new IllegalArgumentException(name + " must be positive");
        return value;
    }

    private static Duration nonNegative(Duration value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isNegative()) throw new IllegalArgumentException(name + " must be non-negative");
        return value;
    }

    private static long deadline(long now, Duration timeout) {
        if (timeout.isZero()) return Long.MAX_VALUE;
        long nanos = timeout.toNanos();
        return Long.MAX_VALUE - now < nanos ? Long.MAX_VALUE : now + nanos;
    }

    private void ensureTimeRemaining(long deadline, String message) throws SocketTimeoutException {
        if (clock.nanoTime() >= deadline) throw new SocketTimeoutException(message);
    }

    private static long remainingMillis(long deadline, long now) {
        if (deadline == Long.MAX_VALUE) return Integer.MAX_VALUE;
        long nanos = deadline - now;
        if (nanos <= 0) return 0;
        return Math.max(1, (nanos + 999_999L) / 1_000_000L);
    }

    private static int toTimeoutMillis(long value) {
        return (int) Math.max(1, Math.min(Integer.MAX_VALUE, value));
    }

    private static long millis(Duration duration) {
        try { return duration.toMillis(); }
        catch (ArithmeticException overflow) { return Long.MAX_VALUE; }
    }

    private static double rate(long bytes, long nanos) {
        return nanos <= 0 ? 0 : bytes * 1_000_000_000.0 / nanos;
    }

    private static long safeAdd(long left, long right) throws IOException {
        if (right < 0) return -1;
        if (Long.MAX_VALUE - left < right) throw new IOException("Response length overflow");
        return left + right;
    }

    private static long saturatedShift(long value, int shifts) {
        long result = value;
        for (int i = 0; i < shifts; i++) {
            if (result > Long.MAX_VALUE / 2) return Long.MAX_VALUE;
            result *= 2;
        }
        return result;
    }

    private static long saturatedMultiply(long left, long right) {
        return left > Long.MAX_VALUE / right ? Long.MAX_VALUE : left * right;
    }

    private static long size(Path path) throws IOException { return Files.exists(path) ? Files.size(path) : 0; }
    private static Path sibling(Path path, String suffix) { return path.resolveSibling(path.getFileName() + suffix); }
    private static String trim(String value) { return value == null || value.trim().isEmpty() ? null : value.trim(); }
    private static String emptyToNull(String value) { return value == null || value.isEmpty() ? null : value; }

    private static String strongEtag(String value) {
        String etag = trim(value);
        return etag != null && !etag.regionMatches(true, 0, "W/", 0, 2) ? etag : null;
    }

    private static long knownTotal(DownloadRequest request, Metadata metadata) {
        if (request.getExpectedLength() >= 0) return request.getExpectedLength();
        return metadata == null ? -1 : metadata.totalBytes;
    }

    private static void discardPartial(Path partial, Path metadata) throws IOException {
        Files.deleteIfExists(partial);
        Files.deleteIfExists(metadata);
    }

    private static void checkCancelled(CancellationHandle cancellation) throws DownloadCancelledException {
        if (cancellation.isCancelled()) throw new DownloadCancelledException();
    }

    private static final class Metadata {
        private final String uri;
        private final String etag;
        private final String lastModified;
        private final long totalBytes;
        private final String expectedSha256;
        private Metadata(String uri, String etag, String lastModified, long totalBytes,
                         String expectedSha256) {
            this.uri = uri;
            this.etag = strongEtag(etag);
            this.lastModified = lastModified;
            this.totalBytes = totalBytes;
            this.expectedSha256 = expectedSha256;
        }
        private String validator() { return etag != null ? etag : lastModified; }
    }

    private static final class ContentRange {
        private final long start;
        private final long end;
        private final long total;
        private ContentRange(long start, long end, long total) { this.start = start; this.end = end; this.total = total; }
    }

    private static final class ResponseInfo {
        private final long totalBytes;
        private final String etag;
        private final String lastModified;
        private ResponseInfo(long totalBytes, String etag, String lastModified) {
            this.totalBytes = totalBytes; this.etag = etag; this.lastModified = lastModified;
        }
    }

    private static final class AttemptResult {
        private final long totalBytes;
        private final boolean resumed;
        private final double rate;
        private AttemptResult(long totalBytes, boolean resumed, double rate) {
            this.totalBytes = totalBytes; this.resumed = resumed; this.rate = rate;
        }
    }

    private static class PoisonedPartialException extends IOException {
        private PoisonedPartialException(String message) { super(message); }
    }

    private static class StalePartialException extends IOException {
        private StalePartialException(String message) { super(message); }
        private StalePartialException(String message, Throwable cause) { super(message, cause); }
    }

    private static class NonRetryableDownloadException extends IOException {
        private NonRetryableDownloadException(String message, Throwable cause) { super(message, cause); }
    }

}
