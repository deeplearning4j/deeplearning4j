/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class ResumableModelDownloaderTest {

    private static final URI MODEL_URI = URI.create("https://huggingface.co/acme/model/resolve/0123456789abcdef/model.gguf");

    @TempDir
    Path temporaryDirectory;

    @Test
    void productionDefaultsAreBoundedForMultiGigabyteModels() {
        ResumableModelDownloader.DownloadPolicy policy =
                ResumableModelDownloader.DownloadPolicy.defaults();
        assertEquals(4, policy.getMaxAttempts());
        assertEquals(Duration.ofSeconds(30), policy.getConnectTimeout());
        assertEquals(Duration.ofMinutes(10), policy.getReadIdleTimeout());
        assertEquals(Duration.ofHours(12), policy.getAttemptTimeout());
    }

    @Test
    void verificationReportsMeasuredMonotonicByteProgress() throws Exception {
        byte[] bytes = new byte[256 * 1024];
        for (int index = 0; index < bytes.length; index++) bytes[index] = (byte) (index % 251);
        Path destination = temporaryDirectory.resolve("measured-verification.gguf");
        FakeConnection response = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"measured\"");
        List<ResumableModelDownloader.ProgressEvent> events = new ArrayList<>();

        downloader(new QueueFactory(response), new ArrayList<>()).download(
                request(destination, bytes), policy(), events::add,
                new ResumableModelDownloader.CancellationHandle());

        List<ResumableModelDownloader.ProgressEvent> verification = events.stream()
                .filter(event -> event.getType() == ResumableModelDownloader.EventType.VERIFY)
                .collect(java.util.stream.Collectors.toList());
        assertTrue(verification.size() >= 3, "verification must report start, measured progress, and completion");
        assertEquals(0L, verification.get(0).getBytesDownloaded());
        assertEquals(bytes.length, verification.get(verification.size() - 1).getBytesDownloaded());
        assertTrue(verification.stream().allMatch(event -> event.getTotalBytes() == bytes.length));
        assertTrue(verification.stream().anyMatch(event ->
                event.getBytesDownloaded() > 0 && event.getBytesDownloaded() < bytes.length));
        long previous = -1L;
        for (ResumableModelDownloader.ProgressEvent event : verification) {
            assertTrue(event.getBytesDownloaded() >= previous, "verified bytes must never move backwards");
            previous = event.getBytesDownloaded();
        }
    }

    @Test
    void transientReadFailureRetriesAndResumesWithValidator() throws Exception {
        byte[] bytes = "validator-backed model bytes".getBytes(StandardCharsets.UTF_8);
        int prefixLength = 9;
        FakeConnection interrupted = response(200, bytes.length,
                failingAfter(bytes, prefixLength), "ETag", "\"immutable\"");
        FakeConnection resumed = response(206, bytes.length - prefixLength,
                new ByteArrayInputStream(Arrays.copyOfRange(bytes, prefixLength, bytes.length)),
                "ETag", "\"immutable\"");
        resumed.header("Content-Range", "bytes " + prefixLength + "-" + (bytes.length - 1) + "/" + bytes.length);
        QueueFactory factory = new QueueFactory(interrupted, resumed);
        List<ResumableModelDownloader.ProgressEvent> events = new ArrayList<>();
        List<Long> sleeps = new ArrayList<>();
        ResumableModelDownloader downloader = downloader(factory, sleeps);
        Path destination = temporaryDirectory.resolve("model.gguf");

        ResumableModelDownloader.DownloadResult result = downloader.download(
                request(destination, bytes), policy(), events::add,
                new ResumableModelDownloader.CancellationHandle());

        assertArrayEquals(bytes, Files.readAllBytes(destination));
        assertEquals(2, result.getAttempts());
        assertTrue(result.isResumed());
        assertEquals("bytes=" + prefixLength + "-", resumed.request("Range"));
        assertEquals("\"immutable\"", resumed.request("If-Range"));
        assertEquals(List.of(1_000L), sleeps);
        ResumableModelDownloader.ProgressEvent retry = events.stream()
                .filter(e -> e.getType() == ResumableModelDownloader.EventType.RETRY)
                .findFirst().orElseThrow();
        assertEquals(prefixLength, retry.getBytesDownloaded());
        assertEquals(bytes.length, retry.getTotalBytes());
        assertEquals(1_000L, retry.getDelayMillis());
        assertTrue(events.stream().anyMatch(e -> e.getType() == ResumableModelDownloader.EventType.RESUME));
        assertTrue(events.stream().anyMatch(e -> e.getType() == ResumableModelDownloader.EventType.VERIFY));
    }

    @Test
    void knownLengthCompletesWithoutWaitingForConnectionEof() throws Exception {
        byte[] bytes = "complete body on a persistent CDN connection".getBytes(StandardCharsets.UTF_8);
        FakeConnection persistent = response(200, bytes.length,
                failingAfter(bytes, bytes.length), "ETag", "\"persistent\"");
        QueueFactory factory = new QueueFactory(persistent);
        List<ResumableModelDownloader.ProgressEvent> events = new ArrayList<>();
        Path destination = temporaryDirectory.resolve("persistent.gguf");

        ResumableModelDownloader.DownloadResult result = downloader(factory, new ArrayList<>()).download(
                request(destination, bytes), policy(), events::add,
                new ResumableModelDownloader.CancellationHandle());

        assertArrayEquals(bytes, Files.readAllBytes(destination));
        assertEquals(1, result.getAttempts());
        assertEquals(1, factory.opened);
        assertTrue(events.stream().noneMatch(e -> e.getType() == ResumableModelDownloader.EventType.RETRY));
        assertTrue(events.stream().anyMatch(e -> e.getType() == ResumableModelDownloader.EventType.COMPLETE));
    }

    @Test
    void retryableStatusHonorsBoundedRetryAfter() throws Exception {
        byte[] bytes = "model".getBytes(StandardCharsets.UTF_8);
        FakeConnection throttled = response(503, 0, new ByteArrayInputStream(new byte[0]));
        throttled.header("Retry-After", "9999");
        FakeConnection success = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"v1\"");
        List<Long> sleeps = new ArrayList<>();
        ResumableModelDownloader downloader = downloader(new QueueFactory(throttled, success), sleeps);

        downloader.download(request(temporaryDirectory.resolve("model.gguf"), bytes), policy(), e -> { },
                new ResumableModelDownloader.CancellationHandle());

        assertEquals(List.of(Duration.ofMinutes(2).toMillis()), sleeps);
    }

    @Test
    void nonRetryableStatusFailsOnceAndPublishesNothing() {
        FakeConnection missing = response(404, 0, new ByteArrayInputStream(new byte[0]));
        QueueFactory factory = new QueueFactory(missing);
        Path destination = temporaryDirectory.resolve("missing.gguf");

        IOException failure = assertThrows(IOException.class, () -> downloader(factory, new ArrayList<>()).download(
                ResumableModelDownloader.DownloadRequest.builder(MODEL_URI, destination).maxBytes(1024).build(),
                policy(), e -> { }, new ResumableModelDownloader.CancellationHandle()));

        assertTrue(failure.getMessage().contains("HTTP 404"));
        assertEquals(1, factory.opened);
        assertFalse(Files.exists(destination));
    }

    @Test
    void legacyAndroidSidecarIsRecognizedForStoragePreflightAndResume() throws Exception {
        byte[] bytes = "legacy-compatible-model".getBytes(StandardCharsets.UTF_8);
        int prefixLength = 7;
        Path destination = temporaryDirectory.resolve("legacy.gguf");
        Path partial = destination.resolveSibling(destination.getFileName() + ".partial");
        Path metadata = destination.resolveSibling(destination.getFileName() + ".partial.metadata");
        Files.write(partial, Arrays.copyOf(bytes, prefixLength));
        String oldSidecar = "version=1\nidentity=" + MODEL_URI + "\nexpectedBytes=" + bytes.length
                + "\nexpectedSha256=" + sha256(bytes) + "\nvalidatorKind=etag\nvalidator=\\\"legacy\\\"\n";
        Files.write(metadata, oldSidecar.getBytes(StandardCharsets.ISO_8859_1));
        FakeConnection resumed = response(206, bytes.length - prefixLength,
                new ByteArrayInputStream(Arrays.copyOfRange(bytes, prefixLength, bytes.length)),
                "ETag", "\"legacy\"");
        resumed.header("Content-Range", "bytes " + prefixLength + "-" + (bytes.length - 1) + "/" + bytes.length);
        ResumableModelDownloader downloader = downloader(new QueueFactory(resumed), new ArrayList<>());
        ResumableModelDownloader.DownloadRequest request = request(destination, bytes);

        assertEquals(prefixLength, downloader.resumableBytes(request));
        downloader.download(request, policy(), e -> { }, new ResumableModelDownloader.CancellationHandle());

        assertArrayEquals(bytes, Files.readAllBytes(destination));
        assertEquals("bytes=" + prefixLength + "-", resumed.request("Range"));
        assertEquals("\"legacy\"", resumed.request("If-Range"));
    }

    @Test
    void malformedResumeResponseIsDiscardedThenSafelyRestarted() throws Exception {
        byte[] bytes = "fresh complete representation".getBytes(StandardCharsets.UTF_8);
        int prefixLength = 5;
        Path destination = temporaryDirectory.resolve("restart.gguf");
        preparePartial(destination, bytes, prefixLength, "\"v1\"");
        FakeConnection malformed = response(206, bytes.length - prefixLength,
                new ByteArrayInputStream(Arrays.copyOfRange(bytes, prefixLength, bytes.length)),
                "ETag", "\"v1\"");
        malformed.header("Content-Range", "bytes 0-1/" + bytes.length);
        FakeConnection fresh = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"v1\"");

        downloader(new QueueFactory(malformed, fresh), new ArrayList<>()).download(
                request(destination, bytes), policy(), e -> { },
                new ResumableModelDownloader.CancellationHandle());

        assertArrayEquals(bytes, Files.readAllBytes(destination));
        assertNull(fresh.request("Range"));
    }

    @Test
    void checksumMismatchPoisonsPartialWithoutRetryOrPublish() {
        byte[] bytes = "wrong bytes".getBytes(StandardCharsets.UTF_8);
        Path destination = temporaryDirectory.resolve("bad.gguf");
        FakeConnection response = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"bad\"");
        QueueFactory factory = new QueueFactory(response);
        String expectedDigest = sha256("expected".getBytes(StandardCharsets.UTF_8));
        String actualDigest = sha256(bytes);
        ResumableModelDownloader.DownloadRequest request = ResumableModelDownloader.DownloadRequest
                .builder(MODEL_URI, destination).maxBytes(1024).expectedLength(bytes.length)
                .expectedSha256(expectedDigest).build();

        IOException failure = assertThrows(IOException.class, () -> downloader(factory, new ArrayList<>()).download(
                request, policy(), e -> { }, new ResumableModelDownloader.CancellationHandle()));

        assertTrue(failure.getMessage().contains("Model verification failed: SHA-256 mismatch"));
        assertTrue(failure.getMessage().contains("expected " + expectedDigest));
        assertTrue(failure.getMessage().contains("actual " + actualDigest));
        assertTrue(failure.getMessage().contains(bytes.length + " bytes"));
        assertEquals(1, factory.opened);
        assertFalse(Files.exists(destination));
        assertFalse(Files.exists(destination.resolveSibling("bad.gguf.partial")));
        assertFalse(Files.exists(destination.resolveSibling("bad.gguf.partial.metadata")));
    }

    @Test
    void cancellationDisconnectsAndSafePartialCanBeRetried() throws Exception {
        byte[] bytes = new byte[160 * 1024];
        for (int index = 0; index < bytes.length; index++) bytes[index] = (byte) (index % 251);
        int prefixLength = 64 * 1024;
        Path destination = temporaryDirectory.resolve("cancel.gguf");
        ResumableModelDownloader.CancellationHandle cancellation = new ResumableModelDownloader.CancellationHandle();
        FakeConnection first = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"cancel\"");
        ResumableModelDownloader firstDownloader = downloader(new QueueFactory(first), new ArrayList<>());

        assertThrows(ResumableModelDownloader.DownloadCancelledException.class, () -> firstDownloader.download(
                request(destination, bytes), policy(), event -> {
                    if (event.getType() == ResumableModelDownloader.EventType.PROGRESS) cancellation.cancel();
                }, cancellation));
        assertTrue(first.disconnected);
        assertFalse(Files.exists(destination));

        FakeConnection retry = response(206, bytes.length - prefixLength,
                new ByteArrayInputStream(Arrays.copyOfRange(bytes, prefixLength, bytes.length)),
                "ETag", "\"cancel\"");
        retry.header("Content-Range", "bytes " + prefixLength + "-" + (bytes.length - 1) + "/" + bytes.length);
        downloader(new QueueFactory(retry), new ArrayList<>()).download(
                request(destination, bytes), policy(), e -> { },
                new ResumableModelDownloader.CancellationHandle());
        assertArrayEquals(bytes, Files.readAllBytes(destination));
        assertEquals("bytes=" + prefixLength + "-", retry.request("Range"));
    }

    @Test
    void cancellationDuringVerificationLeavesCompletePartialThatPublishesWithoutAnotherRequest() throws Exception {
        byte[] bytes = "complete partial awaiting verification".getBytes(StandardCharsets.UTF_8);
        Path destination = temporaryDirectory.resolve("complete-partial.gguf");
        ResumableModelDownloader.CancellationHandle cancellation =
                new ResumableModelDownloader.CancellationHandle();
        FakeConnection response = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"complete\"");

        assertThrows(ResumableModelDownloader.DownloadCancelledException.class, () ->
                downloader(new QueueFactory(response), new ArrayList<>()).download(
                        request(destination, bytes), policy(), event -> {
                            if (event.getType() == ResumableModelDownloader.EventType.VERIFY) {
                                cancellation.cancel();
                            }
                        }, cancellation));
        assertFalse(Files.exists(destination));
        assertEquals(bytes.length, Files.size(destination.resolveSibling("complete-partial.gguf.partial")));

        QueueFactory noNetwork = new QueueFactory();
        List<ResumableModelDownloader.ProgressEvent> recoveredEvents = new ArrayList<>();
        ResumableModelDownloader.DownloadResult recovered = downloader(noNetwork, new ArrayList<>()).download(
                request(destination, bytes), policy(), recoveredEvents::add,
                new ResumableModelDownloader.CancellationHandle());

        assertEquals(0, noNetwork.opened);
        assertTrue(recovered.isResumed());
        assertEquals(ResumableModelDownloader.EventType.VERIFY, recoveredEvents.get(0).getType());
        assertTrue(recoveredEvents.stream().noneMatch(event ->
                event.getType() == ResumableModelDownloader.EventType.ATTEMPT));
        assertArrayEquals(bytes, Files.readAllBytes(destination));
    }

    @Test
    void weakEtagWithoutDateNeverAuthorizesRangeAppend() throws Exception {
        byte[] bytes = "weak validators require a clean restart".getBytes(StandardCharsets.UTF_8);
        int prefixLength = 8;
        FakeConnection interrupted = response(200, bytes.length,
                failingAfter(bytes, prefixLength), "ETag", "W/\"weak\"");
        FakeConnection restarted = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"strong\"");
        QueueFactory factory = new QueueFactory(interrupted, restarted);
        Path destination = temporaryDirectory.resolve("weak-etag.gguf");

        downloader(factory, new ArrayList<>()).download(request(destination, bytes), policy(),
                event -> { }, new ResumableModelDownloader.CancellationHandle());

        assertNull(restarted.request("Range"));
        assertNull(restarted.request("If-Range"));
        assertArrayEquals(bytes, Files.readAllBytes(destination));
    }

    @Test
    void cancellationTriggeredByConnectionFailureIsReportedAsCancellationOnFinalAttempt() {
        ResumableModelDownloader.CancellationHandle cancellation =
                new ResumableModelDownloader.CancellationHandle();
        ResumableModelDownloader downloader = new ResumableModelDownloader(uri -> {
            cancellation.cancel();
            throw new IOException("socket closed by disconnect");
        }, new SystemClock(), (millis, handle) -> { }, (delay, attempt) -> delay);
        ResumableModelDownloader.DownloadPolicy oneAttempt =
                ResumableModelDownloader.DownloadPolicy.builder().maxAttempts(1).build();

        assertThrows(ResumableModelDownloader.DownloadCancelledException.class, () -> downloader.download(
                ResumableModelDownloader.DownloadRequest.builder(
                        MODEL_URI, temporaryDirectory.resolve("cancel-race.gguf")).maxBytes(1024).build(),
                oneAttempt, event -> { }, cancellation));
    }

    @Test
    void verificationFailureCannotReenterTheTransferRetryLoop() throws Exception {
        byte[] bytes = "one transfer followed by one verification".getBytes(StandardCharsets.UTF_8);
        Path destination = temporaryDirectory.resolve("one-shot-verification.gguf");
        Path partial = destination.resolveSibling("one-shot-verification.gguf.partial");
        QueueFactory factory = new QueueFactory(response(200, bytes.length,
                new ByteArrayInputStream(bytes), "ETag", "\"one-shot\""));
        List<ResumableModelDownloader.ProgressEvent> events = new ArrayList<>();
        AtomicBoolean removed = new AtomicBoolean();

        assertThrows(IOException.class, () -> downloader(factory, new ArrayList<>()).download(
                request(destination, bytes), policy(), event -> {
                    events.add(event);
                    if (event.getType() == ResumableModelDownloader.EventType.VERIFY
                            && removed.compareAndSet(false, true)) {
                        try {
                            Files.delete(partial);
                        } catch (IOException failure) {
                            throw new AssertionError(failure);
                        }
                    }
                }, new ResumableModelDownloader.CancellationHandle()));

        assertEquals(1, factory.opened, "verification must never open another HTTP connection");
        assertTrue(events.stream().anyMatch(event ->
                event.getType() == ResumableModelDownloader.EventType.VERIFY));
        assertTrue(events.stream().noneMatch(event ->
                event.getType() == ResumableModelDownloader.EventType.RETRY));
        assertFalse(Files.exists(destination));
    }

    @Test
    void metadataCleanupFailureAfterPublishCannotRestartDownload() throws Exception {
        byte[] bytes = "published model survives sidecar cleanup".getBytes(StandardCharsets.UTF_8);
        Path destination = temporaryDirectory.resolve("cleanup-failure.gguf");
        Path metadata = destination.resolveSibling("cleanup-failure.gguf.partial.metadata");
        QueueFactory factory = new QueueFactory(response(200, bytes.length,
                new ByteArrayInputStream(bytes), "ETag", "\"cleanup\""));
        List<ResumableModelDownloader.ProgressEvent> events = new ArrayList<>();
        AtomicBoolean obstructed = new AtomicBoolean();

        ResumableModelDownloader.DownloadResult result =
                downloader(factory, new ArrayList<>()).download(
                        request(destination, bytes), policy(), event -> {
                            events.add(event);
                            if (event.getType() == ResumableModelDownloader.EventType.VERIFY
                                    && event.getBytesDownloaded() == bytes.length
                                    && obstructed.compareAndSet(false, true)) {
                                try {
                                    Files.delete(metadata);
                                    Files.createDirectory(metadata);
                                    Files.write(metadata.resolve("keep"), new byte[] { 1 });
                                } catch (IOException failure) {
                                    throw new AssertionError(failure);
                                }
                            }
                        }, new ResumableModelDownloader.CancellationHandle());

        assertEquals(destination, result.getPath());
        assertArrayEquals(bytes, Files.readAllBytes(destination));
        assertEquals(1, factory.opened, "post-publish cleanup must never reopen HTTP");
        assertTrue(events.stream().noneMatch(event ->
                event.getType() == ResumableModelDownloader.EventType.RETRY));
        assertTrue(events.stream().anyMatch(event ->
                event.getType() == ResumableModelDownloader.EventType.COMPLETE
                        && event.getMessage().contains("cleanup deferred")));
    }

    @Test
    void verifiedModelPublishesOnAFileSystemWithoutHardLinkSupport() throws Exception {
        byte[] bytes = "portable Android publication".getBytes(StandardCharsets.UTF_8);
        Path archive = temporaryDirectory.resolve("app-storage.zip");
        URI archiveUri = URI.create("jar:" + archive.toUri());
        List<ResumableModelDownloader.ProgressEvent> events = new ArrayList<>();
        FakeConnection response = response(200, bytes.length, new ByteArrayInputStream(bytes));

        try (FileSystem appStorage = FileSystems.newFileSystem(
                archiveUri, java.util.Collections.singletonMap("create", "true"))) {
            Path destination = appStorage.getPath("/model.gguf");
            ResumableModelDownloader.DownloadResult result =
                    downloader(new QueueFactory(response), new ArrayList<>()).download(
                            request(destination, bytes), policy(), events::add,
                            new ResumableModelDownloader.CancellationHandle());

            assertEquals(destination, result.getPath());
            assertArrayEquals(bytes, Files.readAllBytes(destination));
            assertFalse(Files.exists(destination.resolveSibling("model.gguf.partial")));
            assertTrue(events.stream().anyMatch(event ->
                    event.getType() == ResumableModelDownloader.EventType.COMPLETE));
            assertTrue(events.stream().noneMatch(event ->
                    event.getType() == ResumableModelDownloader.EventType.RETRY));
        }
    }

    @Test
    void atomicPublishNeverOverwritesAConcurrentlyCreatedDestination() throws Exception {
        byte[] bytes = "downloaded candidate".getBytes(StandardCharsets.UTF_8);
        byte[] concurrent = "concurrent owner".getBytes(StandardCharsets.UTF_8);
        Path destination = temporaryDirectory.resolve("publish-race.gguf");
        FakeConnection response = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"publish\"");

        assertThrows(IOException.class, () -> downloader(new QueueFactory(response), new ArrayList<>()).download(
                request(destination, bytes), policy(), event -> {
                    if (event.getType() == ResumableModelDownloader.EventType.VERIFY) {
                        try {
                            Files.write(destination, concurrent);
                        } catch (IOException failure) {
                            throw new AssertionError(failure);
                        }
                    }
                }, new ResumableModelDownloader.CancellationHandle()));

        assertArrayEquals(concurrent, Files.readAllBytes(destination));
    }

    @Test
    void zeroAttemptTimeoutAllowsContinuousProgressBeyondAnyWallClockCap() throws Exception {
        byte[] bytes = new byte[128 * 1024];
        for (int index = 0; index < bytes.length; index++) bytes[index] = (byte) (index % 251);
        AdvancingClock clock = new AdvancingClock();
        ByteArrayInputStream delegate = new ByteArrayInputStream(bytes);
        InputStream continuouslyProgressing = new InputStream() {
            @Override public int read() {
                clock.advance(Duration.ofHours(13));
                return delegate.read();
            }
            @Override public int read(byte[] buffer, int offset, int length) {
                clock.advance(Duration.ofHours(13));
                return delegate.read(buffer, offset, Math.min(length, 8 * 1024));
            }
        };
        FakeConnection connection = response(200, bytes.length, continuouslyProgressing,
                "ETag", "\"slow-progress\"");
        ResumableModelDownloader downloader = new ResumableModelDownloader(
                new QueueFactory(connection), clock, (millis, handle) -> { },
                (delay, attempt) -> delay);
        ResumableModelDownloader.DownloadPolicy noWallClockCap =
                ResumableModelDownloader.DownloadPolicy.builder()
                        .maxAttempts(1)
                        .connectTimeout(Duration.ofSeconds(30))
                        .readIdleTimeout(Duration.ofMinutes(10))
                        .attemptTimeout(Duration.ZERO)
                        .progressInterval(Duration.ZERO)
                        .build();
        Path destination = temporaryDirectory.resolve("slow-progress.gguf");

        downloader.download(request(destination, bytes), noWallClockCap, event -> { },
                new ResumableModelDownloader.CancellationHandle());

        assertArrayEquals(bytes, Files.readAllBytes(destination));
        assertTrue(clock.nanoTime() > Duration.ofHours(12).toNanos());
        assertEquals(Duration.ofMinutes(10).toMillis(), connection.getReadTimeout());
    }

    @Test
    void configuredTimeoutsReachEveryConnection() throws Exception {
        byte[] bytes = "timeouts".getBytes(StandardCharsets.UTF_8);
        FakeConnection connection = response(200, bytes.length, new ByteArrayInputStream(bytes),
                "ETag", "\"timeouts\"");
        ResumableModelDownloader.DownloadPolicy policy = ResumableModelDownloader.DownloadPolicy.builder()
                .maxAttempts(1).connectTimeout(Duration.ofSeconds(17))
                .readIdleTimeout(Duration.ofMinutes(7)).attemptTimeout(Duration.ofHours(1)).build();

        downloader(new QueueFactory(connection), new ArrayList<>()).download(
                request(temporaryDirectory.resolve("timeouts.gguf"), bytes), policy, e -> { },
                new ResumableModelDownloader.CancellationHandle());

        assertEquals(Duration.ofSeconds(17).toMillis(), connection.getConnectTimeout());
        assertTrue(connection.getReadTimeout() > 0);
        assertTrue(connection.getReadTimeout() <= Duration.ofMinutes(7).toMillis());
    }

    private static ResumableModelDownloader.DownloadRequest request(Path destination, byte[] bytes) {
        return ResumableModelDownloader.DownloadRequest.builder(MODEL_URI, destination)
                .maxBytes(Math.max(1024L, bytes.length)).expectedLength(bytes.length)
                .expectedSha256(sha256(bytes))
                .header("Accept", "application/octet-stream").build();
    }

    private static ResumableModelDownloader.DownloadPolicy policy() {
        return ResumableModelDownloader.DownloadPolicy.builder().progressInterval(Duration.ZERO).build();
    }

    private static ResumableModelDownloader downloader(QueueFactory factory, List<Long> sleeps) {
        return new ResumableModelDownloader(factory, new SystemClock(),
                (millis, cancellation) -> sleeps.add(millis), (delay, attempt) -> delay);
    }

    private void preparePartial(Path destination, byte[] bytes, int length, String validator) throws IOException {
        Files.write(destination.resolveSibling(destination.getFileName() + ".partial"),
                Arrays.copyOf(bytes, length));
        String metadata = "version=1\nuri=" + MODEL_URI + "\netag=" + validator.replace("\\", "\\\\")
                + "\nlastModified=\ntotalBytes=" + bytes.length + "\nexpectedSha256=" + sha256(bytes) + "\n";
        Files.write(destination.resolveSibling(destination.getFileName() + ".partial.metadata"),
                metadata.getBytes(StandardCharsets.ISO_8859_1));
    }

    private static InputStream failingAfter(byte[] bytes, int prefixLength) {
        return new InputStream() {
            int position;
            @Override public int read() throws IOException {
                if (position >= prefixLength) throw new SocketTimeoutException("simulated idle timeout");
                return bytes[position++] & 0xff;
            }
            @Override public int read(byte[] buffer, int offset, int length) throws IOException {
                if (position >= prefixLength) throw new SocketTimeoutException("simulated idle timeout");
                int count = Math.min(length, prefixLength - position);
                System.arraycopy(bytes, position, buffer, offset, count);
                position += count;
                return count;
            }
        };
    }

    private static FakeConnection response(int status, long length, InputStream input, String... headers) {
        FakeConnection connection;
        try {
            connection = new FakeConnection(MODEL_URI.toURL(), status, length, input);
        } catch (Exception failure) {
            throw new AssertionError(failure);
        }
        for (int index = 0; index < headers.length; index += 2) {
            connection.header(headers[index], headers[index + 1]);
        }
        return connection;
    }

    private static String sha256(byte[] bytes) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
            StringBuilder output = new StringBuilder(64);
            for (byte value : digest) output.append(String.format("%02x", value & 0xff));
            return output.toString();
        } catch (Exception failure) {
            throw new AssertionError(failure);
        }
    }

    private static final class QueueFactory implements ResumableModelDownloader.ConnectionFactory {
        private final Deque<FakeConnection> connections = new ArrayDeque<>();
        private int opened;
        private QueueFactory(FakeConnection... connections) { this.connections.addAll(Arrays.asList(connections)); }
        @Override public HttpURLConnection open(URI uri) throws IOException {
            opened++;
            FakeConnection connection = connections.pollFirst();
            if (connection == null) throw new IOException("No fake connection remaining for " + uri);
            return connection;
        }
    }

    private static final class AdvancingClock implements ResumableModelDownloader.MonotonicClock {
        private long nanos;
        private void advance(Duration duration) { nanos += duration.toNanos(); }
        @Override public long nanoTime() { return nanos; }
        @Override public long currentTimeMillis() { return nanos / 1_000_000L; }
    }

    private static final class SystemClock implements ResumableModelDownloader.MonotonicClock {
        @Override public long nanoTime() { return System.nanoTime(); }
        @Override public long currentTimeMillis() { return System.currentTimeMillis(); }
    }

    private static final class FakeConnection extends HttpURLConnection {
        private final int status;
        private final long length;
        private final InputStream input;
        private final Map<String, String> headers = new HashMap<>();
        private final Map<String, String> requests = new HashMap<>();
        private boolean disconnected;
        private FakeConnection(URL url, int status, long length, InputStream input) {
            super(url);
            this.status = status;
            this.length = length;
            this.input = input;
        }
        private void header(String name, String value) { headers.put(name, value); }
        private String request(String name) { return requests.get(name); }
        @Override public void connect() { }
        @Override public void disconnect() { disconnected = true; }
        @Override public boolean usingProxy() { return false; }
        @Override public int getResponseCode() { return status; }
        @Override public InputStream getInputStream() { return input; }
        @Override public long getContentLengthLong() { return length; }
        @Override public String getHeaderField(String name) { return headers.get(name); }
        @Override public void setRequestProperty(String key, String value) {
            super.setRequestProperty(key, value);
            requests.put(key, value);
        }
    }
}
