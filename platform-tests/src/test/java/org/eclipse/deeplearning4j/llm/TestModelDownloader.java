/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.llm;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.eclipse.deeplearning4j.model.download.ModelDownloader;
import org.eclipse.deeplearning4j.model.download.ModelDownloader.DownloadResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.common.config.ND4JSystemProperties;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/** No ND4J initialization, model downloads or external network: all bodies come from loopback. */
@Timeout(30)
public class TestModelDownloader {
    private static final String NAME = "checkpoint.bin";
    private static final String ETAG = "\"checkpoint-v1\"";
    private static final int PREFIX = 4096;
    @TempDir
    Path directory;
    private HttpServer server;
    private String base;
    private final byte[] data = new byte[32 * 1024];

    @BeforeEach
    void startServer() throws IOException {
        for (int i = 0; i < data.length; i++) {
            data[i] = (byte) (i * 31);
        }
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        base = "http://127.0.0.1:" + server.getAddress().getPort();
        server.start();
    }

    @AfterEach
    void stopServer() {
        if (server != null) server.stop(0);
    }

    @Test
    void fullTransferAndLegacyCacheHit() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        server.createContext("/file", exchange -> {
            calls.incrementAndGet();
            send(exchange, 200, ETAG, null, data);
        });
        DownloadResult first = download("/file");
        assertTrue(first.isDownloadedNow());
        assertEquals(data.length, first.getFileSizeBytes());
        assertArrayEquals(data, Files.readAllBytes(target()));
        assertFalse(Files.exists(partial()));
        assertFalse(Files.exists(metadata()));
        // Cache semantics remain independent of the URL and sidecar existence.
        assertFalse(download("/does-not-exist").isDownloadedNow());
        assertEquals(1, calls.get());
    }

    @Test
    void resumesAcrossInvocationsWithExactValidatorAndSource() throws Exception {
        seed(base + "/file", ETAG, data.length);
        AtomicReference<String> range = new AtomicReference<>();
        AtomicReference<String> validator = new AtomicReference<>();
        server.createContext("/file", exchange -> {
            range.set(exchange.getRequestHeaders().getFirst("Range"));
            validator.set(exchange.getRequestHeaders().getFirst("If-Range"));
            send(exchange, 206, ETAG, remainderRange(), Arrays.copyOfRange(data, PREFIX, data.length));
        });
        download("/file");
        assertEquals("bytes=" + PREFIX + "-", range.get());
        assertEquals(ETAG, validator.get());
        assertArrayEquals(data, Files.readAllBytes(target()));
    }

    @Test
    void ignoredRangeRestartsInsteadOfConcatenating() throws Exception {
        seed(base + "/file", ETAG, data.length);
        AtomicReference<String> range = new AtomicReference<>();
        byte[] replacement = new byte[100];
        Arrays.fill(replacement, (byte) 7);
        server.createContext("/file", exchange -> {
            range.set(exchange.getRequestHeaders().getFirst("Range"));
            send(exchange, 200, "\"replacement\"", null, replacement);
        });
        download("/file");
        assertEquals("bytes=" + PREFIX + "-", range.get());
        assertArrayEquals(replacement, Files.readAllBytes(target()));
    }

    @ParameterizedTest
    @ValueSource(strings = {"source", "weak", "missing", "malformed", "oversized"})
    void unsafeMetadataNeverAppends(String variant) throws Exception {
        seed("source".equals(variant) ? base + "/other-revision" : base + "/file",
                "weak".equals(variant) ? "W/" + ETAG : ETAG, data.length);
        if ("missing".equals(variant)) {
            Files.delete(metadata());
        } else if ("malformed".equals(variant)) {
            Files.writeString(metadata(), "length=broken\n");
        } else if ("oversized".equals(variant)) {
            Files.write(partial(), new byte[data.length + 1]);
        }
        AtomicReference<String> range = new AtomicReference<>();
        server.createContext("/file", exchange -> {
            range.set(exchange.getRequestHeaders().getFirst("Range"));
            send(exchange, 200, ETAG, null, data);
        });
        download("/file");
        assertNull(range.get());
        assertArrayEquals(data, Files.readAllBytes(target()));
    }

    @ParameterizedTest
    @ValueSource(strings = {"start", "end", "total", "etag", "missing-etag", "length", "invalid", "416"})
    void invalidResumeResponsePreservesPrefixAndDoesNotPromote(String variant) throws Exception {
        seed(base + "/file", ETAG, data.length);
        AtomicInteger calls = new AtomicInteger();
        server.createContext("/file", exchange -> {
            calls.incrementAndGet();
            String range = remainderRange();
            String etag = ETAG;
            byte[] remainder = Arrays.copyOfRange(data, PREFIX, data.length);
            switch (variant) {
                case "start": range = "bytes 0-32767/32768"; break;
                case "end": range = "bytes 4096-32766/32768"; break;
                case "total": range = "bytes 4096-32768/32769"; break;
                case "etag": etag = "\"different\""; break;
                case "missing-etag": etag = null; break;
                case "length": remainder = new byte[12]; break;
                case "invalid": range = "bytes nonsense"; break;
                default: break;
            }
            send(exchange, "416".equals(variant) ? 416 : 206, etag, range, remainder);
        });
        assertThrows(IOException.class, () -> download("/file"));
        assertEquals(1, calls.get(), "Protocol/validator errors must not be retried");
        assertFalse(Files.exists(target()));
        assertArrayEquals(Arrays.copyOf(data, PREFIX), Files.readAllBytes(partial()));
        assertTrue(Files.exists(metadata()));
    }

    @Test
    void truncatedTransferRetriesFromSavedOffset() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        AtomicReference<String> range = new AtomicReference<>();
        server.createContext("/file", exchange -> {
            if (calls.incrementAndGet() == 1) {
                truncate(exchange);
            } else {
                range.set(exchange.getRequestHeaders().getFirst("Range"));
                send(exchange, 206, ETAG, remainderRange(), Arrays.copyOfRange(data, PREFIX, data.length));
            }
        });
        download("/file");
        assertEquals(2, calls.get());
        assertEquals("bytes=" + PREFIX + "-", range.get());
        assertArrayEquals(data, Files.readAllBytes(target()));
    }

    @Test
    void exhaustedTransientFailuresPreservePartialForNextCall() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        AtomicBoolean recover = new AtomicBoolean();
        server.createContext("/file", exchange -> {
            int call = calls.incrementAndGet();
            if (call == 1) {
                truncate(exchange);
            } else if (!recover.get()) {
                send(exchange, 503, null, null, new byte[]{1});
            } else {
                send(exchange, 206, ETAG, remainderRange(), Arrays.copyOfRange(data, PREFIX, data.length));
            }
        });
        assertThrows(IOException.class, () -> download("/file"));
        assertEquals(4, calls.get());
        assertFalse(Files.exists(target()));
        assertArrayEquals(Arrays.copyOf(data, PREFIX), Files.readAllBytes(partial()));
        Properties state = new Properties();
        try (InputStream in = Files.newInputStream(metadata())) {
            state.load(in);
        }
        assertEquals(base + "/file", state.getProperty("source"));
        assertEquals(ETAG, state.getProperty("etag"));
        assertEquals(Integer.toString(data.length), state.getProperty("length"));
        recover.set(true);
        download("/file");
        assertArrayEquals(data, Files.readAllBytes(target()));
    }

    @Test
    void cancelledRequestPreservesSeedAndDoesNotMakeNetworkRequest() throws Exception {
        seed(base + "/file", ETAG, data.length);
        AtomicInteger calls = new AtomicInteger();
        server.createContext("/file", exchange -> {
            calls.incrementAndGet();
            send(exchange, 200, ETAG, null, data);
        });
        Thread.currentThread().interrupt();
        try {
            assertThrows(InterruptedIOException.class, () -> download("/file"));
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted(); // Do not contaminate JUnit's worker thread.
        }
        assertEquals(0, calls.get());
        assertFalse(Files.exists(target()));
        assertArrayEquals(Arrays.copyOf(data, PREFIX), Files.readAllBytes(partial()));
    }

    @ParameterizedTest
    @ValueSource(ints = {401, 403, 404})
    void permanentOriginErrorsAreNotRetried(int status) throws Exception {
        seed(base + "/file", ETAG, data.length);
        AtomicInteger calls = new AtomicInteger();
        server.createContext("/file", exchange -> {
            calls.incrementAndGet();
            send(exchange, status, null, null, new byte[]{1});
        });
        assertThrows(IOException.class, () -> download("/file"));
        assertEquals(1, calls.get());
        assertFalse(Files.exists(target()));
        assertEquals(PREFIX, Files.size(partial()));
    }

    @Test
    void overlongRangeBodyCannotBePromoted() throws Exception {
        seed(base + "/file", ETAG, data.length);
        server.createContext("/file", exchange -> {
            exchange.getResponseHeaders().set("ETag", ETAG);
            exchange.getResponseHeaders().set("Content-Range", remainderRange());
            exchange.sendResponseHeaders(206, 0); // Chunk framing cannot hide extra bytes behind Content-Length.
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(new byte[data.length - PREFIX + 1]);
            } finally {
                exchange.close();
            }
        });
        assertThrows(IOException.class, () -> download("/file"));
        assertFalse(Files.exists(target()));
        assertArrayEquals(Arrays.copyOf(data, PREFIX), Files.readAllBytes(partial()));
    }

    @Test
    void unknownLengthCannotBePromoted() throws Exception {
        server.createContext("/file", exchange -> {
            exchange.sendResponseHeaders(200, 0); // Chunked: no complete byte count to verify.
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(data);
            } finally {
                exchange.close();
            }
        });
        assertThrows(IOException.class, () -> download("/file"));
        assertFalse(Files.exists(target()));
    }

    @Test
    void relativeRedirectWorksAndLoopsAreBounded() throws Exception {
        server.createContext("/relative", exchange -> redirect(exchange, "file"));
        server.createContext("/file", exchange -> send(exchange, 200, ETAG, null, data));
        download("/relative");
        assertArrayEquals(data, Files.readAllBytes(target()));
        AtomicInteger calls = new AtomicInteger();
        server.createContext("/loop", exchange -> {
            calls.incrementAndGet();
            redirect(exchange, "/loop");
        });
        assertThrows(IOException.class, () -> ModelDownloader.download(base + "/loop", "loop.bin", directory.toFile()));
        assertEquals(1, calls.get());
        AtomicInteger chainCalls = new AtomicInteger();
        server.createContext("/chain", exchange -> {
            chainCalls.incrementAndGet();
            String query = exchange.getRequestURI().getQuery();
            int hop = query == null ? 0 : Integer.parseInt(query);
            redirect(exchange, "/chain?" + (hop + 1));
        });
        assertThrows(IOException.class, () -> ModelDownloader.download(base + "/chain", "chain.bin", directory.toFile()));
        assertEquals(11, chainCalls.get());
    }

    @Test
    void expiredSignedRedirectRefreshesOriginalUrlAndKeepsRange() throws Exception {
        seed(base + "/pinned", ETAG, data.length);
        AtomicInteger resolutions = new AtomicInteger();
        AtomicReference<String> range = new AtomicReference<>();
        server.createContext("/pinned", exchange -> redirect(exchange,
                resolutions.incrementAndGet() == 1 ? "/expired" : "/fresh"));
        server.createContext("/expired", exchange -> send(exchange, 403, null, null, new byte[]{1}));
        server.createContext("/fresh", exchange -> {
            range.set(exchange.getRequestHeaders().getFirst("Range"));
            send(exchange, 206, ETAG, remainderRange(), Arrays.copyOfRange(data, PREFIX, data.length));
        });
        download("/pinned");
        assertEquals(2, resolutions.get());
        assertEquals("bytes=" + PREFIX + "-", range.get());
        assertArrayEquals(data, Files.readAllBytes(target()));
    }

    @Test
    void bearerNeverSentToLocalOriginOrRedirectTarget() throws Exception {
        String previous = System.getProperty(ND4JSystemProperties.HF_TOKEN);
        AtomicReference<String> originAuth = new AtomicReference<>();
        AtomicReference<String> targetAuth = new AtomicReference<>();
        try {
            System.setProperty(ND4JSystemProperties.HF_TOKEN, "test-secret-never-send");
            server.createContext("/origin", exchange -> {
                originAuth.set(exchange.getRequestHeaders().getFirst("Authorization"));
                redirect(exchange, "http://localhost:" + server.getAddress().getPort() + "/thirdparty");
            });
            server.createContext("/thirdparty", exchange -> {
                targetAuth.set(exchange.getRequestHeaders().getFirst("Authorization"));
                send(exchange, 200, ETAG, null, data);
            });
            download("/origin");
            assertNull(originAuth.get());
            assertNull(targetAuth.get());
            // Pure URL policy checks, including spoofed HF hosts, without DNS or network access.
            Method policy = ModelDownloader.class.getDeclaredMethod("isHuggingFaceOrigin", URL.class);
            policy.setAccessible(true);
            assertEquals(true, policy.invoke(null, new URL("https://huggingface.co/model")));
            for (String url : new String[]{"http://huggingface.co/model", "https://huggingface.co.evil/model",
                    "https://cdn.huggingface.co/model", "https://huggingface.co:8443/model",
                    "https://huggingface.co@evil/model", "https://user@huggingface.co/model"}) {
                assertEquals(false, policy.invoke(null, new URL(url)), url);
            }
        } finally {
            if (previous == null) {
                System.clearProperty(ND4JSystemProperties.HF_TOKEN);
            } else {
                System.setProperty(ND4JSystemProperties.HF_TOKEN, previous);
            }
        }
    }

    @Test
    void concurrentRequestsShareOneTransfer() throws Exception {
        AtomicInteger calls = new AtomicInteger();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        server.createContext("/file", exchange -> {
            calls.incrementAndGet();
            entered.countDown();
            try {
                if (!release.await(5, TimeUnit.SECONDS)) {
                    throw new IOException("Test release timed out");
                }
                send(exchange, 200, ETAG, null, data);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                exchange.close();
            }
        });
        ExecutorService workers = Executors.newFixedThreadPool(2);
        try {
            Future<DownloadResult> first = workers.submit(() -> download("/file"));
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            Future<DownloadResult> second = workers.submit(() -> download("/file"));
            release.countDown();
            assertTrue(first.get(10, TimeUnit.SECONDS).isDownloadedNow());
            assertFalse(second.get(10, TimeUnit.SECONDS).isDownloadedNow());
            assertEquals(1, calls.get());
            assertArrayEquals(data, Files.readAllBytes(target()));
        } finally {
            release.countDown();
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    private DownloadResult download(String path) throws IOException {
        return ModelDownloader.download(base + path, NAME, directory.toFile());
    }

    private Path target() { return directory.resolve(NAME); }
    private Path partial() { return directory.resolve(NAME + ".part"); }
    private Path metadata() { return directory.resolve(NAME + ".part.properties"); }
    private String remainderRange() { return "bytes " + PREFIX + "-" + (data.length - 1) + "/" + data.length; }

    private void seed(String source, String etag, long length) throws IOException {
        Files.write(partial(), Arrays.copyOf(data, PREFIX));
        Properties state = new Properties();
        state.setProperty("version", "1");
        state.setProperty("source", source);
        state.setProperty("etag", etag);
        state.setProperty("length", Long.toString(length));
        try (OutputStream out = Files.newOutputStream(metadata())) {
            state.store(out, "isolated test");
        }
    }

    private static void redirect(HttpExchange exchange, String location) throws IOException {
        exchange.getResponseHeaders().set("Location", location);
        exchange.sendResponseHeaders(302, -1);
        exchange.close();
    }

    private static void send(HttpExchange exchange, int status, String etag, String range, byte[] body)
            throws IOException {
        if (etag != null) exchange.getResponseHeaders().set("ETag", etag);
        if (range != null) exchange.getResponseHeaders().set("Content-Range", range);
        exchange.sendResponseHeaders(status, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        } finally {
            exchange.close();
        }
    }

    private void truncate(HttpExchange exchange) throws IOException {
        exchange.getResponseHeaders().set("Connection", "close");
        exchange.getResponseHeaders().set("ETag", ETAG);
        exchange.sendResponseHeaders(200, data.length);
        OutputStream out = exchange.getResponseBody();
        out.write(data, 0, PREFIX);
        out.flush();
        // Close the exchange first: it catches the fixed-length stream's short-body
        // error and closes the underlying connection. Closing out first marks that
        // stream closed before throwing, preventing exchange.close() from seeing the
        // error and leaving the client waiting on an open connection instead of EOF.
        exchange.close();
    }
}
