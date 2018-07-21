/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.api.storage.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.api.storage.StorageType;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import javax.xml.bind.DatatypeConverter;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Asynchronously post all updates to a remote UI that has remote listening enabled.<br>
 * Typically used with UIServer (don't forget to enable remote listener support - UIServer.getInstance().enableRemoteListener()
 *
 * @author Alex Black
 */
@Slf4j
public class RemoteUIStatsStorageRouter implements StatsStorageRouter {

    /**
     * Default path for posting data to the UI - i.e., http://localhost:9000/remoteReceive or similar
     */
    public static final String DEFAULT_PATH = "remoteReceive";
    /**
     * Default maximum number of (consecutive) retries on failure
     */
    public static final int DEFAULT_MAX_RETRIES = 10;
    /**
     * Base delay for retries
     */
    public static final long DEFAULT_BASE_RETR_DELAY_MS = 1000;
    /**
     * Default backoff multiplicative factor for retrying
     */
    public static final double DEFAULT_RETRY_BACKOFF_FACTOR = 2.0;

    private static final long MAX_SHUTDOWN_WARN_COUNT = 5;

    private final String USER_AGENT = "Mozilla/5.0";

    private URL url;
    private int maxRetryCount;
    private long retryDelayMS;
    private double retryBackoffFactor;

    private LinkedBlockingDeque<ToPost> queue = new LinkedBlockingDeque<>();

    private Thread postThread;

    private AtomicBoolean shutdown = new AtomicBoolean(false);
    private AtomicLong shutdownWarnCount = new AtomicLong(0);

    private static final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Create remote UI with defaults for all values except address
     *
     * @param address Address of the remote UI: for example, "http://localhost:9000"
     */
    public RemoteUIStatsStorageRouter(String address) {
        this(address, DEFAULT_MAX_RETRIES, DEFAULT_BASE_RETR_DELAY_MS, DEFAULT_RETRY_BACKOFF_FACTOR);
    }

    /**
     * @param address            Address of the remote UI: for example, "http://localhost:9000"
     * @param maxRetryCount      Maximum number of retries before failing. Set to -1 to always retry
     * @param retryDelayMS       Base delay before retrying, in milliseconds
     * @param retryBackoffFactor Backoff factor for retrying: 2.0 for example gives delays of 1000, 2000, 4000, 8000,
     *                           etc milliseconds, with a base retry delay of 1000
     */
    public RemoteUIStatsStorageRouter(String address, int maxRetryCount, long retryDelayMS, double retryBackoffFactor) {
        this(address, DEFAULT_PATH, maxRetryCount, retryDelayMS, retryBackoffFactor);
    }

    /**
     * @param address            Address of the remote UI: for example, "http://localhost:9000"
     * @param path               Path/endpoint to post to: for example "remoteReceive" -> added to path to become like
     *                           "http://localhost:9000/remoteReceive"
     * @param maxRetryCount      Maximum number of retries before failing. Set to -1 to always retry
     * @param retryDelayMS       Base delay before retrying, in milliseconds
     * @param retryBackoffFactor Backoff factor for retrying: 2.0 for example gives delays of 1000, 2000, 4000, 8000,
     *                           etc milliseconds, with a base retry delay of 1000
     */
    public RemoteUIStatsStorageRouter(String address, String path, int maxRetryCount, long retryDelayMS,
                    double retryBackoffFactor) {
        this.maxRetryCount = maxRetryCount;
        this.retryDelayMS = retryDelayMS;
        this.retryBackoffFactor = retryBackoffFactor;

        String url = address;
        if (path != null) {
            if (url.endsWith("/")) {
                url = url + path;
            } else {
                url = url + "/" + path;
            }
        }

        try {
            this.url = new URL(url);
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }

        postThread = new Thread(new PostRunnable());
        postThread.setDaemon(true);
        postThread.start();
    }

    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        putStorageMetaData(Collections.singleton(storageMetaData));
    }

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData) {
        if (shutdown.get()) {
            long count = shutdownWarnCount.getAndIncrement();
            if (count <= MAX_SHUTDOWN_WARN_COUNT) {
                log.warn("Info posted to RemoteUIStatsStorageRouter but router is shut down.");
            }
            if (count == MAX_SHUTDOWN_WARN_COUNT) {
                log.warn("RemoteUIStatsStorageRouter: Reached max shutdown warnings. No further warnings will be produced.");
            }
        } else {
            for (StorageMetaData m : storageMetaData) {
                queue.add(new ToPost(m, null, null));
            }
        }
    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        putStaticInfo(Collections.singletonList(staticInfo));
    }

    @Override
    public void putStaticInfo(Collection<? extends Persistable> staticInfo) {
        if (shutdown.get()) {
            long count = shutdownWarnCount.getAndIncrement();
            if (count <= MAX_SHUTDOWN_WARN_COUNT) {
                log.warn("Info posted to RemoteUIStatsStorageRouter but router is shut down.");
            }
            if (count == MAX_SHUTDOWN_WARN_COUNT) {
                log.warn("RemoteUIStatsStorageRouter: Reached max shutdown warnings. No further warnings will be produced.");
            }
        } else {
            for (Persistable p : staticInfo) {
                queue.add(new ToPost(null, p, null));
            }
        }
    }

    @Override
    public void putUpdate(Persistable update) {
        putUpdate(Collections.singleton(update));
    }

    @Override
    public void putUpdate(Collection<? extends Persistable> updates) {
        if (shutdown.get()) {
            long count = shutdownWarnCount.getAndIncrement();
            if (count <= MAX_SHUTDOWN_WARN_COUNT) {
                log.warn("Info posted to RemoteUIStatsStorageRouter but router is shut down.");
            }
            if (count == MAX_SHUTDOWN_WARN_COUNT) {
                log.warn("RemoteUIStatsStorageRouter: Reached max shutdown warnings. No further warnings will be produced.");
            }
        } else {
            for (Persistable p : updates) {
                queue.add(new ToPost(null, null, p));
            }
        }
    }

    @AllArgsConstructor
    @Data
    private static class ToPost {
        private final StorageMetaData meta;
        private final Persistable staticInfo;
        private final Persistable update;
    }

    //Runnable class for doing async posting
    private class PostRunnable implements Runnable {

        private int failureCount = 0;
        private long nextDelayMs = retryDelayMS;


        @Override
        public void run() {
            try {
                runHelper();
            } catch (Exception e) {
                log.error("Exception encountered in remote UI posting thread. Shutting down.", e);
                shutdown.set(true);
            }
        }

        private void runHelper() {

            while (!shutdown.get()) {

                List<ToPost> list = new ArrayList<>();
                ToPost t;
                try {
                    t = queue.take(); //Blocking operation
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    continue;
                }
                list.add(t);
                queue.drainTo(list); //Non-blocking

                int successCount = 0;
                for (ToPost toPost : list) {
                    boolean success;
                    try {
                        success = tryPost(toPost);
                    } catch (IOException e) {
                        failureCount++;
                        log.warn("Error posting to remote UI at {}, consecutive failure count = {}. Waiting {} ms before retrying",
                                        url, failureCount, nextDelayMs, e);
                        success = false;
                    }
                    if (!success) {
                        for (int i = list.size() - 1; i > successCount; i--) {
                            queue.addFirst(list.get(i)); //Add remaining back to be processed in original order
                        }
                        waitForRetry();
                        break;
                    } else {
                        successCount++;
                        failureCount = 0;
                        nextDelayMs = retryDelayMS;
                    }
                }
            }
        }

        private void waitForRetry() {
            if (maxRetryCount >= 0 && failureCount > maxRetryCount) {
                throw new RuntimeException("RemoteUIStatsStorageRouter: hit maximum consecutive failures("
                                + maxRetryCount + "). Shutting down remote router thread");
            } else {
                try {
                    Thread.sleep(nextDelayMs);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                nextDelayMs *= retryBackoffFactor;
            }
        }
    }


    private HttpURLConnection getConnection() throws IOException {
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("POST");
        connection.setRequestProperty("User-Agent", USER_AGENT);
        connection.setRequestProperty("Content-Type", "application/json");
        connection.setDoOutput(true);
        return connection;
    }

    private boolean tryPost(ToPost toPost) throws IOException {

        HttpURLConnection connection = getConnection();

        String className;
        byte[] asBytes;
        StorageType type;
        if (toPost.getMeta() != null) {
            StorageMetaData smd = toPost.getMeta();
            className = smd.getClass().getName();
            asBytes = smd.encode();
            type = StorageType.MetaData;
        } else if (toPost.getStaticInfo() != null) {
            Persistable p = toPost.getStaticInfo();
            className = p.getClass().getName();
            asBytes = p.encode();
            type = StorageType.StaticInfo;
        } else {
            Persistable p = toPost.getUpdate();
            className = p.getClass().getName();
            asBytes = p.encode();
            type = StorageType.Update;
        }

        String base64 = DatatypeConverter.printBase64Binary(asBytes);

        Map<String, String> jsonObj = new LinkedHashMap<>();
        jsonObj.put("type", type.name());
        jsonObj.put("class", className);
        jsonObj.put("data", base64);

        String str;
        try {
            str = objectMapper.writeValueAsString(jsonObj);
        } catch (Exception e) {
            throw new RuntimeException(e); //Should never get an exception from simple Map<String,String>
        }

        DataOutputStream dos = new DataOutputStream(connection.getOutputStream());
        dos.writeBytes(str);
        dos.flush();
        dos.close();

        try {
            int responseCode = connection.getResponseCode();

            if (responseCode != 200) {
                BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                String inputLine;
                StringBuilder response = new StringBuilder();

                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();

                log.warn("Error posting to remote UI - received response code {}\tContent: {}", response,
                                response.toString());

                return false;
            }
        } catch (IOException e) {
            String msg = e.getMessage();
            if (msg.contains("403 for URL")) {
                log.warn("Error posting to remote UI at {} (Response code: 403)."
                                + " Remote listener support is not enabled? use UIServer.getInstance().enableRemoteListener()",
                                url, e);
            } else {
                log.warn("Error posting to remote UI at {}", url, e);
            }

            return false;
        }

        return true;
    }
}
