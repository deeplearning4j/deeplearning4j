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
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Created by Alex on 10/11/2016.
 */
@Slf4j
public class RemoteUIStatsStorageRouter implements StatsStorageRouter {


    private final String USER_AGENT = "Mozilla/5.0";

    private URL url;
    private int maxRetryCount;
    private long retryDelayMS;
    private double retryBackoffFactor;

    private LinkedBlockingDeque<ToPost> queue = new LinkedBlockingDeque<>();

    private Thread postThread;

    private AtomicBoolean shutdown = new AtomicBoolean(false);

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public RemoteUIStatsStorageRouter(String url) throws Exception {
        this(url, 10, 1000, 2.0);
    }

    public RemoteUIStatsStorageRouter(String url, int maxRetryCount, long retryDelayMS, double retryBackoffFactor) {
        this.maxRetryCount = maxRetryCount;
        this.retryDelayMS = retryDelayMS;
        this.retryBackoffFactor = retryBackoffFactor;

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
        queue.add(new ToPost(storageMetaData, null, null));
    }

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData) {
        for (StorageMetaData m : storageMetaData) {
            putStorageMetaData(m);
        }
    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        queue.add(new ToPost(null, staticInfo, null));
    }

    @Override
    public void putStaticInfo(Collection<? extends Persistable> staticInfo) {
        for (Persistable p : staticInfo) {
            putStaticInfo(p);
        }
    }

    @Override
    public void putUpdate(Persistable update) {
        queue.add(new ToPost(null, null, update));
    }

    @Override
    public void putUpdate(Collection<? extends Persistable> updates) {
        for (Persistable p : updates) {
            putUpdate(p);
        }
    }

    @AllArgsConstructor
    @Data
    private static class ToPost {
        private final StorageMetaData meta;
        private final Persistable staticInfo;
        private final Persistable update;
    }

    private class PostRunnable implements Runnable {

        private int failureCount = 0;
        private long lastDelayMs = retryDelayMS;


        @Override
        public void run() {
            try {
                runHelper();
            } catch (Exception e) {
                e.printStackTrace();
                //TODO
            }
        }

        private void runHelper() {

            while (!shutdown.get()) {

                List<ToPost> list = new ArrayList<>();
                ToPost t;
                try {
                    t = queue.take();  //Blocking operation
                } catch (InterruptedException e) {
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
                        log.warn("Error posting to remote UI, failure count = {}", failureCount, e);
                        success = false;
                    }
                    if (!success) {
                        for (int i = list.size() - 1; i > successCount; i--) {
                            queue.addFirst(list.get(i));    //Add remaining back to be processed in original order
                        }
                        waitForRetry();
                        break;
                    } else {
                        successCount++;
                        failureCount = 0;
                        lastDelayMs = retryDelayMS;
                    }
                }
            }

        }
    }

    private void waitForRetry() {

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
            throw new RuntimeException(e);  //Should never get an exception from simple Map<String,String>
        }

        log.info("Attempting to post data: {}", str);

        DataOutputStream dos = new DataOutputStream(connection.getOutputStream());
        dos.writeBytes(str);
        dos.flush();
        dos.close();

        int responseCode = connection.getResponseCode();
        System.out.println("\nSending 'POST' request to URL : " + url);
//        System.out.println("Post parameters : " + urlParameters);
        System.out.println("Response Code : " + responseCode);

        BufferedReader in = new BufferedReader(
                new InputStreamReader(connection.getInputStream()));
        String inputLine;
        StringBuilder response = new StringBuilder();

        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();

        //print result
        System.out.println(response.toString());

        return responseCode == 200;
    }
}
