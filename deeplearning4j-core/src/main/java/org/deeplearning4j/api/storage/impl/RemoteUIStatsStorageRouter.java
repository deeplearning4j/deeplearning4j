package org.deeplearning4j.api.storage.impl;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StorageMetaData;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Collection;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Created by Alex on 10/11/2016.
 */
@Slf4j
public class RemoteUIStatsStorageRouter implements StatsStorageRouter {

    public static void main(String[] args) throws Exception {
        new RemoteUIStatsStorageRouter(null);
    }


    private final String USER_AGENT = "Mozilla/5.0";

    private URL url;
    private int maxRetryCount;
    private long retryDelayMS;
    private double retryBackoffFactor;

    private LinkedBlockingQueue<ToPost> queue =  new LinkedBlockingQueue<>();

    private Thread postThread;

    public RemoteUIStatsStorageRouter(String url) throws Exception {
        this(url, 10, 1000, 2.0);
    }

    public RemoteUIStatsStorageRouter(String url, int maxRetryCount, long retryDelayMS, double retryBackoffFactor){
        this.maxRetryCount = maxRetryCount;
        this.retryDelayMS = retryDelayMS;
        this.retryBackoffFactor = retryBackoffFactor;

        try{
            this.url = new URL(url);
        } catch (MalformedURLException e){
            throw new RuntimeException(e);
        }

        postThread = new Thread(new PostRunnable());
        postThread.setDaemon(true);
        postThread.start();
    }

    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        queue.add(new ToPost(storageMetaData,null,null));
    }

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData) {
        for(StorageMetaData m : storageMetaData){
            putStorageMetaData(m);
        }
    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        queue.add(new ToPost(null, staticInfo, null));
    }

    @Override
    public void putStaticInfo(Collection<? extends Persistable> staticInfo) {
        for(Persistable p : staticInfo){
            putStaticInfo(p);
        }
    }

    @Override
    public void putUpdate(Persistable update) {
        queue.add(new ToPost(null, null, update));
    }

    @Override
    public void putUpdate(Collection<? extends Persistable> updates) {
        for(Persistable p : updates){
            putUpdate(p);
        }
    }

    @AllArgsConstructor
    private static class ToPost {
        private final StorageMetaData meta;
        private final Persistable staticInfo;
        private final Persistable update;
    }

    private class PostRunnable implements Runnable {

        @Override
        public void run() {
            try{
                runHelper();
            } catch (Exception e){
                e.printStackTrace();
                //TODO
            }
        }

        private void runHelper(){

            HttpURLConnection con = (HttpURLConnection) u.openConnection();

            // optional default is GET
            con.setRequestMethod("GET");

            //add request header
            con.setRequestProperty("User-Agent", USER_AGENT);

            int responseCode = con.getResponseCode();
            System.out.println("\nSending 'GET' request to URL : " + url);
            System.out.println("Response Code : " + responseCode);

            BufferedReader in = new BufferedReader(
                    new InputStreamReader(con.getInputStream()));
            String inputLine;
            StringBuffer response = new StringBuffer();

            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();

            //print result
            System.out.println(response.toString());
        }
    }
}
