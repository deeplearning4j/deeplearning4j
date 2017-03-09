package org.deeplearning4j.spark.time;

import org.apache.commons.net.ntp.NTPUDPClient;
import org.apache.commons.net.ntp.TimeInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetAddress;
import java.util.Timer;
import java.util.TimerTask;

/**
 * A {@link TimeSource} that utilize Network Time Protocol to determine the system clock offset<br>
 * Instances should be obtained via {@link #getInstance()} or {@link TimeSourceProvider}; one instance may be
 * used per machine<br>
 *
 * Specifically, the implementation uses Apache Commons Net (already a dependency in Spark) to query a NTP server.
 * This querying is done periodically (default: once upon initialization and then every 30 minutes thereafter).<br>
 *
 * The following configuration options can be set via system properties:<br>
 * To set the time update frequency (for querying the NTP server, in <b>milliseconds</b>): org.deeplearning4j.spark.time.NTPTimeSource.frequencyms<br>
 * To set the NTP server address: org.deeplearning4j.spark.time.NTPTimeSource.server<br>
 * Default NTP server: {@link #DEFAULT_NTP_SERVER}
 *
 *
 * @author Alex Black
 */
public class NTPTimeSource implements TimeSource {

    public static final String NTP_SOURCE_UPDATE_FREQUENCY_MS_PROPERTY =
                    "org.deeplearning4j.spark.time.NTPTimeSource.frequencyms";
    public static final String NTP_SOURCE_SERVER_PROPERTY = "org.deeplearning4j.spark.time.NTPTimeSource.server";
    public static final int MAX_QUERY_RETRIES = 10;
    public static final int DEFAULT_NTP_TIMEOUT_MS = 10000;
    public static final long DEFAULT_UPDATE_FREQUENCY = 30 * 60 * 1000L; //30 Minutes
    public static final long MIN_UPDATE_FREQUENCY = 30000L; //30 sec

    public static final String DEFAULT_NTP_SERVER = "0.pool.ntp.org";

    private static Logger log = LoggerFactory.getLogger(NTPTimeSource.class);
    private static NTPTimeSource instance;

    public static synchronized TimeSource getInstance() {
        if (instance == null)
            instance = new NTPTimeSource();
        return instance;
    }

    private volatile long lastOffsetGetTimeSystemMS = -1;
    private volatile long lastOffsetMilliseconds;

    private final long synchronizationFreqMS;
    private final String ntpServer;

    private NTPTimeSource() {
        this(getUpdateFrequencyConfiguration(), getServerConfiguration());
    }

    private NTPTimeSource(long synchronizationFreqMS, String ntpServer) {
        this.synchronizationFreqMS = synchronizationFreqMS;
        this.ntpServer = ntpServer;

        log.debug("Initializing NTPTimeSource with query frequency {} ms using server {}", synchronizationFreqMS,
                        ntpServer);

        queryServerNow();

        //Start a Timer to periodically query the server
        Timer timer = new Timer(true);
        timer.scheduleAtFixedRate(new QueryServerTask(), synchronizationFreqMS, synchronizationFreqMS);

        log.debug("Initialized NTPTimeSource with query frequency {} ms using server {}", synchronizationFreqMS,
                        ntpServer);
    }

    //Query and parse the system property
    private static long getUpdateFrequencyConfiguration() {
        String property = System.getProperty(NTP_SOURCE_UPDATE_FREQUENCY_MS_PROPERTY);
        Long parseAttempt = null;
        long updateFreq;
        if (property != null) {
            try {
                parseAttempt = Long.parseLong(property);
            } catch (Exception e) {
                log.info("Error parsing system property \"{}\" with value \"{}\"",
                                NTP_SOURCE_UPDATE_FREQUENCY_MS_PROPERTY, property);
            }
            if (parseAttempt != null) {
                if (parseAttempt < MIN_UPDATE_FREQUENCY) {
                    log.info("Invalid update frequency (milliseconds): {} is less than minimum {}. Using default update frequency: {} ms",
                                    parseAttempt, MIN_UPDATE_FREQUENCY, DEFAULT_UPDATE_FREQUENCY);
                    updateFreq = DEFAULT_UPDATE_FREQUENCY;
                } else {
                    updateFreq = parseAttempt;
                }
            } else {
                updateFreq = DEFAULT_UPDATE_FREQUENCY;
            }
        } else {
            updateFreq = DEFAULT_UPDATE_FREQUENCY;
        }
        return updateFreq;
    }

    private static String getServerConfiguration() {
        return System.getProperty(NTP_SOURCE_SERVER_PROPERTY, DEFAULT_NTP_SERVER);
    }


    private void queryServerNow() {
        Long offsetResult = null;
        for (int i = 0; i < MAX_QUERY_RETRIES; i++) {
            try {
                NTPUDPClient client = new NTPUDPClient();
                client.setDefaultTimeout(DEFAULT_NTP_TIMEOUT_MS);// Timeout if a response takes longer than 10 seconds

                client.open();
                InetAddress address = InetAddress.getByName(ntpServer);
                TimeInfo info = client.getTime(address);
                info.computeDetails();
                Long offset = info.getOffset();
                if (offset == null) {
                    throw new Exception("Could not calculate time offset (offset is null)");
                } else {
                    offsetResult = offset;
                    break;
                }
            } catch (Exception e) {
                log.error("Error querying NTP server, attempt {} of {}", (i + 1), MAX_QUERY_RETRIES, e);
            }
        }

        if (offsetResult == null) {
            log.error("Could not successfully query NTP server after " + MAX_QUERY_RETRIES + " tries");
            throw new RuntimeException("Could not successfully query NTP server after " + MAX_QUERY_RETRIES + " tries");
        }

        lastOffsetGetTimeSystemMS = System.currentTimeMillis();
        lastOffsetMilliseconds = offsetResult;
        log.debug("Updated local time offset based on NTP server result. Offset = {}", lastOffsetMilliseconds);
    }

    //Timer task to be run periodically
    private class QueryServerTask extends TimerTask {
        public void run() {
            queryServerNow();
        }
    }



    //Get system offset. Note: positive offset means system clock is behind time server; negative offset means system
    // clock is ahead of time server
    private synchronized long getSystemOffset() {
        return lastOffsetMilliseconds;
    }

    public long currentTimeMillis() {
        long offset = getSystemOffset();
        long systemTime = System.currentTimeMillis();
        return systemTime + offset;
    }
}
