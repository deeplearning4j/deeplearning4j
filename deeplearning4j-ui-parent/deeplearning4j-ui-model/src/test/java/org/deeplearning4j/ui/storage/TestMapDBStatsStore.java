package org.deeplearning4j.ui.storage;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.stats.storage.StatsStorageListener;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStore;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Created by Alex on 03/10/2016.
 */
public class TestMapDBStatsStore {

    @Test
    public void testMapDBStatsStore() throws IOException {

        File f = Files.createTempFile("TestMapDbStatsStore",".db").toFile();
        f.delete(); //Don't want file to exist...
        StatsStorage ss = new MapDBStatsStore.Builder()
                .file(f)
                .build();

        CountingListener l = new CountingListener();
        ss.registerStatsStorageListener(l);
        assertEquals(1, ss.getListeners().size());

        assertEquals(0, ss.listSessionIDs().size());
        assertNull(ss.getLatestUpdate("sessionID","workerID"));
        assertEquals(0, ss.listSessionIDs().size());


        byte[] b0 = randomBytes(123);
        ss.putStaticInfo("sid0","wid0",b0);
        assertEquals(1, l.countNewSession);
        assertEquals(1, l.countNewWorkerId);
        assertEquals(1, l.countStaticInfo);
        assertEquals(0, l.countUpdate);

        assertEquals(Arrays.asList("sid0"), ss.listSessionIDs());
        assertTrue(ss.sessionExists("sid0"));
        assertFalse(ss.sessionExists("sid1"));
        assertArrayEquals(b0, ss.getStaticInfo("sid0","wid0"));
        assertNull(ss.getLatestUpdate("sid0","wid0"));
        assertEquals(0, ss.getAllUpdatesAfter("sid0","wid0",0).size());


        byte[] u0 = randomBytes(100);
        ss.putUpdate("sid0","wid0",100,u0);
        assertArrayEquals(u0, ss.getUpdate("sid0","wid0",100).getRecord());
        assertEquals(1, l.countNewSession);
        assertEquals(1, l.countNewWorkerId);
        assertEquals(1, l.countStaticInfo);
        assertEquals(1, l.countUpdate);

        StatsStorage.UpdateRecord r = ss.getLatestUpdate("sid0","wid0");
        assertArrayEquals(u0, r.getRecord());
        assertEquals(100,r.getTimestamp());

        byte[] u1 = randomBytes(101);
        byte[] u2 = randomBytes(102);
        ss.putUpdate("sid0","wid0",101,u1);
        ss.putUpdate("sid0","wid0",102,u2);
        StatsStorage.UpdateRecord r2 = ss.getLatestUpdate("sid0","wid0");
        assertArrayEquals(u2, r2.getRecord());
        assertEquals(102, r2.getTimestamp());

        List<StatsStorage.UpdateRecord> list = ss.getAllUpdatesAfter("sid0","wid0",100);
        assertEquals(2, list.size());
        assertArrayEquals(u1, list.get(0).getRecord());
        assertArrayEquals(u2, list.get(1).getRecord());

        assertEquals(1, l.countNewSession);
        assertEquals(1, l.countNewWorkerId);
        assertEquals(1, l.countStaticInfo);
        assertEquals(3, l.countUpdate);

        byte[] u4 = randomBytes(103);
        ss.putUpdate("sid0","wid1",100,u4);
        assertEquals(1, l.countNewSession);
        assertEquals(2, l.countNewWorkerId);
        assertEquals(1, l.countStaticInfo);
        assertEquals(4, l.countUpdate);
        assertArrayEquals(u4, ss.getUpdate("sid0","wid1",100).getRecord());


        //Close and re-open
        ss.close();
        assertTrue(ss.isClosed());

        ss = new MapDBStatsStore.Builder()
                .file(f)
                .build();

        assertArrayEquals(u0, ss.getUpdate("sid0","wid0",100).getRecord());

        r2 = ss.getLatestUpdate("sid0","wid0");
        assertArrayEquals(u2, r2.getRecord());
        assertEquals(102, r2.getTimestamp());

        list = ss.getAllUpdatesAfter("sid0","wid0",100);
        assertEquals(2, list.size());
        assertArrayEquals(u1, list.get(0).getRecord());
        assertArrayEquals(u2, list.get(1).getRecord());

        assertArrayEquals(u4, ss.getUpdate("sid0","wid1",100).getRecord());
    }

    private static byte[] randomBytes(int length){
        Random r = new Random(12345);
        byte[] bytes = new byte[length];
        r.nextBytes(bytes);
        return bytes;
    }

    @NoArgsConstructor @Data
    private static class CountingListener implements StatsStorageListener {

        private int countNewSession;
        private int countNewWorkerId;
        private int countStaticInfo;
        private int countUpdate;

        @Override
        public void notifyNewSession(String sessionID) {
            countNewSession++;
        }

        @Override
        public void notifyNewWorkerID(String sessionID, String workerID) {
            countNewWorkerId++;
        }

        @Override
        public void notifyStaticInfo(String sessionID, String workerID) {
            countStaticInfo++;
        }

        @Override
        public void notifyStatusUpdate(String sessionID, String workerID, long timestamp) {
            countUpdate++;
        }
    }
}
