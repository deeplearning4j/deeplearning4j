package org.deeplearning4j.ui.storage;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.agrona.DirectBuffer;
import org.agrona.MutableDirectBuffer;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.api.StatsReport;
import org.deeplearning4j.ui.stats.impl.SbeStatsInitializationReport;
import org.deeplearning4j.ui.stats.impl.SbeStatsReport;
import org.deeplearning4j.ui.stats.storage.StatsStorageListener;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
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
        StatsStorage ss = new MapDBStatsStorage.Builder()
                .file(f)
                .build();

        CountingListener l = new CountingListener();
        ss.registerStatsStorageListener(l);
        assertEquals(1, ss.getListeners().size());

        assertEquals(0, ss.listSessionIDs().size());
        assertNull(ss.getLatestUpdate("sessionID","typeID","workerID"));
        assertEquals(0, ss.listSessionIDs().size());


        byte[] b0 = randomBytes(123);


        ss.putStaticInfo(getInitReport(0));
        assertEquals(1, l.countNewSession);
        assertEquals(1, l.countNewWorkerId);
        assertEquals(1, l.countStaticInfo);
        assertEquals(0, l.countUpdate);

        assertEquals(Collections.singletonList("sid0"), ss.listSessionIDs());
        assertTrue(ss.sessionExists("sid0"));
        assertFalse(ss.sessionExists("sid1"));
        Persistable expected = getInitReport(0);
        Persistable p = ss.getStaticInfo("sid0","tid0","wid0");
        assertEquals(expected, p);
        assertNull(ss.getLatestUpdate("sid0","tid0","wid0"));
        assertEquals(0, ss.getAllUpdatesAfter("sid0","tid0","wid0",0).size());



        ss.putUpdate(getReport(0,0,0,12345));
        assertEquals(1, ss.getNumUpdateRecordsFor("sid0"));
        List<Persistable> list = ss.getLatestUpdateAllWorkers("sid0","tid0");
        assertEquals(1, list.size());
        assertEquals(getReport(0,0,0,12345), ss.getUpdate("sid0","tid0","wid0",12345));
        assertEquals(1, l.countNewSession);
        assertEquals(1, l.countNewWorkerId);
        assertEquals(1, l.countStaticInfo);
        assertEquals(1, l.countUpdate);

        ss.putUpdate(getReport(0,0,0,12346));
        assertEquals(1, ss.getLatestUpdateAllWorkers("sid0","tid0").size());
        assertEquals(getReport(0,0,0,12346), ss.getLatestUpdate("sid0","tid0","wid0"));
        assertEquals(getReport(0,0,0,12346), ss.getUpdate("sid0","tid0","wid0",12346));

        ss.putUpdate(getReport(0,0,1,12345));
        assertEquals(2, ss.getLatestUpdateAllWorkers("sid0","tid0").size());
        assertEquals(getReport(0,0,1,12345), ss.getLatestUpdate("sid0","tid0","wid1"));
        assertEquals(getReport(0,0,1,12345), ss.getUpdate("sid0","tid0","wid1",12345));

        assertEquals(1, l.countNewSession);
        assertEquals(2, l.countNewWorkerId);
        assertEquals(1, l.countStaticInfo);
        assertEquals(3, l.countUpdate);



        //Close and re-open
        ss.close();
        assertTrue(ss.isClosed());

        ss = new MapDBStatsStorage.Builder()
                .file(f)
                .build();

        assertEquals(getReport(0,0,0,12345), ss.getUpdate("sid0","tid0","wid0",12345));
        assertEquals(getReport(0,0,0,12346), ss.getLatestUpdate("sid0","tid0","wid0"));
        assertEquals(getReport(0,0,0,12346), ss.getUpdate("sid0","tid0","wid0",12346));
        assertEquals(getReport(0,0,1,12345), ss.getLatestUpdate("sid0","tid0","wid1"));
        assertEquals(getReport(0,0,1,12345), ss.getUpdate("sid0","tid0","wid1",12345));
        assertEquals(2, ss.getLatestUpdateAllWorkers("sid0","tid0").size());
    }

    private static byte[] randomBytes(int length){
        Random r = new Random(12345);
        byte[] bytes = new byte[length];
        r.nextBytes(bytes);
        return bytes;
    }

    private static StatsInitializationReport getInitReport(int idNumber){
        StatsInitializationReport rep = new SbeStatsInitializationReport();
        rep.reportModelInfo("classname","jsonconfig",new String[]{"p0","p1"},1,10);
        rep.reportIDs("sid"+idNumber,"tid"+idNumber,"wid"+idNumber,12345);
        rep.reportHardwareInfo(0,2,1000,2000,new long[]{3000,4000},new String[]{"dev0","dev1"},"hardwareuid");
        rep.reportSoftwareInfo("arch","osName","jvmName","jvmVersion","1.8","backend","dtype","hostname","jvmuid");
        return rep;
    }

    private static StatsReport getReport(int sid, int tid, int wid, long time){
        StatsReport rep = new SbeStatsReport(new String[]{"p0","p1"});
        rep.reportIDs("sid"+sid,"tid"+tid,"wid"+wid,time);
        rep.reportScore(100.0);
        rep.reportPerformance(1000,1001,1002,1003.0,1004.0);
        return rep;
    }

    @NoArgsConstructor @Data
    private static class CountingListener implements StatsStorageListener {

        private int countNewSession;
        private int countNewTypeID;
        private int countNewWorkerId;
        private int countStaticInfo;
        private int countUpdate;
        private int countMetaData;

        @Override
        public void notifyNewSession(String sessionID) {
            countNewSession++;
        }

        @Override
        public void notifyNewTypeID(String sessionID, String typeID) {
            countNewTypeID++;
        }

        @Override
        public void notifyNewWorkerID(String sessionID, String workerID) {
            countNewWorkerId++;
        }

        @Override
        public void notifyStaticInfo(String sessionID, String typeID, String workerID) {
            countStaticInfo++;
        }

        @Override
        public void notifyStatusUpdate(String sessionID, String typeID, String workerID, long timestamp) {
            countUpdate++;
        }

        @Override
        public void notifyStorageMetaData(String sessionID, String typeID) {
            countMetaData++;
        }
    }


}
