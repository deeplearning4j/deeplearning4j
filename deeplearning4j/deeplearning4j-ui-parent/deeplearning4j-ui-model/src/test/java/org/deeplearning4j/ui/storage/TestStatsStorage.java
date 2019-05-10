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

package org.deeplearning4j.ui.storage;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.api.StatsReport;
import org.deeplearning4j.ui.stats.impl.SbeStatsInitializationReport;
import org.deeplearning4j.ui.stats.impl.SbeStatsReport;
import org.deeplearning4j.ui.stats.impl.java.JavaStatsInitializationReport;
import org.deeplearning4j.ui.stats.impl.java.JavaStatsReport;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by Alex on 03/10/2016.
 */
public class TestStatsStorage {

    @Rule
    public final TemporaryFolder testDir = new TemporaryFolder();


    @Test
    public void testStatsStorage() throws IOException {

        for (boolean useJ7Storage : new boolean[] {false, true}) {
            for (int i = 0; i < 3; i++) {

                StatsStorage ss;
                switch (i) {
                    case 0:
                        File f = createTempFile("TestMapDbStatsStore", ".db");
                        f.delete(); //Don't want file to exist...
                        ss = new MapDBStatsStorage.Builder().file(f).build();
                        break;
                    case 1:
                        File f2 = createTempFile("TestJ7FileStatsStore", ".db");
                        f2.delete(); //Don't want file to exist...
                        ss = new J7FileStatsStorage(f2);
                        break;
                    case 2:
                        ss = new InMemoryStatsStorage();
                        break;
                    default:
                        throw new RuntimeException();
                }


                CountingListener l = new CountingListener();
                ss.registerStatsStorageListener(l);
                assertEquals(1, ss.getListeners().size());

                assertEquals(0, ss.listSessionIDs().size());
                assertNull(ss.getLatestUpdate("sessionID", "typeID", "workerID"));
                assertEquals(0, ss.listSessionIDs().size());


                ss.putStaticInfo(getInitReport(0, 0, 0, useJ7Storage));
                assertEquals(1, l.countNewSession);
                assertEquals(1, l.countNewWorkerId);
                assertEquals(1, l.countStaticInfo);
                assertEquals(0, l.countUpdate);

                assertEquals(Collections.singletonList("sid0"), ss.listSessionIDs());
                assertTrue(ss.sessionExists("sid0"));
                assertFalse(ss.sessionExists("sid1"));
                Persistable expected = getInitReport(0, 0, 0, useJ7Storage);
                Persistable p = ss.getStaticInfo("sid0", "tid0", "wid0");
                assertEquals(expected, p);
                List<Persistable> allStatic = ss.getAllStaticInfos("sid0", "tid0");
                assertEquals(Collections.singletonList(expected), allStatic);
                assertNull(ss.getLatestUpdate("sid0", "tid0", "wid0"));
                assertEquals(Collections.singletonList("tid0"), ss.listTypeIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSessionAndType("sid0", "tid0"));
                assertEquals(0, ss.getAllUpdatesAfter("sid0", "tid0", "wid0", 0).size());
                assertEquals(0, ss.getNumUpdateRecordsFor("sid0"));
                assertEquals(0, ss.getNumUpdateRecordsFor("sid0", "tid0", "wid0"));


                //Put first update
                ss.putUpdate(getReport(0, 0, 0, 12345, useJ7Storage));
                assertEquals(1, ss.getNumUpdateRecordsFor("sid0"));
                assertEquals(getReport(0, 0, 0, 12345, useJ7Storage), ss.getLatestUpdate("sid0", "tid0", "wid0"));
                assertEquals(Collections.singletonList("tid0"), ss.listTypeIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSessionAndType("sid0", "tid0"));
                assertEquals(Collections.singletonList(getReport(0, 0, 0, 12345, useJ7Storage)),
                                ss.getAllUpdatesAfter("sid0", "tid0", "wid0", 0));
                assertEquals(1, ss.getNumUpdateRecordsFor("sid0"));
                assertEquals(1, ss.getNumUpdateRecordsFor("sid0", "tid0", "wid0"));

                List<Persistable> list = ss.getLatestUpdateAllWorkers("sid0", "tid0");
                assertEquals(1, list.size());
                assertEquals(getReport(0, 0, 0, 12345, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid0", 12345));
                assertEquals(1, l.countNewSession);
                assertEquals(1, l.countNewWorkerId);
                assertEquals(1, l.countStaticInfo);
                assertEquals(1, l.countUpdate);

                //Put second update
                ss.putUpdate(getReport(0, 0, 0, 12346, useJ7Storage));
                assertEquals(1, ss.getLatestUpdateAllWorkers("sid0", "tid0").size());
                assertEquals(Collections.singletonList("tid0"), ss.listTypeIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSessionAndType("sid0", "tid0"));
                assertEquals(getReport(0, 0, 0, 12346, useJ7Storage), ss.getLatestUpdate("sid0", "tid0", "wid0"));
                assertEquals(getReport(0, 0, 0, 12346, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid0", 12346));

                ss.putUpdate(getReport(0, 0, 1, 12345, useJ7Storage));
                assertEquals(2, ss.getLatestUpdateAllWorkers("sid0", "tid0").size());
                assertEquals(getReport(0, 0, 1, 12345, useJ7Storage), ss.getLatestUpdate("sid0", "tid0", "wid1"));
                assertEquals(getReport(0, 0, 1, 12345, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid1", 12345));

                assertEquals(1, l.countNewSession);
                assertEquals(2, l.countNewWorkerId);
                assertEquals(1, l.countStaticInfo);
                assertEquals(3, l.countUpdate);


                //Put static info and update with different session, type and worker IDs
                ss.putStaticInfo(getInitReport(100, 200, 300, useJ7Storage));
                assertEquals(2, ss.getLatestUpdateAllWorkers("sid0", "tid0").size());

                ss.putUpdate(getReport(100, 200, 300, 12346, useJ7Storage));
                assertEquals(Collections.singletonList(getReport(100, 200, 300, 12346, useJ7Storage)),
                                ss.getLatestUpdateAllWorkers("sid100", "tid200"));
                assertEquals(Collections.singletonList("tid200"), ss.listTypeIDsForSession("sid100"));
                List<String> temp = ss.listWorkerIDsForSession("sid100");
                System.out.println("temp: " + temp);
                assertEquals(Collections.singletonList("wid300"), ss.listWorkerIDsForSession("sid100"));
                assertEquals(Collections.singletonList("wid300"),
                                ss.listWorkerIDsForSessionAndType("sid100", "tid200"));
                assertEquals(getReport(100, 200, 300, 12346, useJ7Storage),
                                ss.getLatestUpdate("sid100", "tid200", "wid300"));
                assertEquals(getReport(100, 200, 300, 12346, useJ7Storage),
                                ss.getUpdate("sid100", "tid200", "wid300", 12346));

                assertEquals(2, l.countNewSession);
                assertEquals(3, l.countNewWorkerId);
                assertEquals(2, l.countStaticInfo);
                assertEquals(4, l.countUpdate);




                //Test get updates times:
                long[] expTSWid0 = new long[]{12345, 12346};
                long[] actTSWid0 = ss.getAllUpdateTimes("sid0", "tid0", "wid0");
                assertArrayEquals(expTSWid0, actTSWid0);

                long[] expTSWid1 = new long[]{12345};
                long[] actTSWid1 = ss.getAllUpdateTimes("sid0", "tid0", "wid1");
                assertArrayEquals(expTSWid1, actTSWid1);



                ss.putUpdate(getReport(100, 200, 300, 12347, useJ7Storage));
                ss.putUpdate(getReport(100, 200, 300, 12348, useJ7Storage));
                ss.putUpdate(getReport(100, 200, 300, 12349, useJ7Storage));

                long[] expTSWid300 = new long[]{12346, 12347, 12348, 12349};
                long[] actTSWid300 = ss.getAllUpdateTimes("sid100", "tid200", "wid300");
                assertArrayEquals(expTSWid300, actTSWid300);

                //Test subset query:
                List<Persistable> subset = ss.getUpdates("sid100", "tid200", "wid300", new long[]{12346, 12349});
                assertEquals(2, subset.size());
                assertEquals(Arrays.asList(getReport(100, 200, 300, 12346, useJ7Storage),
                        getReport(100, 200, 300, 12349, useJ7Storage)),
                        subset);
            }
        }
    }


    @Test
    public void testFileStatsStore() throws IOException {

        for (boolean useJ7Storage : new boolean[] {false, true}) {
            for (int i = 0; i < 2; i++) {
                File f;
                if (i == 0) {
                    f = createTempFile("TestMapDbStatsStore", ".db");
                } else {
                    f = createTempFile("TestSqliteStatsStore", ".db");
                }

                f.delete(); //Don't want file to exist...
                StatsStorage ss;
                if (i == 0) {
                    ss = new MapDBStatsStorage.Builder().file(f).build();
                } else {
                    ss = new J7FileStatsStorage(f);
                }


                CountingListener l = new CountingListener();
                ss.registerStatsStorageListener(l);
                assertEquals(1, ss.getListeners().size());

                assertEquals(0, ss.listSessionIDs().size());
                assertNull(ss.getLatestUpdate("sessionID", "typeID", "workerID"));
                assertEquals(0, ss.listSessionIDs().size());


                ss.putStaticInfo(getInitReport(0, 0, 0, useJ7Storage));
                assertEquals(1, l.countNewSession);
                assertEquals(1, l.countNewWorkerId);
                assertEquals(1, l.countStaticInfo);
                assertEquals(0, l.countUpdate);

                assertEquals(Collections.singletonList("sid0"), ss.listSessionIDs());
                assertTrue(ss.sessionExists("sid0"));
                assertFalse(ss.sessionExists("sid1"));
                Persistable expected = getInitReport(0, 0, 0, useJ7Storage);
                Persistable p = ss.getStaticInfo("sid0", "tid0", "wid0");
                assertEquals(expected, p);
                List<Persistable> allStatic = ss.getAllStaticInfos("sid0", "tid0");
                assertEquals(Collections.singletonList(expected), allStatic);
                assertNull(ss.getLatestUpdate("sid0", "tid0", "wid0"));
                assertEquals(Collections.singletonList("tid0"), ss.listTypeIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSessionAndType("sid0", "tid0"));
                assertEquals(0, ss.getAllUpdatesAfter("sid0", "tid0", "wid0", 0).size());
                assertEquals(0, ss.getNumUpdateRecordsFor("sid0"));
                assertEquals(0, ss.getNumUpdateRecordsFor("sid0", "tid0", "wid0"));


                //Put first update
                ss.putUpdate(getReport(0, 0, 0, 12345, useJ7Storage));
                assertEquals(1, ss.getNumUpdateRecordsFor("sid0"));
                assertEquals(getReport(0, 0, 0, 12345, useJ7Storage), ss.getLatestUpdate("sid0", "tid0", "wid0"));
                assertEquals(Collections.singletonList("tid0"), ss.listTypeIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSessionAndType("sid0", "tid0"));
                assertEquals(Collections.singletonList(getReport(0, 0, 0, 12345, useJ7Storage)),
                                ss.getAllUpdatesAfter("sid0", "tid0", "wid0", 0));
                assertEquals(1, ss.getNumUpdateRecordsFor("sid0"));
                assertEquals(1, ss.getNumUpdateRecordsFor("sid0", "tid0", "wid0"));

                List<Persistable> list = ss.getLatestUpdateAllWorkers("sid0", "tid0");
                assertEquals(1, list.size());
                assertEquals(getReport(0, 0, 0, 12345, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid0", 12345));
                assertEquals(1, l.countNewSession);
                assertEquals(1, l.countNewWorkerId);
                assertEquals(1, l.countStaticInfo);
                assertEquals(1, l.countUpdate);

                //Put second update
                ss.putUpdate(getReport(0, 0, 0, 12346, useJ7Storage));
                assertEquals(1, ss.getLatestUpdateAllWorkers("sid0", "tid0").size());
                assertEquals(Collections.singletonList("tid0"), ss.listTypeIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSession("sid0"));
                assertEquals(Collections.singletonList("wid0"), ss.listWorkerIDsForSessionAndType("sid0", "tid0"));
                assertEquals(getReport(0, 0, 0, 12346, useJ7Storage), ss.getLatestUpdate("sid0", "tid0", "wid0"));
                assertEquals(getReport(0, 0, 0, 12346, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid0", 12346));

                ss.putUpdate(getReport(0, 0, 1, 12345, useJ7Storage));
                assertEquals(2, ss.getLatestUpdateAllWorkers("sid0", "tid0").size());
                assertEquals(getReport(0, 0, 1, 12345, useJ7Storage), ss.getLatestUpdate("sid0", "tid0", "wid1"));
                assertEquals(getReport(0, 0, 1, 12345, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid1", 12345));

                assertEquals(1, l.countNewSession);
                assertEquals(2, l.countNewWorkerId);
                assertEquals(1, l.countStaticInfo);
                assertEquals(3, l.countUpdate);


                //Put static info and update with different session, type and worker IDs
                ss.putStaticInfo(getInitReport(100, 200, 300, useJ7Storage));
                assertEquals(2, ss.getLatestUpdateAllWorkers("sid0", "tid0").size());

                ss.putUpdate(getReport(100, 200, 300, 12346, useJ7Storage));
                assertEquals(Collections.singletonList(getReport(100, 200, 300, 12346, useJ7Storage)),
                                ss.getLatestUpdateAllWorkers("sid100", "tid200"));
                assertEquals(Collections.singletonList("tid200"), ss.listTypeIDsForSession("sid100"));
                List<String> temp = ss.listWorkerIDsForSession("sid100");
                System.out.println("temp: " + temp);
                assertEquals(Collections.singletonList("wid300"), ss.listWorkerIDsForSession("sid100"));
                assertEquals(Collections.singletonList("wid300"),
                                ss.listWorkerIDsForSessionAndType("sid100", "tid200"));
                assertEquals(getReport(100, 200, 300, 12346, useJ7Storage),
                                ss.getLatestUpdate("sid100", "tid200", "wid300"));
                assertEquals(getReport(100, 200, 300, 12346, useJ7Storage),
                                ss.getUpdate("sid100", "tid200", "wid300", 12346));

                assertEquals(2, l.countNewSession);
                assertEquals(3, l.countNewWorkerId);
                assertEquals(2, l.countStaticInfo);
                assertEquals(4, l.countUpdate);


                //Close and re-open
                ss.close();
                assertTrue(ss.isClosed());

                if (i == 0) {
                    ss = new MapDBStatsStorage.Builder().file(f).build();
                } else {
                    ss = new J7FileStatsStorage(f);
                }


                assertEquals(getReport(0, 0, 0, 12345, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid0", 12345));
                assertEquals(getReport(0, 0, 0, 12346, useJ7Storage), ss.getLatestUpdate("sid0", "tid0", "wid0"));
                assertEquals(getReport(0, 0, 0, 12346, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid0", 12346));
                assertEquals(getReport(0, 0, 1, 12345, useJ7Storage), ss.getLatestUpdate("sid0", "tid0", "wid1"));
                assertEquals(getReport(0, 0, 1, 12345, useJ7Storage), ss.getUpdate("sid0", "tid0", "wid1", 12345));
                assertEquals(2, ss.getLatestUpdateAllWorkers("sid0", "tid0").size());


                assertEquals(1, ss.getLatestUpdateAllWorkers("sid100", "tid200").size());
                assertEquals(Collections.singletonList("tid200"), ss.listTypeIDsForSession("sid100"));
                assertEquals(Collections.singletonList("wid300"), ss.listWorkerIDsForSession("sid100"));
                assertEquals(Collections.singletonList("wid300"),
                                ss.listWorkerIDsForSessionAndType("sid100", "tid200"));
                assertEquals(getReport(100, 200, 300, 12346, useJ7Storage),
                                ss.getLatestUpdate("sid100", "tid200", "wid300"));
                assertEquals(getReport(100, 200, 300, 12346, useJ7Storage),
                                ss.getUpdate("sid100", "tid200", "wid300", 12346));
            }
        }
    }

    private static StatsInitializationReport getInitReport(int idNumber, int tid, int wid, boolean useJ7Storage) {
        StatsInitializationReport rep;
        if (useJ7Storage) {
            rep = new JavaStatsInitializationReport();
        } else {
            rep = new SbeStatsInitializationReport();
        }

        rep.reportModelInfo("classname", "jsonconfig", new String[] {"p0", "p1"}, 1, 10);
        rep.reportIDs("sid" + idNumber, "tid" + tid, "wid" + wid, 12345);
        rep.reportHardwareInfo(0, 2, 1000, 2000, new long[] {3000, 4000}, new String[] {"dev0", "dev1"}, "hardwareuid");
        Map<String, String> envInfo = new HashMap<>();
        envInfo.put("envInfo0", "value0");
        envInfo.put("envInfo1", "value1");
        rep.reportSoftwareInfo("arch", "osName", "jvmName", "jvmVersion", "1.8", "backend", "dtype", "hostname",
                        "jvmuid", envInfo);
        return rep;
    }

    private static StatsReport getReport(int sid, int tid, int wid, long time, boolean useJ7Storage) {
        StatsReport rep;
        if (useJ7Storage) {
            rep = new JavaStatsReport();
        } else {
            rep = new SbeStatsReport();
        }

        rep.reportIDs("sid" + sid, "tid" + tid, "wid" + wid, time);
        rep.reportScore(100.0);
        rep.reportPerformance(1000, 1001, 1002, 1003.0, 1004.0);
        return rep;
    }

    @NoArgsConstructor
    @Data
    private static class CountingListener implements StatsStorageListener {

        private int countNewSession;
        private int countNewTypeID;
        private int countNewWorkerId;
        private int countStaticInfo;
        private int countUpdate;
        private int countMetaData;

        @Override
        public void notify(StatsStorageEvent event) {
            System.out.println("Event: " + event);
            switch (event.getEventType()) {
                case NewSessionID:
                    countNewSession++;
                    break;
                case NewTypeID:
                    countNewTypeID++;
                    break;
                case NewWorkerID:
                    countNewWorkerId++;
                    break;
                case PostMetaData:
                    countMetaData++;
                    break;
                case PostStaticInfo:
                    countStaticInfo++;
                    break;
                case PostUpdate:
                    countUpdate++;
                    break;
            }
        }
    }

    private File createTempFile(String prefix, String suffix) throws IOException {
        return testDir.newFile(prefix + "-" + System.nanoTime() + suffix);
    }

}
