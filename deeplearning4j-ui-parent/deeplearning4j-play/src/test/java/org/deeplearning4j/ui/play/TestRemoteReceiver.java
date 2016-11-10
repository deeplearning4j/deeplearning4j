package org.deeplearning4j.ui.play;

import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.api.storage.impl.CollectionStatsStorageRouter;
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.impl.SbeStatsInitializationReport;
import org.deeplearning4j.ui.stats.impl.SbeStatsReport;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.storage.impl.SbeStorageMetaData;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 10/11/2016.
 */
public class TestRemoteReceiver {

    @Test
    public void testRemote() throws Exception {

        List<Persistable> updates = new ArrayList<>();
        List<Persistable> staticInfo = new ArrayList<>();
        List<StorageMetaData> metaData = new ArrayList<>();
        CollectionStatsStorageRouter collectionRouter = new CollectionStatsStorageRouter(metaData, staticInfo, updates);


        UIServer s = UIServer.getInstance();
        s.enableRemoteListener(collectionRouter,false);


        RemoteUIStatsStorageRouter remoteRouter = new RemoteUIStatsStorageRouter("http://localhost:9000/remoteReceive");

        SbeStatsReport update1 = new SbeStatsReport();
        update1.setDeviceCurrentBytes(new long[]{1,2});
        update1.reportIterationCount(10);
        update1.reportIDs("sid","tid","wid",123456);
        update1.reportPerformance(10,20,30,40,50);

        SbeStatsReport update2 = new SbeStatsReport();
        update2.setDeviceCurrentBytes(new long[]{3,4});
        update2.reportIterationCount(20);
        update2.reportIDs("sid2","tid2","wid2",123456);
        update2.reportPerformance(11,21,31,40,50);

        StorageMetaData smd1 = new SbeStorageMetaData(123,"sid","typeid","wid","initTypeClass","updaterTypeClass");
        StorageMetaData smd2 = new SbeStorageMetaData(456,"sid2","typeid2","wid2","initTypeClass2","updaterTypeClass2");

        SbeStatsInitializationReport init1 = new SbeStatsInitializationReport();
        init1.reportIDs("sid","wid","tid",3145253452L);
        init1.reportHardwareInfo(1,2,3,4,null,null,"2344253");

        remoteRouter.putUpdate(update1);
        Thread.sleep(100);
        remoteRouter.putStorageMetaData(smd1);
        Thread.sleep(100);
        remoteRouter.putStaticInfo(init1);
        Thread.sleep(100);
        remoteRouter.putUpdate(update2);
        Thread.sleep(100);
        remoteRouter.putStorageMetaData(smd2);


        Thread.sleep(2000);

        assertEquals(2, metaData.size());
        assertEquals(2, updates.size());
        assertEquals(1, staticInfo.size());

        assertEquals(Arrays.asList(update1, update2), updates);
        assertEquals(Arrays.asList(smd1, smd2), metaData);
        assertEquals(Collections.singletonList(init1), staticInfo);
    }

}
