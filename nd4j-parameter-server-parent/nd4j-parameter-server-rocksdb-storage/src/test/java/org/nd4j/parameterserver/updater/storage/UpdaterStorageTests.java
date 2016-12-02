package org.nd4j.parameterserver.updater.storage;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

/**
 * Created by agibsonccc on 12/2/16.
 */
public class UpdaterStorageTests {

    @Test
    public void testInMemory() {
        UpdateStorage updateStorage = new RocksDbStorage("/tmp/rocksdb");
        updateStorage.addUpdate(Nd4j.scalar(1.0));
        assertEquals(1,updateStorage.numUpdates());
        assertEquals(Nd4j.scalar(1.0),updateStorage.getUpdate(0));
        updateStorage.clear();
        assertEquals(0,updateStorage.numUpdates());
    }

}
