package org.nd4j.parameterserver.updater.storage;

import org.junit.Test;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

/**
 * Created by agibsonccc on 12/2/16.
 */
public class UpdaterStorageTests {


    @Test(expected = UnsupportedOperationException.class)
    public void testNone() {
        UpdateStorage updateStorage = new NoUpdateStorage();
        NDArrayMessage message = NDArrayMessage.wholeArrayUpdate(Nd4j.scalar(1.0));
        updateStorage.addUpdate(message);
        assertEquals(1, updateStorage.numUpdates());
        assertEquals(message, updateStorage.getUpdate(0));
        updateStorage.close();
    }

    @Test
    public void testInMemory() {
        UpdateStorage updateStorage = new InMemoryUpdateStorage();
        NDArrayMessage message = NDArrayMessage.wholeArrayUpdate(Nd4j.scalar(1.0));
        updateStorage.addUpdate(message);
        assertEquals(1, updateStorage.numUpdates());
        assertEquals(message, updateStorage.getUpdate(0));
        updateStorage.clear();
        assertEquals(0, updateStorage.numUpdates());
        updateStorage.close();
    }

}
