package org.nd4j.parameterserver.updater;

import org.junit.Test;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.aeron.ndarrayholder.InMemoryNDArrayHolder;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.updater.storage.NoUpdateStorage;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 12/2/16.
 */
public class ParameterServerUpdaterTests {

    @Test
    public void synchronousTest() {
        int cores = Runtime.getRuntime().availableProcessors();
        ParameterServerUpdater updater = new SynchronousParameterUpdater(new NoUpdateStorage(),
                        new InMemoryNDArrayHolder(Nd4j.zeros(2, 2)), cores);
        for (int i = 0; i < cores; i++) {
            updater.update(NDArrayMessage.wholeArrayUpdate(Nd4j.ones(2, 2)));
        }

        assertTrue(updater.shouldReplicate());
        updater.reset();
        assertFalse(updater.shouldReplicate());
        assumeNotNull(updater.toJson());


    }

}
