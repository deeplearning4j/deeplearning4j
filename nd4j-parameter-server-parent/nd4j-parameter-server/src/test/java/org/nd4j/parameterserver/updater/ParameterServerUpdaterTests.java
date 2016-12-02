package org.nd4j.parameterserver.updater;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
        ParameterServerUpdater updater = new SynchronousParameterUpdater(cores);
        INDArray arr = Nd4j.zeros(2,2);
        for(int i = 0; i < cores; i++) {
            updater.update(Nd4j.ones(2,2),arr);
        }

        assertTrue(updater.shouldReplicate());
        updater.reset();
        assertFalse(updater.shouldReplicate());
        assumeNotNull(updater.toJson());


    }

}
