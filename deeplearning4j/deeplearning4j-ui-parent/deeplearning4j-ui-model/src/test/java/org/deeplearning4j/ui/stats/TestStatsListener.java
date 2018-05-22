package org.deeplearning4j.ui.stats;

import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Created by Alex on 07/10/2016.
 */
public class TestStatsListener {

    @Test
    public void testListenerBasic() {

        for (boolean useJ7 : new boolean[] {false, true}) {

            DataSet ds = new IrisDataSetIterator(150, 150).next();

            MultiLayerConfiguration conf =
                            new NeuralNetConfiguration.Builder()
                                            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                            .list().layer(0,
                                                            new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                                            .nIn(4).nOut(3).build())
                                            .pretrain(false).backprop(true).build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            StatsStorage ss = new MapDBStatsStorage(); //in-memory

            if (useJ7) {
                net.setListeners(new J7StatsListener(ss, 1));
            } else {
                net.setListeners(new StatsListener(ss, 1));
            }


            for (int i = 0; i < 3; i++) {
                net.fit(ds);
            }

            List<String> sids = ss.listSessionIDs();
            assertEquals(1, sids.size());
            String sessionID = ss.listSessionIDs().get(0);
            assertEquals(1, ss.listTypeIDsForSession(sessionID).size());
            String typeID = ss.listTypeIDsForSession(sessionID).get(0);
            assertEquals(1, ss.listWorkerIDsForSession(sessionID).size());
            String workerID = ss.listWorkerIDsForSession(sessionID).get(0);

            Persistable staticInfo = ss.getStaticInfo(sessionID, typeID, workerID);
            assertNotNull(staticInfo);
            System.out.println(staticInfo);

            List<Persistable> updates = ss.getAllUpdatesAfter(sessionID, typeID, workerID, 0);
            assertEquals(3, updates.size());
            for (Persistable p : updates) {
                System.out.println(p);
            }

        }

    }

}
