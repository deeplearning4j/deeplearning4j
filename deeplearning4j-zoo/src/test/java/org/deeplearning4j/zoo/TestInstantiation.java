package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.CifarLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Tests workflow for zoo model instantiation.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TestInstantiation {

    @Test
    public void testMultipleCnnTraining() {
        Map<ZooType, ZooModel> models = ModelSelector.select(ZooType.CNN, CifarLoader.NUM_LABELS);

        for(Map.Entry<ZooType, ZooModel> entry : models.entrySet()) {
            log.info("Testing training on zoo model "+entry.getKey());
            ZooModel model = entry.getValue();

            // set up data iterator
            int[] inputShape = model.metaData().getInputShape()[0];
            CifarDataSetIterator iter = new CifarDataSetIterator(16, CifarLoader.NUM_LABELS*16, new int[]{inputShape[1], inputShape[2], inputShape[0]});

            Model initializedModel = model.init();
            try {
                while (iter.hasNext()) {
                    DataSet ds = iter.next();
                    if (initializedModel instanceof ComputationGraph)
                        ((ComputationGraph) initializedModel).fit(ds);
                    else if (initializedModel instanceof MultiLayerNetwork)
                        ((MultiLayerNetwork) initializedModel).fit(ds);
                    else
                        throw new IllegalStateException("Zoo models are only MultiLayerNetwork or ComputationGraph.");
                }
            } catch(Exception e) {
                log.error("Test failed on model type: "+entry.getKey());
                throw new RuntimeException(e);
            }

            // clean up for current model
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            System.gc();
        }
    }

}
