package org.nd4j;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertTrue;

@Slf4j
public class DebugTraining {

    @Test
    public void testBasic() throws Exception {

//        for(String u : new String[]{"sgd", "adam", "nesterov"}) {
        for(String u : new String[]{"nesterov"}) {
            log.info("Starting: " + u);
            SameDiff sd = SameDiff.create();

            //TODO placeholders
            SDVariable in = sd.var("input", -1, 784);
            SDVariable label = sd.var("label", -1, 10);

            sd.addAsPlaceHolder("input");
            sd.addAsPlaceHolder("label");

            SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28 * 28, 128), 28 * 28, 128);
            SDVariable b0 = sd.zero("b0", 1, 128);

            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 128, 10), 128, 10);
            SDVariable b1 = sd.zero("b1", 1, 10);

            SDVariable z0 = in.mmul(w0).add(b0);
            SDVariable a0 = sd.tanh(z0);
            SDVariable z1 = a0.mmul(w1).add("prediction", b1);
            SDVariable a1 = sd.softmax(z1);

            SDVariable diff = sd.f().squaredDifference(a1, label);
            SDVariable lossMse = diff.mul(diff).mean();

            DataSetIterator iter = new MnistDataSetIterator(32, true, 12345);

            IUpdater updater;
            switch (u){
                case "sgd":
                    updater = new Sgd(1e-2);
                    break;
                case "adam":
                    updater = new Adam(1e-2);
                    break;
                case "nesterov":
                    updater = new Nesterovs(1e-3);
                    break;
                default:
                    throw new RuntimeException();
            }

            TrainingConfig conf = new TrainingConfig.Builder()
                    .l2(1e-4)
                    .updater(updater)
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .build();

            sd.setTrainingConfig(conf);

//        iter = new SingletonDataSetIterator(iter.next());
            sd.fit(iter, 1);

            Evaluation e = new Evaluation();
            Map<String, List<IEvaluation>> evalMap = new HashMap<>();
            evalMap.put("prediction", Collections.singletonList(e));

            Map<String, Integer> labelMap = Collections.singletonMap("prediction", 0);

            DataSetIterator test = new MnistDataSetIterator(32, false, 12345);
            sd.evaluate(new MultiDataSetIteratorAdapter(test), evalMap, labelMap);

            System.out.println(e.stats());

            double acc = e.accuracy();
            assertTrue(String.valueOf(acc), acc > 0.8);
        }
    }

}
