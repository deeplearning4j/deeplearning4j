package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.IrisDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertTrue;

@Slf4j
public class SameDiffTrainingTest {

    @Test
    public void irisTrainingSanityCheck() {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(iter);
        iter.setPreProcessor(std);

        for (String u : new String[]{"sgd", "adam", "nesterov", "adamax", "amsgrad"}) {
            Nd4j.getRandom().setSeed(12345);
            log.info("Starting: " + u);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.placeHolder("input", -1, 4);
            SDVariable label = sd.placeHolder("label", -1, 3);

            SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), DataType.FLOAT, 4, 10);
            SDVariable b0 = sd.zero("b0", 1, 10);

            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), DataType.FLOAT, 10, 3);
            SDVariable b1 = sd.zero("b1", 1, 3);

            SDVariable z0 = in.mmul(w0).add(b0);
            SDVariable a0 = sd.tanh(z0);
            SDVariable z1 = a0.mmul(w1).add("prediction", b1);
            SDVariable a1 = sd.softmax(z1);

            SDVariable diff = sd.f().squaredDifference(a1, label);
            SDVariable lossMse = diff.mul(diff).mean();

            IUpdater updater;
            switch (u) {
                case "sgd":
                    updater = new Sgd(3e-1);
                    break;
                case "adam":
                    updater = new Adam(1e-2);
                    break;
                case "nesterov":
                    updater = new Nesterovs(1e-1);
                    break;
                case "adamax":
                    updater = new AdaMax(1e-2);
                    break;
                case "amsgrad":
                    updater = new AMSGrad(1e-2);
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

            sd.fit(iter, 100);

            Evaluation e = new Evaluation();
            Map<String, List<IEvaluation>> evalMap = new HashMap<>();
            evalMap.put("prediction", Collections.singletonList(e));

            sd.evaluateMultiple(iter, evalMap);

            System.out.println(e.stats());

            double acc = e.accuracy();
            assertTrue(u + " - " + acc, acc >= 0.8);
        }
    }
}
