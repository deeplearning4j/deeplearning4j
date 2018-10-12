package org.nd4j;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;

public class DebugTraining {

    @Test
    public void test() throws Exception {

        SameDiff sd = SameDiff.create();

        //TODO placeholders
        SDVariable in = sd.var("input", -1, 784);
        SDVariable label = sd.var("label", -1, 10);

        sd.addAsPlaceHolder("input");
        sd.addAsPlaceHolder("label");

        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28*28, 128), 28*28, 128);
        SDVariable b0 = sd.zero("b0", 1, 128);

        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 128, 10), 128, 10);
        SDVariable b1 = sd.zero("b1", 1, 10);

        SDVariable z0 = in.mmul(w0).add(b0);
        SDVariable a0 = sd.tanh(z0);
        SDVariable z1 = a0.mmul(w1).add(b1);
        SDVariable a1 = sd.softmax(z1);

        SDVariable diff = sd.f().squaredDifference(a1, label);
        SDVariable lossMse = diff.mul(diff).mean();

        DataSetIterator iter = new MnistDataSetIterator(32, true, 12345);

        TrainingConfig conf = new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(new Sgd(0.01))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .build();

        sd.setTrainingConfig(conf);

        sd.fit(iter, 1);

        System.out.println("DONE");
    }

}
