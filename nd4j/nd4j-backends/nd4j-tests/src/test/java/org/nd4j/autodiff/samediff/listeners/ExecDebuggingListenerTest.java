package org.nd4j.autodiff.samediff.listeners;

import org.junit.Test;
import org.nd4j.autodiff.listeners.debugging.ExecDebuggingListener;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;

public class ExecDebuggingListenerTest extends BaseNd4jTest {

    public ExecDebuggingListenerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testExecDebugListener(){

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 3);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 1, 2);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 3, 2));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 2));
        SDVariable sm = sd.nn.softmax("softmax", in.mmul(w).add(b));
        SDVariable loss = sd.loss.logLoss("loss", label, sm);

        INDArray i = Nd4j.rand(DataType.FLOAT, 1, 3);
        INDArray l = Nd4j.rand(DataType.FLOAT, 1, 2);

        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")
                .dataSetLabelMapping("label")
                .updater(new Adam(0.001))
                .build());

        for(ExecDebuggingListener.PrintMode pm : ExecDebuggingListener.PrintMode.values()){
            sd.setListeners(new ExecDebuggingListener(pm, -1, true));
//            sd.output(m, "softmax");
            sd.fit(new DataSet(i, l));

            System.out.println("\n\n\n");
        }

    }


    @Override
    public char ordering() {
        return 'c';
    }
}
