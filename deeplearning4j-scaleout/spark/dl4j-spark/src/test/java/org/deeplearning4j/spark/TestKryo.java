package org.deeplearning4j.spark;

import org.apache.spark.serializer.SerializerInstance;
import org.deeplearning4j.eval.*;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 04/07/2017.
 */
public class TestKryo extends BaseSparkKryoTest {

    @Test
    public void testSerializationConfigurations(){

        //Check configurations



        fail();
    }


    @Test
    public void testSerializationEvaluation(){

        Evaluation e = new Evaluation();
        e.eval(Nd4j.create(new double[]{1,0,0}), Nd4j.create(new double[]{0.2,0.5,0.3}));

        EvaluationBinary eb = new EvaluationBinary();
        eb.eval(Nd4j.create(new double[]{1,0,0}), Nd4j.create(new double[]{0.2,0.6,0.3}));

        ROC roc = new ROC(30);
        roc.eval(Nd4j.create(new double[]{1}), Nd4j.create(new double[]{0.2}));
        ROC roc2 = new ROC();
        roc2.eval(Nd4j.create(new double[]{1}), Nd4j.create(new double[]{0.2}));

        ROCMultiClass rocM = new ROCMultiClass(30);
        rocM.eval(Nd4j.create(new double[]{1,0,0}), Nd4j.create(new double[]{0.2,0.5,0.3}));
        ROCMultiClass rocM2 = new ROCMultiClass();
        rocM2.eval(Nd4j.create(new double[]{1,0,0}), Nd4j.create(new double[]{0.2,0.5,0.3}));

        ROCBinary rocB = new ROCBinary(30);
        rocB.eval(Nd4j.create(new double[]{1,0,0}), Nd4j.create(new double[]{0.2,0.6,0.3}));

        ROCBinary rocB2 = new ROCBinary();
        rocB2.eval(Nd4j.create(new double[]{1,0,0}), Nd4j.create(new double[]{0.2,0.6,0.3}));

        RegressionEvaluation re = new RegressionEvaluation();
        re.eval(Nd4j.rand(1,5), Nd4j.rand(1,5));

        IEvaluation[] evaluations = new IEvaluation[]{
                new Evaluation(), e,
                new EvaluationBinary(), eb,
                new ROC(), roc, roc2,
                new ROCMultiClass(), rocM, rocM2,
                new ROCBinary(), rocB, rocB2,
                new RegressionEvaluation(), re
        };

        SerializerInstance si = sc.env().serializer().newInstance();

        for(IEvaluation ie : evaluations ){
            //System.out.println(ie.getClass());
            ByteBuffer bb = si.serialize(ie, null);
            IEvaluation ie2 = si.deserialize(bb, null);

            assertEquals(ie, ie2);
        }





    }

}
