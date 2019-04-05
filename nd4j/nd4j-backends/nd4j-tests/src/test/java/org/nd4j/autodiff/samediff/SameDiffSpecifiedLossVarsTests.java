package org.nd4j.autodiff.samediff;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static junit.framework.TestCase.assertNotNull;
import static junit.framework.TestCase.assertNull;

/**
 * Created by Alex on 04/04/2019.
 */
public class SameDiffSpecifiedLossVarsTests {

    @Test
    public void testSpecifiedLoss1(){
        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.placeHolder("ph", DataType.FLOAT, 3, 4);
        ph1.setArray(Nd4j.create(DataType.FLOAT, 3, 4));

        SDVariable add = ph1.add(1);

        SDVariable shape = add.shape();
        SDVariable out = add.sum("sum");

        sd.setLossVariables("sum");
        sd.createGradFunction();

        assertNull(shape.gradient());
        assertNotNull(out.gradient());
        assertNotNull(add.gradient());
        assertNotNull(ph1.gradient());
    }

    @Test
    public void testSpecifiedLoss2(){
        for( int i=0; i<3; i++ ) {
            SameDiff sd = SameDiff.create();
            SDVariable ph1 = sd.placeHolder("ph", DataType.FLOAT, 3, 4);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 5));
            SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 5));

            SDVariable mmul = ph1.mmul(w);
            SDVariable badd = mmul.add(b);

            SDVariable add = badd.add(1);

            SDVariable shape = add.shape();
            SDVariable loss1 = add.std("l1", true);
            SDVariable loss2 = mmul.mean("l2");

            System.out.println(sd.summary());

            if(i == 0){
                sd.setLossVariables("l1", "l2");
                sd.createGradFunction();
                for(String s : new String[]{"w", "b", badd.getVarName(), add.getVarName(), /*"l1", "l2"*/}){
                    SDVariable gradVar = sd.getVariable(s).gradient();
                    assertNotNull(s, gradVar);
                }
                assertNull(shape.gradient());
            } else if(i == 1){
                TrainingConfig tc = TrainingConfig.builder()
                        .updater(new Adam(0.01))
                        .minimize("l1","l2")
                        .dataSetFeatureMapping("ph1")
                        .build();
                sd.setTrainingConfig(tc);
                DataSet ds = new DataSet(Nd4j.create(3,4), null);
                sd.fit(ds);
            } else {

            }


            Map<String, INDArray> m = new HashMap<>();
            m.put("ph", Nd4j.create(DataType.FLOAT, 3, 4));
        }

    }

}
