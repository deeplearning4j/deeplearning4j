/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;

import static org.junit.jupiter.api.Assertions.*;

public class SameDiffSpecifiedLossVarsTests extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpecifiedLoss1(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.var("ph", DataType.FLOAT, 3, 4);
        ph1.setArray(Nd4j.create(DataType.FLOAT, 3, 4));

        SDVariable add = ph1.add(1);

        SDVariable shape = add.shape();
        SDVariable out = add.sum("sum");

        sd.setLossVariables("sum");
        sd.createGradFunction();

        assertFalse(shape.hasGradient());
        try{ assertNull(shape.gradient()); } catch (IllegalStateException e){ assertTrue(e.getMessage().contains("only floating point variables")); }
        assertNotNull(out.gradient());
        assertNotNull(add.gradient());
        assertNotNull(ph1.gradient());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSpecifiedLoss2(Nd4jBackend backend) {
        for( int i = 0; i < 2; i++) {
            SameDiff sd = SameDiff.create();
            SDVariable ph = sd.placeHolder("ph", DataType.FLOAT, 3, 4);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 5));
            SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 5));

            SDVariable mmul = ph.mmul(w);
            SDVariable badd = mmul.add(b);

            SDVariable add = badd.add(1);

            SDVariable shape = add.shape();
            SDVariable unused1 = ph.mul(2);
            SDVariable unused2 = ph.sub(4);
            SDVariable unused3 = unused1.div(unused2);
            SDVariable loss1 = add.std("l1", true);
            SDVariable loss2 = mmul.mean("l2");

//            System.out.println(sd.summary());
            sd.summary();

            if(i == 0){
                sd.setLossVariables("l1", "l2");
                sd.createGradFunction();
            } else {
                TrainingConfig tc = TrainingConfig.builder()
                        .updater(new Adam(0.01))
                        .minimize("l1","l2")
                        .dataSetFeatureMapping("ph")
                        .markLabelsUnused()
                        .build();
                sd.setTrainingConfig(tc);
                DataSet ds = new DataSet(Nd4j.create(3,4), null);
                sd.fit(ds);
                sd.fit(ds);
            }

            for(String s : new String[]{"w", "b", badd.name(), add.name(), "l1", "l2"}){
                SDVariable gradVar = sd.getVariable(s).gradient();
                assertNotNull(gradVar,s);
            }
            //Unused:
            assertFalse(shape.hasGradient());
            try{ assertNull(shape.gradient()); } catch (IllegalStateException e){ assertTrue(e.getMessage().contains("only floating point variables")); }
            for(String s : new String[]{unused1.name(), unused2.name(), unused3.name()}){
                assertNull(sd.getVariable(s).gradient());
            }
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingDifferentLosses(Nd4jBackend backend) {
        //Net with 2 losses: train on the first one, then change losses
        //Also check that if modifying via add/setLossVariables the training config changes

        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.placeHolder("ph1", DataType.FLOAT, 3, 4);
        SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 4, 5));
        SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 5));

        SDVariable mmul1 = ph1.mmul(w1);
        SDVariable badd1 = mmul1.add(b1);


        SDVariable ph2 = sd.placeHolder("ph2", DataType.FLOAT, 3, 2);
        SDVariable w2 = sd.var("w2", Nd4j.rand(DataType.FLOAT, 2, 6));
        SDVariable b2 = sd.var("b2", Nd4j.rand(DataType.FLOAT, 6));

        SDVariable mmul2 = ph2.mmul(w2);
        SDVariable badd2 = mmul2.add(b2);

        SDVariable loss1 = badd1.std("loss1",true);
        SDVariable loss2 = badd2.std("loss2", true);


        //First: create grad function for optimizing loss 1 only
        sd.setLossVariables("loss1");
        sd.createGradFunction();
        for(SDVariable v : new SDVariable[]{ph1, w1, b1, mmul1, badd1, loss1}){
            assertNotNull(v.gradient(),v.name());
        }
        for(SDVariable v : new SDVariable[]{ph2, w2, b2, mmul2, badd2, loss2}){
            assertNull(v.gradient(),v.name());
        }

        //Now, set to other loss function
        sd.setLossVariables("loss2");
        sd.createGradFunction();
        for(SDVariable v : new SDVariable[]{ph1, w1, b1, mmul1, badd1, loss1}){
            assertNull(v.gradient(),v.name());
        }
        for(SDVariable v : new SDVariable[]{ph2, w2, b2, mmul2, badd2, loss2}){
            assertNotNull(v.gradient(),v.name());
        }

        //Train the first side of the graph. The other side should remain unmodified!
        sd.setLossVariables("loss1");
        INDArray w1Before = w1.getArr().dup();
        INDArray b1Before = b1.getArr().dup();
        INDArray w2Before = w2.getArr().dup();
        INDArray b2Before = b2.getArr().dup();


        TrainingConfig tc = TrainingConfig.builder()
                .updater(new Adam(1e-2))
                .dataSetFeatureMapping("ph1","ph2")
                .markLabelsUnused()
                .build();
        sd.setTrainingConfig(tc);

        MultiDataSet mds = new MultiDataSet(new INDArray[]{Nd4j.rand(DataType.FLOAT, 3,4), Nd4j.rand(DataType.FLOAT, 3,2)}, new INDArray[0]);

        sd.fit(new SingletonMultiDataSetIterator(mds), 3);
        assertNotEquals(w1Before, w1.getArr());
        assertNotEquals(b1Before, b1.getArr());
        assertEquals(w2Before, w2.getArr());
        assertEquals(b2Before, b2.getArr());

        //Train second side of graph; first side should be unmodified
        sd.setLossVariables("loss2");
        w1Before = w1.getArr().dup();
        b1Before = b1.getArr().dup();
        w2Before = w2.getArr().dup();
        b2Before = b2.getArr().dup();

        sd.fit(new SingletonMultiDataSetIterator(mds), 3);
        assertEquals(w1Before, w1.getArr());
        assertEquals(b1Before, b1.getArr());
        assertNotEquals(w2Before, w2.getArr());
        assertNotEquals(b2Before, b2.getArr());

    }
}
