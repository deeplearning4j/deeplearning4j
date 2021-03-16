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

package org.nd4j.autodiff.ui;

import com.google.flatbuffers.Table;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.listeners.impl.UIListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.graph.UIEvent;
import org.nd4j.graph.UIGraphStructure;
import org.nd4j.graph.UIStaticInfoRecord;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.IrisDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class UIListenerTest extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }



    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUIListenerBasic(@TempDir Path testDir,Nd4jBackend backend) throws Exception {
        Nd4j.getRandom().setSeed(12345);

        IrisDataSetIterator iter = new IrisDataSetIterator(150, 150);

        SameDiff sd = getSimpleNet();

        File dir = testDir.toFile();
        File f = new File(dir, "logFile.bin");
        UIListener l = UIListener.builder(f)
                .plotLosses(1)
                .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();

        sd.setListeners(l);

        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")
                .dataSetLabelMapping("label")
                .updater(new Adam(1e-1))
                .weightDecay(1e-3, true)
                .build());

        sd.fit(iter, 20);

        //Test inference after training with UI Listener still around
        Map<String, INDArray> m = new HashMap<>();
        iter.reset();
        m.put("in", iter.next().getFeatures());
        INDArray out = sd.outputSingle(m, "softmax");
        assertNotNull(out);
        assertArrayEquals(new long[]{150, 3}, out.shape());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUIListenerContinue(@TempDir Path testDir,Nd4jBackend backend) throws Exception {
        IrisDataSetIterator iter = new IrisDataSetIterator(150, 150);

        SameDiff sd1 = getSimpleNet();
        SameDiff sd2 = getSimpleNet();

        File dir = testDir.toFile();
        File f = new File(dir, "logFileNoContinue.bin");
        f.delete();
        UIListener l1 = UIListener.builder(f)
                .plotLosses(1)
                .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();

        sd1.setListeners(l1);

        sd1.fit(iter, 2);


        //Do some thing with 2nd net, in 2 sets
        File f2 = new File(dir, "logFileContinue.bin");
        UIListener l2 = UIListener.builder(f2)
                .plotLosses(1)
                .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();

        sd2.setListeners(l2);
        sd2.fit(iter, 1);

        l2 = UIListener.builder(f2)
                .plotLosses(1)
                .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();
        sd2.setListeners(l2);
        sd2.setListeners(l2);
        sd2.fit(iter, 1);

        assertEquals(f.length(), f2.length());

        LogFileWriter lfw1 = new LogFileWriter(f);
        LogFileWriter lfw2 = new LogFileWriter(f2);


        //Check static info are equal:
        LogFileWriter.StaticInfo si1 = lfw1.readStatic();
        LogFileWriter.StaticInfo si2 = lfw2.readStatic();

        List<Pair<UIStaticInfoRecord, Table>> ls1 = si1.getData();
        List<Pair<UIStaticInfoRecord, Table>> ls2 = si2.getData();

        assertEquals(ls1.size(), ls2.size());
        for( int i=0; i<ls1.size(); i++ ){
            Pair<UIStaticInfoRecord, Table> p1 = ls1.get(i);
            Pair<UIStaticInfoRecord, Table> p2 = ls2.get(i);
            assertEquals(p1.getFirst().infoType(), p2.getFirst().infoType());
            if(p1.getSecond() == null){
                assertNull(p2.getSecond());
            } else {
                assertEquals(p1.getSecond().getClass(), p2.getSecond().getClass());
                if(p1.getSecond() instanceof UIGraphStructure){
                    UIGraphStructure g1 = (UIGraphStructure) p1.getSecond();
                    UIGraphStructure g2 = (UIGraphStructure) p2.getSecond();

                    assertEquals(g1.inputsLength(), g2.inputsLength());
                    assertEquals(g1.outputsLength(), g2.outputsLength());
                    assertEquals(g1.opsLength(), g2.opsLength());
                }
            }
        }

        //Check events:
        List<Pair<UIEvent, Table>> e1 = lfw1.readEvents();
        List<Pair<UIEvent, Table>> e2 = lfw2.readEvents();
        assertEquals(e1.size(), e2.size());

        for( int i=0; i<e1.size(); i++ ){
            Pair<UIEvent, Table> p1 = e1.get(i);
            Pair<UIEvent, Table> p2 = e2.get(i);
            UIEvent ev1 = p1.getFirst();
            UIEvent ev2 = p2.getFirst();

            assertEquals(ev1.eventType(), ev2.eventType());
            assertEquals(ev1.epoch(), ev2.epoch());
            assertEquals(ev1.iteration(), ev2.iteration());
        }
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testUIListenerBadContinue(@TempDir Path testDir,Nd4jBackend backend) throws Exception {
        IrisDataSetIterator iter = new IrisDataSetIterator(150, 150);
        SameDiff sd1 = getSimpleNet();

        File dir = testDir.toFile();
        File f = new File(dir, "logFile.bin");
        f.delete();
        UIListener l1 = UIListener.builder(f)
                .plotLosses(1)
                .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();

        sd1.setListeners(l1);

        sd1.fit(iter, 2);

        //Now, fit with different net - more placeholders
        SameDiff sd2 = SameDiff.create();
        SDVariable in1 = sd2.placeHolder("in1", DataType.FLOAT, -1, 4);
        SDVariable in2 = sd2.placeHolder("in2", DataType.FLOAT, -1, 4);
        SDVariable w = sd2.var("w", DataType.FLOAT, 1, 4);
        SDVariable mul = in1.mul(in2).mul(w);
        SDVariable loss = mul.std(true);
        sd2.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")
                .dataSetLabelMapping("label")
                .updater(new Adam(1e-1))
                .build());

        UIListener l2 = UIListener.builder(f)
                .plotLosses(1)
                .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();

        sd2.setListeners(l2);
        try {
            sd2.fit(iter, 2);
            fail("Expected exception");
        } catch (Throwable t){
            String m = t.getMessage();
            assertTrue(m.contains("placeholder"),m);
            assertTrue(m.contains("FileMode.CREATE_APPEND_NOCHECK"),m);
        }


        //fit with different net - more variables
        SameDiff sd3 = getSimpleNet();
        sd3.var("SomeNewVar", DataType.FLOAT, 3,4);
        UIListener l3 = UIListener.builder(f)
                .plotLosses(1)
                .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();

        sd3.setListeners(l3);

        try {
            sd3.fit(iter, 2);
            fail("Expected exception");
        } catch (Throwable t){
            String m = t.getMessage();
            assertTrue(m.contains("variable"),m);
            assertTrue(m.contains("FileMode.CREATE_APPEND_NOCHECK"),m);
        }


        //Fit with proper net:
        SameDiff sd4 = getSimpleNet();
        UIListener l4 = UIListener.builder(f)
                .plotLosses(1)
                .trainEvaluationMetrics("softmax", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();

        sd4.setListeners(l4);
        sd4.fit(iter, 2);
    }


    private static SameDiff getSimpleNet(){
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);
        SDVariable w = sd.var("W", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", DataType.FLOAT, 1, 3);
        SDVariable mmul = in.mmul(w).add(b);
        SDVariable softmax = sd.nn.softmax("softmax", mmul);
        SDVariable loss = sd.loss().logLoss("loss", label, softmax);

        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")
                .dataSetLabelMapping("label")
                .updater(new Adam(1e-1))
                .weightDecay(1e-3, true)
                .build());
        return sd;
    }

}
