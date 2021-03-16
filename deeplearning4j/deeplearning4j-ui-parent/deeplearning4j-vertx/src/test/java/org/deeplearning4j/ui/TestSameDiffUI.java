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

package org.deeplearning4j.ui;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.ui.api.UIServer;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;

@Disabled
public class TestSameDiffUI extends BaseDL4JTest {
    private static Logger log = LoggerFactory.getLogger(TestSameDiffUI.class.getName());

    @Disabled
    @Test
    public void testSameDiff(@TempDir Path testDir) throws Exception {
        File dir = testDir.toFile();
        File f = new File(dir, "ui_data.bin");
        log.info("File path: {}", f.getAbsolutePath());

        f.getParentFile().mkdirs();
        f.delete();

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 3);
        SDVariable w = sd.var("w", DataType.FLOAT, 3,4);
        SDVariable b = sd.var("b", DataType.FLOAT, 1, 4);

        SDVariable z = in.mmul(w).add(b);
        SDVariable a = sd.math().tanh(z);

        LogFileWriter lfw = new LogFileWriter(f);
        lfw.writeGraphStructure(sd);
        lfw.writeFinishStaticMarker();

        //Append a number of events
        lfw.registerEventName("accuracy");
        lfw.registerEventName("precision");
        long t = System.currentTimeMillis();
        for( int iter = 0; iter < 50; iter++) {
            double d = Math.cos(0.1 * iter);
            d *= d;
            lfw.writeScalarEvent("accuracy", LogFileWriter.EventSubtype.EVALUATION, t + iter, iter, 0, d);

            double prec = Math.min(0.05 * iter, 1.0);
            lfw.writeScalarEvent("precision", LogFileWriter.EventSubtype.EVALUATION, t+iter, iter, 0, prec);
        }

        //Add some histograms:
        lfw.registerEventName("histogramDiscrete");
        lfw.registerEventName("histogramEqualSpacing");
        lfw.registerEventName("histogramCustomBins");
        for(int i = 0; i < 3; i++) {
            INDArray discreteY = Nd4j.createFromArray(0, 1, 2);
            lfw.writeHistogramEventDiscrete("histogramDiscrete", LogFileWriter.EventSubtype.TUNING_METRIC,  t+i, i, 0, Arrays.asList("zero", "one", "two"), discreteY);

            INDArray eqSpacingY = Nd4j.createFromArray(-0.5 + 0.5 * i, 0.75 * i + i, 1.0 * i + 1.0);
            lfw.writeHistogramEventEqualSpacing("histogramEqualSpacing", LogFileWriter.EventSubtype.TUNING_METRIC, t+i, i, 0, 0.0, 1.0, eqSpacingY);

            INDArray customBins = Nd4j.createFromArray(new double[][]{
                    {0.0, 0.5, 0.9},
                    {0.2, 0.55, 1.0}
            });
            System.out.println(Arrays.toString(customBins.data().asFloat()));
            System.out.println(customBins.shapeInfoToString());
            lfw.writeHistogramEventCustomBins("histogramCustomBins", LogFileWriter.EventSubtype.TUNING_METRIC, t+i, i, 0, customBins, eqSpacingY);
        }


        UIServer uiServer = UIServer.getInstance();


        Thread.sleep(1_000_000_000);
    }

}
