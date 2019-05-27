/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.play;

import org.deeplearning4j.ui.api.UIServer;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;

@Ignore
public class TestSameDiffUI {

//    @Ignore
    @Test
    public void testSameDiff() throws Exception {

//
//        File f = new File("C:/Temp/SameDiffUI/ui_data.bin");
//        f.getParentFile().mkdirs();
//        f.delete();
//
//        SameDiff sd = SameDiff.create();
//        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 3);
//        SDVariable w = sd.var("w", DataType.FLOAT, 3,4);
//        SDVariable b = sd.var("b", DataType.FLOAT, 1, 4);
//
//        SDVariable z = in.mmul(w).add(b);
//        SDVariable a = sd.nn().tanh(z);
//
//        LogFileWriter lfw = new LogFileWriter(f);
//        lfw.writeGraphStructure(sd);
//        lfw.writeFinishStaticMarker();
//
//        //Append a number of events
//        lfw.registerEventName("accuracy");
//        lfw.registerEventName("precision");
//        long t = System.currentTimeMillis();
//        for( int iter=0; iter<50; iter++) {
//            double d = Math.cos(0.1*iter);
//            d *= d;
//            lfw.writeScalarEvent("accuracy", t + iter, iter, 0, d);
//
//            double prec = Math.min(0.05 * iter, 1.0);
//            lfw.writeScalarEvent("precision", t+iter, iter, 0, prec);
//        }
//
//        //Add some histograms:
//        lfw.registerEventName("histogramDiscrete");
//        lfw.registerEventName("histogramEqualSpacing");
//        lfw.registerEventName("histogramCustomBins");
//        for( int i=0; i<3; i++ ){
//            INDArray discreteY = Nd4j.createFromArray(0, 1, 2);
//            lfw.writeHistogramEventDiscrete("histogramDiscrete", t+i, i, 0, Arrays.asList("zero", "one", "two"), discreteY);
//
//            INDArray eqSpacingY = Nd4j.createFromArray(-0.5 + 0.5 * i, 0.75 * i + i, 1.0 * i + 1.0);
//            lfw.writeHistogramEventEqualSpacing("histogramEqualSpacing", t+i, i, 0, 0.0, 1.0, eqSpacingY);
//
//            INDArray customBins = Nd4j.createFromArray(new double[][]{
//                    {0.0, 0.5, 0.9},
//                    {0.2, 0.55, 1.0}
//            });
//            System.out.println(Arrays.toString(customBins.data().asFloat()));
//            System.out.println(customBins.shapeInfoToString());
//            lfw.writeHistogramEventCustomBins("histogramCustomBins", t+i, i, 0, customBins, eqSpacingY);
//        }


        UIServer uiServer = UIServer.getInstance();


        Thread.sleep(1_000_000_000);
    }

}
