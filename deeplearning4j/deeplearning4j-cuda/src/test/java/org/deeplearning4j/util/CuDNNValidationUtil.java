/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.util;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.CollectScoresListener;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.lang.reflect.Field;
import java.util.*;

import static org.junit.Assert.*;

@Slf4j
public class CuDNNValidationUtil {

    public static final double MAX_REL_ERROR = 1e-4;
//    public static final double MAX_REL_ERROR = 1e-3;
//    public static final double MIN_ABS_ERROR = 1e-3;
    public static final double MIN_ABS_ERROR = 1e-5;

    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    @Builder
    public static class TestCase {
        private String testName;
        private List<Class<?>> allowCudnnHelpersForClasses;
        @Builder.Default private boolean testForward = true;
        @Builder.Default private boolean testScore = true;
        @Builder.Default private boolean testBackward = true;
        @Builder.Default private boolean testTraining = true;
        @Builder.Default private boolean trainFirst = false;
        INDArray features;
        INDArray labels;
        private DataSetIterator data;
    }

    public static void validateMLN(MultiLayerNetwork netOrig, TestCase t){
        assertNotNull(t.getAllowCudnnHelpersForClasses());
        assertFalse(t.getAllowCudnnHelpersForClasses().isEmpty());

        //Don't allow fallback:
        for(Layer l : netOrig.getLayers()){
            org.deeplearning4j.nn.conf.layers.Layer lConf = l.conf().getLayer();
            if(lConf instanceof ConvolutionLayer){
                ((ConvolutionLayer) lConf).setCudnnAllowFallback(false);
            } else if(lConf instanceof SubsamplingLayer){
                ((SubsamplingLayer) lConf).setCudnnAllowFallback(false);
            }
        }


        MultiLayerNetwork net1NoCudnn = new MultiLayerNetwork(netOrig.getLayerWiseConfigurations().clone());
        net1NoCudnn.init();
        log.info("Removing all CuDNN helpers from network copy 1");
        removeHelpers(net1NoCudnn.getLayers(), null);

        MultiLayerNetwork net2With = new MultiLayerNetwork(netOrig.getLayerWiseConfigurations().clone());
        net2With.init();
        net2With.params().assign(netOrig.params());
        log.info("Removing all except for specified CuDNN helpers from network copy 2: " + t.getAllowCudnnHelpersForClasses());
        removeHelpers(net2With.getLayers(), t.getAllowCudnnHelpersForClasses());




        if(t.isTrainFirst()){
            Preconditions.checkState(t.getData() != null, "Test data iterator is ");
            log.info("Validation - training first...");
            log.info("*** NOT YET IMPLEMENTED***");


        }

        if(t.isTestForward()){
            Preconditions.checkNotNull(t.getFeatures(), "Features are not set (null)");

            for (boolean train : new boolean[]{false, true}) {
                assertEquals(net1NoCudnn.params(), net2With.params());
                String s = "Feed forward test - " + t.getTestName() + " - " + (train ? "Train: " : "Test: ");
                List<INDArray> ff1 = net1NoCudnn.feedForward(t.getFeatures(), train);
                List<INDArray> ff2 = net2With.feedForward(t.getFeatures(), train);
                List<String> paramKeys = new ArrayList<>(net1NoCudnn.paramTable().keySet());
                Collections.sort(paramKeys);
                for (String p : paramKeys) {
                    INDArray p1 = net1NoCudnn.getParam(p);
                    INDArray p2 = net2With.getParam(p);
                    INDArray re = relError(p1, p2, MIN_ABS_ERROR);
                    double maxRE = re.maxNumber().doubleValue();
                    if (maxRE >= MAX_REL_ERROR) {
                        System.out.println("Failed param values");
                        System.out.println(p1);
                        System.out.println(p2);
                    }
                    assertTrue(s + " - param changed during forward pass: " + p, maxRE < MAX_REL_ERROR);
                }

                for( int i=0; i<ff1.size(); i++ ){
                    int layerIdx = i-1; //FF includes input
                    String layerName = "layer_" + layerIdx + " - " +
                            (i == 0 ? "input" : net1NoCudnn.getLayer(layerIdx).getClass().getSimpleName());
                    INDArray arr1 = ff1.get(i);
                    INDArray arr2 = ff2.get(i);

                    INDArray relError = relError(arr1, arr2, MIN_ABS_ERROR);
                    double maxRE = relError.maxNumber().doubleValue();
                    int idx = relError.argMax(Integer.MAX_VALUE).getInt(0);
                    if(maxRE >= MAX_REL_ERROR){
                        double d1 = arr1.dup('c').getDouble(idx);
                        double d2 = arr2.dup('c').getDouble(idx);
                        System.out.println("Different values at index " + idx + ": " + d1 + ", " + d2 + " - RE = " + maxRE);
                    }
                    assertTrue(s + layerName + " - max RE: " + maxRE, maxRE < MAX_REL_ERROR);
                    log.info("Forward pass, max relative error: " + layerName + " - " + maxRE);
                }

                INDArray out1 = net1NoCudnn.output(t.getFeatures(), train);
                INDArray out2 = net2With.output(t.getFeatures(), train);
                INDArray relError = relError(out1, out2, MIN_ABS_ERROR);
                double maxRE = relError.maxNumber().doubleValue();
                log.info(s + "Output, max relative error: " + maxRE);

                assertEquals(net1NoCudnn.params(), net2With.params());  //Check that forward pass does not modify params
                assertTrue(s + "Max RE: " + maxRE, maxRE < MAX_REL_ERROR);
            }
        }


        if(t.isTestScore()) {
            Preconditions.checkNotNull(t.getFeatures(), "Features are not set (null)");
            Preconditions.checkNotNull(t.getLabels(), "Labels are not set (null)");

            log.info("Validation - checking scores");
            double s1 = net1NoCudnn.score(new DataSet(t.getFeatures(), t.getLabels()));
            double s2 = net2With.score(new DataSet(t.getFeatures(), t.getLabels()));

            double re = relError(s1, s2);
            String s = "Relative error: " + re;
            assertTrue(s, re < MAX_REL_ERROR);
        }

        if(t.isTestBackward()) {
            Preconditions.checkNotNull(t.getFeatures(), "Features are not set (null)");
            Preconditions.checkNotNull(t.getLabels(), "Labels are not set (null)");
            log.info("Validation - checking backward pass");

            //Check gradients
            net1NoCudnn.setInput(t.getFeatures());
            net1NoCudnn.setLabels(t.getLabels());

            net2With.setInput(t.getFeatures());
            net2With.setLabels(t.getLabels());

            net1NoCudnn.computeGradientAndScore();
            net2With.computeGradientAndScore();

            List<String> paramKeys = new ArrayList<>(net1NoCudnn.paramTable().keySet());
            Collections.sort(paramKeys);
            for(String p : paramKeys){
                INDArray g1 = net1NoCudnn.gradient().gradientForVariable().get(p);
                INDArray g2 = net2With.gradient().gradientForVariable().get(p);

                if(g1 == null || g2 == null){
                    throw new RuntimeException("Null gradients");
                }

                INDArray re = relError(g1, g2, MIN_ABS_ERROR);
                double maxRE = re.maxNumber().doubleValue();
                if (maxRE >= MAX_REL_ERROR) {
                    System.out.println("Failed param values");
                    System.out.println(Arrays.toString(g1.dup().data().asFloat()));
                    System.out.println(Arrays.toString(g2.dup().data().asFloat()));
                }
                assertTrue("Gradients are not equal: " + p, maxRE < MAX_REL_ERROR);
            }
        }

        if(t.isTestTraining()){
            Preconditions.checkNotNull(t.getData(), "DataSetIterator is not set (null)");
            log.info("Testing run-to-run consistency of training with CuDNN");

            net2With = new MultiLayerNetwork(netOrig.getLayerWiseConfigurations().clone());
            net2With.init();
            net2With.params().assign(netOrig.params());
            log.info("Removing all except for specified CuDNN helpers from network copy 2: " + t.getAllowCudnnHelpersForClasses());
            removeHelpers(net2With.getLayers(), t.getAllowCudnnHelpersForClasses());

            CollectScoresListener listener = new CollectScoresListener(1);
            net2With.setListeners(listener);
            net2With.fit(t.getData());

            for( int i=0; i<2; i++ ) {

                net2With = new MultiLayerNetwork(netOrig.getLayerWiseConfigurations().clone());
                net2With.init();
                net2With.params().assign(netOrig.params());
                log.info("Removing all except for specified CuDNN helpers from network copy 2: " + t.getAllowCudnnHelpersForClasses());
                removeHelpers(net2With.getLayers(), t.getAllowCudnnHelpersForClasses());

                CollectScoresListener listener2 = new CollectScoresListener(1);
                net2With.setListeners(listener2);
                net2With.fit(t.getData());

                DoubleArrayList listOrig = listener.getListScore();
                DoubleArrayList listNew = listener2.getListScore();

                assertEquals(listOrig.size(), listNew.size());
                for (int j = 0; j < listOrig.size(); j++) {
                    double d1 = listOrig.get(j);
                    double d2 = listNew.get(j);
                    double re = relError(d1, d2);
                    String msg = "Scores at iteration " + j + " - relError = " + re + ", score1 = " + d1 + ", score2 = " + d2;
                    assertTrue(msg, re < MAX_REL_ERROR);
                    System.out.println("j=" + j + ", d1 = " + d1 + ", d2 = " + d2);
                }
            }
        }
    }

    private static void removeHelpers(Layer[] layers, List<Class<?>> keepHelpersFor){

        Map<Class<?>, Integer> map = new HashMap<>();
        for(Layer l : layers){
            Field f;
            try{
                f = l.getClass().getDeclaredField("helper");
            } catch (Exception e){
                //OK, may not be a CuDNN supported layer
                continue;
            }

            f.setAccessible(true);
            boolean keepAndAssertPresent = false;
            if(keepHelpersFor != null) {
                for (Class<?> c : keepHelpersFor) {
                    if(c.isAssignableFrom(l.getClass())){
                        keepAndAssertPresent = true;
                        break;
                    }
                }
            }
            try {
                if (keepAndAssertPresent) {
                    Object o = f.get(l);
                    assertNotNull(o);
                } else {
                    f.set(l, null);
                    Integer i = map.get(l.getClass());
                    if(i == null){
                        i = 0;
                    }
                    map.put(l.getClass(), i+1);
                }
            } catch (IllegalAccessException e){
                throw new RuntimeException(e);
            }
        }

        for(Map.Entry<Class<?>,Integer> c : map.entrySet()){
            System.out.println("Removed " + c.getValue() + " CuDNN helpers instances from layer " + c.getKey());
        }
    }

    private static double relError(double d1, double d2){
        Preconditions.checkState(!Double.isNaN(d1), "d1 is NaN");
        Preconditions.checkState(!Double.isNaN(d2), "d2 is NaN");
        if(d1 == 0.0 && d2 == 0.0){
            return 0.0;
        }

        return Math.abs(d1-d2) / (Math.abs(d1) + Math.abs(d2));
    }

    private static INDArray relError(@NonNull INDArray a1, @NonNull INDArray a2, double minAbsError){
        long numNaN1 = Nd4j.getExecutioner().exec(new MatchCondition(a1, Conditions.isNan()), Integer.MAX_VALUE).getInt(0);
        long numNaN2 = Nd4j.getExecutioner().exec(new MatchCondition(a2, Conditions.isNan()), Integer.MAX_VALUE).getInt(0);
        Preconditions.checkState(numNaN1 == 0, "Array 1 has NaNs");
        Preconditions.checkState(numNaN2 == 0, "Array 2 has NaNs");


//        INDArray isZero1 = a1.eq(0.0);
//        INDArray isZero2 = a2.eq(0.0);
//        INDArray bothZero = isZero1.muli(isZero2);

        INDArray abs1 = Transforms.abs(a1, true);
        INDArray abs2 = Transforms.abs(a2, true);
        INDArray absDiff = Transforms.abs(a1.sub(a2), false);

        //abs(a1-a2) < minAbsError ? 1 : 0
        INDArray greaterThanMinAbs = Transforms.abs(a1.sub(a2), false);
        BooleanIndexing.replaceWhere(greaterThanMinAbs, 0.0, Conditions.lessThan(minAbsError));
        BooleanIndexing.replaceWhere(greaterThanMinAbs, 1.0, Conditions.greaterThan(0.0));

        INDArray result = absDiff.divi(abs1.add(abs2));
        //Only way to have NaNs given there weren't any in original : both 0s
        BooleanIndexing.replaceWhere(result, 0.0, Conditions.isNan());
        //Finally, set to 0 if less than min abs error, or unchanged otherwise
        result.muli(greaterThanMinAbs);

//        double maxRE = result.maxNumber().doubleValue();
//        if(maxRE > MAX_REL_ERROR){
//            System.out.println();
//        }
        return result;
    }

}
