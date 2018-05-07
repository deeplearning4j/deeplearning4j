package org.deeplearning4j.util;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

@Slf4j
public class CuDNNValidationUtil {

    public static final double MAX_REL_ERROR = 1e-5;
    public static final double MIN_ABS_ERROR = 1e-6;

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
        @Builder.Default private boolean trainFirst = false;
        INDArray features;
        INDArray labels;
        private DataSetIterator data;
    }

    public static void validateMLN(MultiLayerNetwork net, TestCase t){
        assertNotNull(t.getAllowCudnnHelpersForClasses());
        assertFalse(t.getAllowCudnnHelpersForClasses().isEmpty());


        MultiLayerNetwork net2 = new MultiLayerNetwork(net.getLayerWiseConfigurations().clone());
        net2.init();
        net2.params().assign(net.params());
        removeHelpers(net.getLayers(), null);



        if(t.isTrainFirst()){
            log.info("Validation - training first...");
            log.info("*** NOT YET IMPLEMENTED***");

            //TODO
        }

        if(t.isTestForward()){
            Preconditions.checkNotNull(t.getFeatures(), "Features are not set (null)");
            log.info("Validation - checking forward pass");

            for(boolean train : new boolean[]{false, true}) {
                String tr = train ? "Train: " : "Test: ";
                List<INDArray> ff1 = net.feedForward(t.getFeatures(), train);
                List<INDArray> ff2 = net2.feedForward(t.getFeatures(), train);
                for( int i=0; i<ff1.size(); i++ ){
                    int layerIdx = i-1; //FF includes input
                    String layerName = "layer_" + layerIdx + " - " +
                            (i == 0 ? "input" : net.getLayer(layerIdx).getClass().getSimpleName());
                    INDArray arr1 = ff1.get(i);
                    INDArray arr2 = ff2.get(i);

                    INDArray relError = relError(arr1, arr2, MIN_ABS_ERROR);
                    double maxRE = relError.maxNumber().doubleValue();
                    assertTrue(tr + layerName + " - max RE: " + maxRE, maxRE < MAX_REL_ERROR);
                    log.info("Forward pass, max relative error: " + layerName + " - " + maxRE);
                }

                INDArray out1 = net.output(t.getFeatures(), train);
                INDArray out2 = net2.output(t.getFeatures(), train);
                INDArray relError = relError(out1, out2, MIN_ABS_ERROR);
                double maxRE = relError.maxNumber().doubleValue();
                log.info("Output, max relative error: " + maxRE);
                assertTrue(tr + "Max RE: " + maxRE, maxRE < MAX_REL_ERROR);
            }
        }


        if(t.isTestScore()) {
            Preconditions.checkNotNull(t.getFeatures(), "Features are not set (null)");
            Preconditions.checkNotNull(t.getLabels(), "Labels are not set (null)");

            log.info("Validation - checking scores");
            double s1 = net.score(new DataSet(t.getFeatures(), t.getLabels()));
            double s2 = net2.score(new DataSet(t.getFeatures(), t.getLabels()));

            double re = relError(s1, s2);
            String s = "Relative error: " + re;
            assertTrue(s, re < MAX_REL_ERROR);
        }

        if(t.isTestBackward()) {
            log.info("Validation - checking backward pass");


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
            if(keepHelpersFor != null) {
                boolean keepAndAssertPresent = false;
                for (Class<?> c : keepHelpersFor) {
                    if(c.isAssignableFrom(l.getClass())){
                        keepAndAssertPresent = true;
                        break;
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

    private static INDArray relError(INDArray a1, INDArray a2, double minAbsError){
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
        INDArray lessThanMinAbs = Transforms.abs(a1.sub(a2), false);
        BooleanIndexing.replaceWhere(lessThanMinAbs, 0.0, Conditions.lessThan(minAbsError));
        lessThanMinAbs = Transforms.not(lessThanMinAbs);    //Sets 0 to 1, and anything other than 0 to 0

        INDArray result = absDiff.divi(abs1.addi(abs2));
        //Only way to have NaNs given there weren't any in original : both 0s
        BooleanIndexing.replaceWhere(result, 0.0, Conditions.isNan());
        //Finally, set to 0 if less than min abs error:
        result.muli(lessThanMinAbs);

        double maxRE = result.maxNumber().doubleValue();
        if(maxRE > MAX_REL_ERROR){
            System.out.println();
        }
        return result;
    }

}
