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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@Slf4j
public class CuDNNValidation {

    public static final double MAX_REL_ERROR = 1e-5;

    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    @Builder
    public static class TestCase {
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

            //TODO
        }

        if(t.isTestForward()){
            log.info("Validation - checking forward pass");

        }


        if(t.isTestScore()) {
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

        return Math.abs(d1-d2) / (Math.abs(d1) - Math.abs(d2));
    }

}
