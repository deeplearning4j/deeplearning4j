package org.deeplearning4j;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer;
import org.deeplearning4j.nn.layers.normalization.BatchNormalization;
import org.deeplearning4j.nn.layers.recurrent.LSTM;
import org.nd4j.base.Preconditions;

import java.lang.reflect.Field;

/**
 * Test utility methods specific to CuDNN
 *
 * @author Alex Black
 */
public class CuDNNTestUtils {

    private CuDNNTestUtils(){ }

    public static void removeHelpers(Layer[] layers) throws Exception {
        for(Layer l : layers){

            if(l instanceof ConvolutionLayer){
                Field f1 = ConvolutionLayer.class.getDeclaredField("helper");
                f1.setAccessible(true);
                f1.set(l, null);
            } else if(l instanceof SubsamplingLayer){
                Field f2 = SubsamplingLayer.class.getDeclaredField("helper");
                f2.setAccessible(true);
                f2.set(l, null);
            } else if(l instanceof BatchNormalization){
                Field f3 = BatchNormalization.class.getDeclaredField("helper");
                f3.setAccessible(true);
                f3.set(l, null);
            } else if(l instanceof LSTM){
                Field f4 = LSTM.class.getDeclaredField("helper");
                f4.setAccessible(true);
                f4.set(l, null);
            }


            if(l.getHelper() != null){
                throw new IllegalStateException("Did not remove helper");
            }
        }
    }

    public static void assertHelpersPresent(Layer[] layers) throws Exception {
        for(Layer l : layers){
            //Don't use instanceof here - there are sub conv subclasses
            if(l.getClass() == ConvolutionLayer.class || l instanceof SubsamplingLayer || l instanceof BatchNormalization || l instanceof LSTM){
                Preconditions.checkNotNull(l.getHelper(), l.conf().getLayer().getLayerName());
            }
        }
    }

    public static void assertHelpersAbsent(Layer[] layers) throws Exception {
        for(Layer l : layers){
            Preconditions.checkState(l.getHelper() == null, l.conf().getLayer().getLayerName());
        }
    }

}
