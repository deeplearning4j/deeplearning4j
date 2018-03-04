package org.deeplearning4j.nn.conf.layers.samediff;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;

import java.util.HashMap;
import java.util.Map;

public class SameDiffLayerUtils {

    private static Map<Class<?>, Activation> activationMap;

    private SameDiffLayerUtils(){ }

    public static Activation fromIActivation(IActivation a){

        if(activationMap == null){
            Map<Class<?>,Activation> m = new HashMap<>();
            for(Activation act : Activation.values()){
                m.put(act.getActivationFunction().getClass(), act);
            }
            activationMap = m;
        }

        return activationMap.get(a.getClass());
    }

}
