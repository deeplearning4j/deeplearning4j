package org.nd4j.linalg.lossfunctions;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.IActivation;

public class SDLossMSE extends SameDiffLoss {
    public SDLossMSE(){
        super();
    }

    IActivation activationFn;

    @Override
    public SDVariable defineLoss(SameDiff sd, SDVariable layerInput, SDVariable labels) {

        SDVariable out = sd.nn().softplus("out", sd.getVariable("layerInput"));
        return  labels.squaredDifference(out).div(labels.getArr().size(1));


    }
}
