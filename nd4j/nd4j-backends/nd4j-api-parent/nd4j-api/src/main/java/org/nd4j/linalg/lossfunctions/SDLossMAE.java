package org.nd4j.linalg.lossfunctions;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SDLossMAE extends SameDiffLoss {
    public SDLossMAE(){
        super();
    }

    @Override
    public SDVariable defineLoss(SameDiff sd, SDVariable layerInput, SDVariable labels) {

        return sd.math.abs(labels.sub(sd.getVariable("out"))).mean(1);

    }
}
