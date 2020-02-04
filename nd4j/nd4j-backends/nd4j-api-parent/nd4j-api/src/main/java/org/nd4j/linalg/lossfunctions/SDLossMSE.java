package org.nd4j.linalg.lossfunctions;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SDLossMSE extends SameDiffLoss {
    public SDLossMSE(){
        super();
    }

    @Override
    public SDVariable defineLoss(SameDiff sameDiff, SDVariable layerInput, SDVariable labels) {
        return null;
    }
}
