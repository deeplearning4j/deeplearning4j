package org.nd4j.linalg.api.ops.impl.transforms.clip;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

public class ClipByNormBp extends DynamicCustomOp {

    private double clipValue;

    public ClipByNormBp() {
        //
    }

    public ClipByNormBp(SameDiff sameDiff, SDVariable x, SDVariable eps, double clipValue, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{x, eps});
        this.clipValue = clipValue;
        this.dimensions = dimensions;
        addIArgument(dimensions);
        addTArgument(clipValue);
    }

    @Override
    public String opName() {
        return "clipbynorm_bp";
    }

}
