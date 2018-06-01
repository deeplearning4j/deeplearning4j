package org.nd4j.linalg.api.ops.impl.scatter;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by farizrahman4u on 3/23/18.
 */

public class ScatterDiv extends DynamicCustomOp {

    public ScatterDiv(SameDiff sameDiff, SDVariable ref, SDVariable indices, SDVariable updates) {
        super(null, sameDiff, new SDVariable[]{ref, indices, updates}, false);
    }

    public ScatterDiv() {}

    @Override
    public String opName() {
        return "scatter_div";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ScatterDiv";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradOut){
        //3 args: ref, indices, updates
        //For non-modified indices, input gradient (referenc) is same as output gradient
        //For modified indices, dL/dref = dL/dOut * dOut/dRef = dL/dOut * d(ref / update)/dRef = dL/dOut / update
        //And for updates, dL/du = dL/dOut * dOut/du = dL/dOut * d(ref / update)/du = dL/dOut * ref / u^2

        SDVariable ref = arg(0);
        SDVariable indices = arg(1);
        SDVariable updates = arg(2);

        List<SDVariable> ret = new ArrayList<>(3);
        SDVariable gradRef = f().scatterDiv(gradOut.get(0), indices, updates);
        ret.add(gradRef);            //Reference array
        ret.add(f().zerosLike(arg(1)));  //Indices

        SDVariable gatherOutGrad = f().gather(gradOut.get(0), indices, 0);       //Updates
        SDVariable gatherRef = f().gather(ref, indices, 0);
        SDVariable updateGrad = gatherOutGrad.mul(gatherRef).div(f().square(updates)).neg();
        ret.add(updateGrad);

        return ret;
    }

}
