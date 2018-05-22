package org.nd4j.linalg.api.ops.impl.transforms.comparison;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

public class ListDiff extends DynamicCustomOp {

    public ListDiff() {
        //
    }

    @Override
    public String tensorflowName() {
        return "ListDiff";
    }

    @Override
    public String opName() {
        return "listdiff";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException();
    }
}
