package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ops.Op;

@Slf4j
public class ConcatV2 extends Concat {


    @Override
    public String opName() {
        return "concatv2";
    }
    @Override
    public String onnxName() {
        return "ConcatV2";
    }

    @Override
    public String tensorflowName() {
        return "ConcatV2";
    }



    @Override
    public Op.Type opType() {
        return Op.Type.SHAPE;
    }
}
