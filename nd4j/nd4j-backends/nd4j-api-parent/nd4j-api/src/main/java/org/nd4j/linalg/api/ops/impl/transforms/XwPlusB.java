package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.NoArgsConstructor;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * Composed op: mmul (X, W) + b
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class XwPlusB extends DynamicCustomOp {


    public XwPlusB(SameDiff sameDiff, SDVariable input, SDVariable weights, SDVariable bias) {
        super(null, sameDiff, new SDVariable[] {input, weights, bias}, false);

    }

    @Override
    public String opName() {
        return "xw_plus_b";
    }


    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow name found for shape " + opName());
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        SDVariable in = arg(0);
        SDVariable w = arg(1);
        SDVariable dLdOut = gradient.get(0);

        SDVariable dLdb = dLdOut.sum(0);
        SDVariable dLdIn = sameDiff.mmul(dLdOut, w, MMulTranspose.builder()
                .transposeB(true)
                .build());
        SDVariable dLdW = sameDiff.mmul(in, dLdOut, MMulTranspose.builder()
                .transposeA(true)
                .build());

        return Arrays.asList(dLdIn, dLdW, dLdb);
    }

}
