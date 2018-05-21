package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;

import java.util.List;

public class DefaultOpConverter extends BaseOp {
    private static DefaultOpConverter INSTANCE = new DefaultOpConverter();
    public static DefaultOpConverter getInstance() {
        return INSTANCE;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "defaultop";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

}
