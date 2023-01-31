package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@UserDefinedOp
public class TestUdf extends UserDefinedCustomOp {
    public TestUdf() {
        super();
    }

    public TestUdf(SameDiff sameDiff, SDVariable arg) {
        super(sameDiff, arg);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(dataTypes.get(0));
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {

    }

    @Override
    public Object getValue(Field property) {
        return null;
    }

    @Override
    public void setValueFor(Field target, Object value) {

    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return Collections.emptyMap();
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    @Override
    public String opName() {
        return "test_udf";
    }

    @Override
    public void configureFromArguments() {

    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    @Override
    public boolean isInplaceCall() {
        return false;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        return Arrays.asList(inputArguments.get(0).shapeDescriptor());
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc) {
        return Arrays.asList(oc.getInputArrays().get(0).shapeDescriptor());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(sameDiff.onesLike(f1.get(0)));
    }

    @Override
    public void exec() {

    }

    @Override
    public void exec(OpContext opContext) {

    }
}
