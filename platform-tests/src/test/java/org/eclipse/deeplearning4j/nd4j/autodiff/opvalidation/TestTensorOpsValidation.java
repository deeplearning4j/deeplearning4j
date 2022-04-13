package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;


import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import javax.annotation.concurrent.NotThreadSafe;
import java.util.Map;

import static org.junit.Assert.assertEquals;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@NotThreadSafe
public class TestTensorOpsValidation extends BaseOpValidation {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSize(Nd4jBackend backend, TestInfo testInfo) {
        SameDiff sameDiff = SameDiff.create();
        TensorArray tensorArray = sameDiff.tensorArray(DataType.DOUBLE);
        SDVariable one = sameDiff.var("x", Nd4j.ones(1));
        SDVariable two = sameDiff.var(one);
        SDVariable write = tensorArray.write(one, 0, two);
        write.addControlDependency(two);
        SDVariable three = tensorArray.write(one,1,two);
        three.addControlDependency(write);
        SDVariable size =tensorArray.size(tensorArray.getVar());
        size.addControlDependency(three);
        Map<String, INDArray> output = sameDiff.output((Map<String, INDArray>) null, size.name());
        assertEquals(Nd4j.createFromArray(2).reshape(output.get(size.name()).shape()),output.get(size.name()));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTensorInsertRemove(Nd4jBackend backend, TestInfo testInfo) {
        SameDiff sameDiff = SameDiff.create();
        TensorArray tensorArray = sameDiff.tensorArray(DataType.DOUBLE);
        SDVariable one = sameDiff.var("x", Nd4j.ones(1));
        SDVariable two = sameDiff.var(one);
        SDVariable write = tensorArray.write(one, 0, two);
        write.addControlDependency(two);
        SDVariable read = tensorArray.read(0);
        read.addControlDependency(write);
        Map<String, INDArray> output = sameDiff.output((Map<String, INDArray>) null, read.name());
        assertEquals(Nd4j.ones(DataType.DOUBLE,1).reshape(1),output.get(read.name()).reshape(1));
        SDVariable remove = tensorArray.remove(tensorArray.getVar(), 0);
        remove.addControlDependency(read);
        SDVariable read2  = tensorArray.size(tensorArray.getVar());
        output = sameDiff.output((Map<String, INDArray>) null, read2.name());
        assertEquals(Nd4j.zeros(DataType.INT32,1).reshape(1),output.get(read2.name()).reshape(1));
    }

}
