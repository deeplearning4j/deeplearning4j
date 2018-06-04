package org.nd4j.autodiff.gradcheck;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.nd4j.linalg.BaseNd4jTest.assertArrayEquals;

public class TestMisc {


    @Test
    public void testShapeFn() {

        INDArray in = Nd4j.create(new long[]{1, 2});

        List<long[]> shapes = Nd4j.getExecutioner().calculateOutputShape(DynamicCustomOp.builder("shape")
                .addInputs(in)
                .build());

        assertEquals(1, shapes.size());

        assertArrayEquals(new long[]{2}, shapes.get(0));

    }

    @Test
    public void testShapeFn2() {

        INDArray i = Nd4j.create(1,3);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", i);
        SDVariable shape = sd.shape(var);
        SDVariable sum = sd.sum(shape);

        sd.execAndEndResult();
        sd.execBackwards();
    }


    @Test
    public void testMergeRank1(){

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", Nd4j.create(new long[]{1}).assign(5));

        SDVariable merged = sd.mergeAvg(var);
        SDVariable sum = sd.sum(merged);

        sd.execAndEndResult();
        sd.execBackwards();

        INDArray out = merged.getArr();
        assertEquals(1, out.rank());

        INDArray inGrad = var.getGradient().getArr();
        assertEquals(1, inGrad.rank());         //Fails here, getting rank 2
    }

    @Test
    public void testDiagPart() {

        INDArray i = Nd4j.create(5,5);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("in", i);
        SDVariable diag = sd.diagPart(var);

        INDArray out = sd.execAndEndResult();
        assertEquals(1, out.rank());

    }

    @Test
    public void testDiagShapeFn() {

        INDArray i = Nd4j.create(5,5);

        CustomOp op = DynamicCustomOp.builder("diag_part")
                .addInputs(i).build();

        List<long[]> outShape = Nd4j.getExecutioner().calculateOutputShape(op);

        assertEquals(1, outShape.size());
        assertArrayEquals(new long[]{5}, outShape.get(0));
    }



}
