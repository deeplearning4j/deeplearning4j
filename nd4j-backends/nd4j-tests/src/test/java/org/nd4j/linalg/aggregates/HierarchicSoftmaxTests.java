package org.nd4j.linalg.aggregates;

import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.impl.HierarchicSoftmax;
import org.nd4j.linalg.api.ops.aggregates.impl.SkipGram;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * This tests pack covers simple gradient checks for SkipGram, CBOW and HierarchicSoftmax
 *
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class HierarchicSoftmaxTests extends BaseNd4jTest {


    public HierarchicSoftmaxTests(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void setUp() {

    }

    @Test
    public void testHSGradient1() throws Exception {
        INDArray syn0 = Nd4j.ones(10, 10).assign(0.01f);
        INDArray syn1 = Nd4j.ones(10, 10).assign(0.02f);
        INDArray expTable = Nd4j.create(10000).assign(0.5f);
        INDArray neu1e = Nd4j.create(10);

        INDArray expSyn0 = Nd4j.create(10).assign(0.01f);
        INDArray expSyn1 = Nd4j.create(10).assign(0.020005);
        INDArray expNeu1e = Nd4j.create(10).assign(0.00001f);

        int idxSyn0 = 1;
        int idxSyn1 = 1;
        int code = 0;

        double lr = 0.001;

        HierarchicSoftmax op = new HierarchicSoftmax(syn0, syn1, expTable, neu1e, idxSyn0, idxSyn1, code, lr);

        Nd4j.getExecutioner().exec(op);

        INDArray syn0row = syn0.getRow(idxSyn0);
        INDArray syn1row = syn1.getRow(idxSyn1);

        // expected gradient is 0.0005
        // expected neu1 = 0.00001
        // expected syn1 = 0.020005

        assertEquals(expNeu1e, neu1e);

        assertEquals(expSyn1, syn1row);

        // we hadn't modified syn0 at all yet
        assertEquals(expSyn0, syn0row);
    }

    @Test
    public void testSGGradient1() throws Exception {
        INDArray syn0 = Nd4j.create(10, 10).assign(0.01f);
        INDArray syn1 = Nd4j.create(10, 10).assign(0.02f);
        INDArray syn1Neg = Nd4j.ones(10, 10).assign(0.03f);
        INDArray expTable = Nd4j.create(10000).assign(0.5f);

        double lr = 0.001;

        int idxSyn0 = 0;

        INDArray expSyn0 = Nd4j.create(10).assign(0.01001f);
        INDArray expSyn1_1 = Nd4j.create(10).assign(0.020005);

        INDArray syn0row = syn0.getRow(idxSyn0);

        log.info("syn0row before: {}", Arrays.toString(syn0row.dup().data().asFloat()));

        SkipGram op = new SkipGram(syn0, syn1, syn1Neg, expTable, null, idxSyn0, new int[]{1}, new int[]{0}, 0, 0, 10, lr, 1L);

        Nd4j.getExecutioner().exec(op);

        log.info("syn0row after: {}", Arrays.toString(syn0row.dup().data().asFloat()));

        assertEquals(expSyn0, syn0row);
        assertEquals(expSyn1_1, syn1.getRow(1));
    }

    @Test
    public void testSGGradient2() throws Exception {
        INDArray syn0 = Nd4j.create(10, 10).assign(0.01f);
        INDArray syn1 = Nd4j.create(10, 10).assign(0.02f);
        INDArray syn1Neg = Nd4j.ones(10, 10).assign(0.03f);
        INDArray expTable = Nd4j.create(10000).assign(0.5f);

        double lr = 0.001;

        int idxSyn0 = 0;

        INDArray expSyn0 = Nd4j.create(10).assign(0.01f);
        INDArray expSyn1_1 = Nd4j.create(10).assign(0.020005); // gradient is 0.00005
        INDArray expSyn1_2 = Nd4j.create(10).assign(0.019995f); // gradient is -0.00005


        INDArray syn0row = syn0.getRow(idxSyn0);


        log.info("syn1row2 before: {}", Arrays.toString(syn1.getRow(2).dup().data().asFloat()));

        SkipGram op = new SkipGram(syn0, syn1, null, expTable, null, idxSyn0, new int[]{1, 2}, new int[]{0, 1}, 0, 0, 10, lr, 1L);

        Nd4j.getExecutioner().exec(op);

        /*
            Since expTable contains all-equal values, and only difference for ANY index is code being 0 or 1, syn0 row will stay intact,
            because neu1e will be full of 0.0f, and axpy will have no actual effect
         */
        assertEquals(expSyn0, syn0row);

        // syn1 row 1 modified only once
        assertArrayEquals(expSyn1_1.data().asFloat(), syn1.getRow(1).dup().data().asFloat(), 1e-7f);

        log.info("syn1row2 after: {}", Arrays.toString(syn1.getRow(2).dup().data().asFloat()));

        // syn1 row 2 modified only once
        assertArrayEquals(expSyn1_2.data().asFloat(), syn1.getRow(2).dup().data().asFloat(), 1e-7f);
    }

    /**
     * This particular test does nothing: neither HS or Neh is executed
     *
     * @throws Exception
     */
    @Test
    public void testSGGradientNoOp() throws Exception {
        INDArray syn0 = Nd4j.create(10, 10).assign(0.01f);
        INDArray syn1 = Nd4j.create(10, 10).assign(0.02f);
        INDArray syn1Neg = Nd4j.ones(10, 10).assign(0.03f);
        INDArray expTable = Nd4j.create(10000).assign(0.5f);
        INDArray table = null;

        double lr = 0.001;

        int idxSyn0 = 0;
        INDArray expSyn0 = Nd4j.create(10).assign(0.01f);
        INDArray expSyn1 = syn1.dup();

        SkipGram op = new SkipGram(syn0, syn1, syn1Neg, expTable, table, idxSyn0, new int[]{}, new int[]{}, 0, 0, 10, lr, 1L);

        Nd4j.getExecutioner().exec(op);

        assertEquals(expSyn0, syn0.getRow(idxSyn0));
        assertEquals(expSyn1, syn1);
    }

    @Test
    public void testSGGradientNegative1() throws Exception {
        INDArray syn0 = Nd4j.create(10, 10).assign(0.01f);
        INDArray syn1 = Nd4j.create(10, 10).assign(0.02f);
        INDArray syn1Neg = Nd4j.ones(10, 10).assign(0.03f);
        INDArray expTable = Nd4j.create(10000).assign(0.5f);
        INDArray table = Nd4j.create(100000);

        double lr = 0.001;

        INDArray expSyn0 = Nd4j.create(10).assign(0.01f);

        int idxSyn0 = 1;

        log.info("syn0row1 after: {}", Arrays.toString(syn0.getRow(idxSyn0).dup().data().asFloat()));


        SkipGram op = new SkipGram(syn0, syn1, syn1Neg, expTable, table, idxSyn0, new int[]{}, new int[]{}, 1, 3, 10, lr, 2L);

        Nd4j.getExecutioner().exec(op);

        log.info("syn0row1 after: {}", Arrays.toString(syn0.getRow(idxSyn0).dup().data().asFloat()));

        // we expect syn0 to be equal, since 2 rounds with +- gradients give the same output value for neu1e
        assertEquals(expSyn0, syn0.getRow(idxSyn0));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
