package org.nd4j.linalg.specials;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacpp.IntPointer;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class LongTests extends BaseNd4jTest {

    DataBuffer.Type initialType;

    public LongTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Test
    @Ignore
    public void testSomething1() {
        // we create 2D array, total nr. of elements is 2.4B elements, > MAX_INT
        INDArray huge = Nd4j.create(8000000, 300);

// we apply element-wise scalar ops, just to make sure stuff still works
        huge.subi(0.5).divi(2);


// now we're checking different rows, they should NOT equal
        INDArray row0 = huge.getRow(100001).assign(1.0);
        INDArray row1 = huge.getRow(100002).assign(2.0);
        assertNotEquals(row0, row1);


// same idea, but this code is broken: rowA and rowB will be pointing to the same offset
        INDArray rowA = huge.getRow(huge.rows() - 3);
        INDArray rowB = huge.getRow(huge.rows() - 10);

// safety check, to see if we're really working on the same offset.
        rowA.addi(1.0);

// and this fails, so rowA and rowB are pointing to the same offset, despite different getRow() arguments were used
        assertNotEquals(rowA, rowB);
    }

    @Test
    public void testSomething2() {
        // we create 2D array, total nr. of elements is 2.4B elements, > MAX_INT
        INDArray huge = Nd4j.create(100, 10);

// we apply element-wise scalar ops, just to make sure stuff still works
        huge.subi(0.5).divi(2);


// now we're checking different rows, they should NOT equal
        INDArray row0 = huge.getRow(73).assign(1.0);
        INDArray row1 = huge.getRow(74).assign(2.0);
        assertNotEquals(row0, row1);


// same idea, but this code is broken: rowA and rowB will be pointing to the same offset
        INDArray rowA = huge.getRow(huge.rows() - 3);
        INDArray rowB = huge.getRow(huge.rows() - 10);

// safety check, to see if we're really working on the same offset.
        rowA.addi(1.0);

// and this fails, so rowA and rowB are pointing to the same offset, despite different getRow() arguments were used
        assertNotEquals(rowA, rowB);
    }

    @Test
    public void testLongTadOffsets1() {
        INDArray huge = Nd4j.create(230000000, 10);

        Pair<DataBuffer, DataBuffer> tad = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(huge, 1);

        assertEquals(230000000, tad.getSecond().length());
    }

    @Test
    public void testLongTadOp1() {

        double exp = Transforms.manhattanDistance(Nd4j.create(1000).assign(1.0), Nd4j.create(1000).assign(2.0));

        INDArray hugeX = Nd4j.create(2200000, 1000).assign(1.0);
        INDArray hugeY = Nd4j.create(2200000, 1000).assign(2.0);

        for (int x = 0; x < hugeX.rows(); x++){
            assertEquals("Failed at row " + x, 1000, hugeX.getRow(x).sumNumber().intValue());
        }

        INDArray result = Transforms.allManhattanDistances(hugeX, hugeY, 1);
        for (int x = 0; x < hugeX.rows(); x++) {
            assertEquals(exp, result.getDouble(x), 1e-5);
        }
    }

    @Test
    public void testLongTadOp2() {

        INDArray hugeX = Nd4j.create(2300000, 1000).assign(1.0);
        hugeX.addiRowVector(Nd4j.create(1000).assign(2.0));

        for (int x = 0; x < hugeX.rows(); x++){
            assertEquals("Failed at row " + x, 3000, hugeX.getRow(x).sumNumber().intValue());
        }
    }

    @Test
    public void testLongTadOp2_micro() {

        INDArray hugeX = Nd4j.create(230, 1000).assign(1.0);
        hugeX.addiRowVector(Nd4j.create(1000).assign(2.0));

        for (int x = 0; x < hugeX.rows(); x++){
            assertEquals("Failed at row " + x, 3000, hugeX.getRow(x).sumNumber().intValue());
        }
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
