package org.nd4j.linalg.ops.elementwise;

import org.nd4j.linalg.util.Shape;
import org.junit.Test;
import static org.junit.Assert.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class ShapeTests {
    private static Logger log = LoggerFactory.getLogger(ShapeTests.class);
    @Test
    public void testShape() {
        assertTrue(Shape.isColumnVectorShape(new int[]{10,1}));
        assertFalse(Shape.isColumnVectorShape(new int[]{1,10}));
        assertTrue(Shape.isRowVectorShape(new int[]{1}));
        assertTrue(Shape.isRowVectorShape(new int[]{10}));
        assertTrue(Shape.isRowVectorShape(new int[]{1,10}));
    }


}
