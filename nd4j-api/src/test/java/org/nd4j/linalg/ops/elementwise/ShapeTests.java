/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.ops.elementwise;

import org.junit.Test;
import org.nd4j.linalg.util.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * @author Adam Gibson
 */
public class ShapeTests {
    private static Logger log = LoggerFactory.getLogger(ShapeTests.class);

    @Test
    public void testShape() {
        assertTrue(Shape.isColumnVectorShape(new int[]{10, 1}));
        assertFalse(Shape.isColumnVectorShape(new int[]{1, 10}));
        assertTrue(Shape.isRowVectorShape(new int[]{1}));
        assertTrue(Shape.isRowVectorShape(new int[]{10}));
        assertTrue(Shape.isRowVectorShape(new int[]{1, 10}));
    }


}
