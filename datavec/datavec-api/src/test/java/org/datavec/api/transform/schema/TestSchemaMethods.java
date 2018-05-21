/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.schema;

import org.datavec.api.transform.ColumnType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 04/09/2016.
 */
public class TestSchemaMethods {

    @Test
    public void testNumberedColumnAdding() {

        Schema schema = new Schema.Builder().addColumnsDouble("doubleCol_%d", 0, 2).addColumnsLong("longCol_%d", 3, 5)
                        .addColumnsString("stringCol_%d", 6, 8).build();

        assertEquals(9, schema.numColumns());

        for (int i = 0; i < 9; i++) {
            if (i <= 2) {
                assertEquals("doubleCol_" + i, schema.getName(i));
                assertEquals(ColumnType.Double, schema.getType(i));
            } else if (i <= 5) {
                assertEquals("longCol_" + i, schema.getName(i));
                assertEquals(ColumnType.Long, schema.getType(i));
            } else {
                assertEquals("stringCol_" + i, schema.getName(i));
                assertEquals(ColumnType.String, schema.getType(i));
            }
        }

    }

}
