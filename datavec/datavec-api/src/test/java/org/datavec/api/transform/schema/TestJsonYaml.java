/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.transform.schema;

import org.datavec.api.transform.metadata.ColumnMetaData;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 18/07/2016.
 */
public class TestJsonYaml {

    @Test
    public void testToFromJsonYaml() {

        Schema schema = new Schema.Builder().addColumnCategorical("Cat", "State1", "State2").addColumnDouble("Dbl")
                        .addColumnDouble("Dbl2", null, 100.0, true, false).addColumnInteger("Int")
                        .addColumnInteger("Int2", 0, 10).addColumnLong("Long").addColumnLong("Long2", -100L, null)
                        .addColumnString("Str").addColumnString("Str2", "someregexhere", 1, null)
                        .addColumnTime("TimeCol", DateTimeZone.UTC)
                        .addColumnTime("TimeCol2", DateTimeZone.UTC, null, 1000L).build();

        String asJson = schema.toJson();
        //        System.out.println(asJson);

        Schema schema2 = Schema.fromJson(asJson);

        int count = schema.numColumns();
        for (int i = 0; i < count; i++) {
            ColumnMetaData c1 = schema.getMetaData(i);
            ColumnMetaData c2 = schema2.getMetaData(i);
            assertEquals(c1, c2);
        }
        assertEquals(schema, schema2);


        String asYaml = schema.toYaml();
        //        System.out.println(asYaml);

        Schema schema3 = Schema.fromYaml(asYaml);
        for (int i = 0; i < schema.numColumns(); i++) {
            ColumnMetaData c1 = schema.getMetaData(i);
            ColumnMetaData c3 = schema3.getMetaData(i);
            assertEquals(c1, c3);
        }
        assertEquals(schema, schema3);
    }

    @Test
    public void testMissingPrimitives() {

        Schema schema = new Schema.Builder().addColumnDouble("Dbl2", null, 100.0, false, false).build();

        String strJson = "{\n" + "  \"Schema\" : {\n" + "    \"columns\" : [ {\n" + "      \"Double\" : {\n"
                        + "        \"name\" : \"Dbl2\",\n" + "        \"maxAllowedValue\" : 100.0\n" +
                        //"        \"allowNaN\" : false,\n" +           //Normally included: but exclude here to test
                        //"        \"allowInfinite\" : false\n" +       //Normally included: but exclude here to test
                        "      }\n" + "    } ]\n" + "  }\n" + "}";

        Schema schema2 = Schema.fromJson(strJson);
        assertEquals(schema, schema2);



        String strYaml = "--- !<Schema>\n" + "columns:\n" + "- !<Double>\n" + "  name: \"Dbl2\"\n"
                        + "  maxAllowedValue: 100.0";
        //"  allowNaN: false\n" +                       //Normally included: but exclude here to test
        //"  allowInfinite: false";                     //Normally included: but exclude here to test

//        Schema schema2a = Schema.fromYaml(strYaml);
//        assertEquals(schema, schema2a);
    }

    @Test
    public void testToFromJsonYamlSequence() {

        Schema schema = new SequenceSchema.Builder().addColumnCategorical("Cat", "State1", "State2")
                        .addColumnDouble("Dbl").addColumnDouble("Dbl2", null, 100.0, true, false)
                        .addColumnInteger("Int").addColumnInteger("Int2", 0, 10).addColumnLong("Long")
                        .addColumnLong("Long2", -100L, null).addColumnString("Str")
                        .addColumnString("Str2", "someregexhere", 1, null).addColumnTime("TimeCol", DateTimeZone.UTC)
                        .addColumnTime("TimeCol2", DateTimeZone.UTC, null, 1000L).build();

        String asJson = schema.toJson();
        //        System.out.println(asJson);

        Schema schema2 = Schema.fromJson(asJson);

        int count = schema.numColumns();
        for (int i = 0; i < count; i++) {
            ColumnMetaData c1 = schema.getMetaData(i);
            ColumnMetaData c2 = schema2.getMetaData(i);
            assertEquals(c1, c2);
        }
        assertEquals(schema, schema2);


        String asYaml = schema.toYaml();
        //        System.out.println(asYaml);

        Schema schema3 = Schema.fromYaml(asYaml);
        for (int i = 0; i < schema.numColumns(); i++) {
            ColumnMetaData c1 = schema.getMetaData(i);
            ColumnMetaData c3 = schema3.getMetaData(i);
            assertEquals(c1, c3);
        }
        assertEquals(schema, schema3);

    }

}
