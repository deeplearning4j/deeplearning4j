/*
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

import org.joda.time.DateTimeZone;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 18/07/2016.
 */
public class TestJsonYamlXml {

    @Test
    public void testToFromJson(){

        Schema schema = new Schema.Builder()
                .addColumnCategorical("Cat","State1","State2")
                .addColumnDouble("Dbl")
                .addColumnInteger("Int",0,10)
                .addColumnLong("Long")
                .addColumnString("Str")
                .addColumnTime("TimeCol", DateTimeZone.UTC)
                .build();

        String asJson = schema.toJson();
        System.out.println(asJson);

        Schema schema2 = Schema.fromJson(asJson);

        assertEquals(schema, schema2);
        System.out.println();
        System.out.println(schema.toYaml());
    }

}
