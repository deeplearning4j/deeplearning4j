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

package org.datavec.api.transform.transform;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.util.Arrays;

/**
 * Created by Alex on 18/07/2016.
 */
public class TestJsonYaml {

    @Test
    public void testToFromJsonYaml(){

        Schema schema = new Schema.Builder()
                .addColumnCategorical("Cat","State1","State2")
                .addColumnCategorical("Cat2","State1","State2")
                .addColumnDouble("Dbl")
                .addColumnDouble("Dbl2",null,100.0,true,false)
                .addColumnInteger("Int")
                .addColumnInteger("Int2",0,10)
                .addColumnLong("Long")
                .addColumnLong("Long2",-100L,null)
                .addColumnString("Str")
                .addColumnString("Str2","someregexhere",1,null)
                .addColumnTime("TimeCol", DateTimeZone.UTC)
                .addColumnTime("TimeCol2", DateTimeZone.UTC, null, 1000L)
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToInteger("Cat")
                .categoricalToOneHot("Cat2")
                .integerToCategorical("Cat", Arrays.asList("State1","State2"))
                .stringToCategorical("Str",Arrays.asList("State1","State2"))
                .duplicateColumn("Str","Str2a")
                .removeColumns("Str2a")
                .renameColumn("Str","Str2a")
                .reorderColumns("Cat","Dbl")

                .build();


        System.out.println(tp.toJson());
        System.out.println(tp.toYaml());

    }

}
