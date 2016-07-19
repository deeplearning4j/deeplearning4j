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

import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.integer.ReplaceEmptyIntegerWithValueTransform;
import org.datavec.api.transform.transform.integer.ReplaceInvalidWithIntegerTransform;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.transform.string.MapAllStringsExceptListTransform;
import org.datavec.api.transform.transform.string.ReplaceEmptyStringTransform;
import org.datavec.api.transform.transform.string.StringListToCategoricalSetTransform;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.DoubleWritable;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

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

        Map<String,String> map = new HashMap<>();
        map.put("from","to");
        map.put("anotherFrom","anotherTo");

        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToInteger("Cat")
                .categoricalToOneHot("Cat2")
                .integerToCategorical("Cat", Arrays.asList("State1","State2"))
                .stringToCategorical("Str",Arrays.asList("State1","State2"))
                .duplicateColumn("Str","Str2a")
                .removeColumns("Str2a")
                .renameColumn("Str2","Str2a")
                .reorderColumns("Cat","Dbl")
                .conditionalCopyValueTransform("Dbl","Dbl2",new DoubleColumnCondition("Dbl", ConditionOp.Equal, 0.0))
                .conditionalReplaceValueTransform("Dbl",new DoubleWritable(1.0), new DoubleColumnCondition("Dbl", ConditionOp.Equal, 1.0))
                .doubleColumnsMathOp("NewDouble", MathOp.Add, "Dbl","Dbl2")
                .doubleMathOp("Dbl",MathOp.Add, 1.0)
                .integerColumnsMathOp("NewInt", MathOp.Subtract, "Int", "Int2")
                .integerMathOp("Int", MathOp.Multiply, 2)
                .transform(new ReplaceEmptyIntegerWithValueTransform("Int",1))
                .transform(new ReplaceInvalidWithIntegerTransform("Int",1))
                .longColumnsMathOp("Long",MathOp.Multiply, "Long", "Long2")
                .longMathOp("Long", MathOp.ScalarMax, 0)
                .transform(new MapAllStringsExceptListTransform("Str","Other",Arrays.asList("Ok","SomeVal")))
                .stringRemoveWhitespaceTransform("Str")
                .transform(new ReplaceEmptyStringTransform("Str","WasEmpty"))
                .transform(new StringListToCategoricalSetTransform("Str",Arrays.asList("StrA","StrB"),Arrays.asList("StrA","StrB"),","))
                .stringMapTransform("Str2a",map)
                .transform(new DeriveColumnsFromTimeTransform.Builder("TimeCol")
                        .addIntegerDerivedColumn("Hour", DateTimeFieldType.hourOfDay())
                        .addStringDerivedColumn("Date","YYYY-MM-dd", DateTimeZone.UTC)
                        .build())
                .stringToTimeTransform("Str2a","YYYY-MM-dd hh:mm:ss", DateTimeZone.UTC)
                .timeMathOp("TimeCol2",MathOp.Add,1, TimeUnit.HOURS)
                .build();


        System.out.println(tp.toJson());
        System.out.println(tp.toYaml());

    }

}
