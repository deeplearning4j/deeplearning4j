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

import org.datavec.api.transform.DataAction;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.condition.column.NullWritableColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator;
import org.datavec.api.transform.sequence.comparator.StringComparator;
import org.datavec.api.transform.sequence.split.SequenceSplitTimeSeparation;
import org.datavec.api.transform.sequence.window.OverlappingTimeWindowFunction;
import org.datavec.api.transform.transform.integer.ReplaceEmptyIntegerWithValueTransform;
import org.datavec.api.transform.transform.integer.ReplaceInvalidWithIntegerTransform;
import org.datavec.api.transform.transform.string.MapAllStringsExceptListTransform;
import org.datavec.api.transform.transform.string.ReplaceEmptyStringTransform;
import org.datavec.api.transform.transform.string.StringListToCategoricalSetTransform;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.comparator.LongWritableComparator;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 18/07/2016.
 */
public class TestJsonYaml {

    @Test
    public void testToFromJsonYaml() {

        Schema schema = new Schema.Builder()
                .addColumnCategorical("Cat", "State1", "State2")
                .addColumnCategorical("Cat2", "State1", "State2")
                .addColumnDouble("Dbl")
                .addColumnDouble("Dbl2", null, 100.0, true, false)
                .addColumnInteger("Int")
                .addColumnInteger("Int2", 0, 10)
                .addColumnLong("Long")
                .addColumnLong("Long2", -100L, null)
                .addColumnString("Str")
                .addColumnString("Str2", "someregexhere", 1, null)
                .addColumnTime("TimeCol", DateTimeZone.UTC)
                .addColumnTime("TimeCol2", DateTimeZone.UTC, null, 1000L)
                .build();

        Map<String, String> map = new HashMap<>();
        map.put("from", "to");
        map.put("anotherFrom", "anotherTo");

        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToInteger("Cat")
                .categoricalToOneHot("Cat2")
                .integerToCategorical("Cat", Arrays.asList("State1", "State2"))
                .stringToCategorical("Str", Arrays.asList("State1", "State2"))
                .duplicateColumn("Str", "Str2a")
                .removeColumns("Str2a")
                .renameColumn("Str2", "Str2a")
                .reorderColumns("Cat", "Dbl")
                .conditionalCopyValueTransform("Dbl", "Dbl2", new DoubleColumnCondition("Dbl", ConditionOp.Equal, 0.0))
                .conditionalReplaceValueTransform("Dbl", new DoubleWritable(1.0), new DoubleColumnCondition("Dbl", ConditionOp.Equal, 1.0))
                .doubleColumnsMathOp("NewDouble", MathOp.Add, "Dbl", "Dbl2")
                .doubleMathOp("Dbl", MathOp.Add, 1.0)
                .integerColumnsMathOp("NewInt", MathOp.Subtract, "Int", "Int2")
                .integerMathOp("Int", MathOp.Multiply, 2)
                .transform(new ReplaceEmptyIntegerWithValueTransform("Int", 1))
                .transform(new ReplaceInvalidWithIntegerTransform("Int", 1))
                .longColumnsMathOp("Long", MathOp.Multiply, "Long", "Long2")
                .longMathOp("Long", MathOp.ScalarMax, 0)
                .transform(new MapAllStringsExceptListTransform("Str", "Other", Arrays.asList("Ok", "SomeVal")))
                .stringRemoveWhitespaceTransform("Str")
                .transform(new ReplaceEmptyStringTransform("Str", "WasEmpty"))
                .transform(new StringListToCategoricalSetTransform("Str", Arrays.asList("StrA", "StrB"), Arrays.asList("StrA", "StrB"), ","))
                .stringMapTransform("Str2a", map)
                .transform(new DeriveColumnsFromTimeTransform.Builder("TimeCol")
                        .addIntegerDerivedColumn("Hour", DateTimeFieldType.hourOfDay())
                        .addStringDerivedColumn("Date", "YYYY-MM-dd", DateTimeZone.UTC)
                        .build())
                .stringToTimeTransform("Str2a", "YYYY-MM-dd hh:mm:ss", DateTimeZone.UTC)
                .timeMathOp("TimeCol2", MathOp.Add, 1, TimeUnit.HOURS)

                //Filters:
                .filter(new FilterInvalidValues("Cat", "Str2a"))
                .filter(new ConditionFilter(new NullWritableColumnCondition("Long")))

                //Convert to/from sequence
                .convertToSequence("Int", new NumericalColumnComparator("TimeCol2"))
                .convertFromSequence()

                //Sequence split
                .convertToSequence("Int", new StringComparator("Str2a"))
                .splitSequence(new SequenceSplitTimeSeparation("TimeCol2", 1, TimeUnit.HOURS))

                //Reducers and reduce by window:
                .reduce(new Reducer.Builder(ReduceOp.TakeFirst)
                        .keyColumns("TimeCol2")
                        .countColumns("Cat")
                        .sumColumns("Dbl").build())
                .reduceSequenceByWindow(new Reducer.Builder(ReduceOp.TakeFirst)
                                .countColumns("Cat2")
                                .stdevColumns("Dbl2").build(),
                        new OverlappingTimeWindowFunction.Builder()
                                .timeColumn("TimeCol2")
                                .addWindowStartTimeColumn(true)
                                .addWindowEndTimeColumn(true)
                                .windowSize(1, TimeUnit.HOURS)
                                .offset(5, TimeUnit.MINUTES)
                                .windowSeparation(15, TimeUnit.MINUTES)
                                .excludeEmptyWindows(true)
                                .build())

                //Calculate sorted rank
                .convertFromSequence()
                .calculateSortedRank("rankColName","TimeCol2",new LongWritableComparator())

                .build();

        String asJson = tp.toJson();
        String asYaml = tp.toYaml();

//        System.out.println(asJson);
//        System.out.println("\n\n\n");
//        System.out.println(asYaml);


        TransformProcess tpFromJson = TransformProcess.fromJson(asJson);
        TransformProcess tpFromYaml = TransformProcess.fromYaml(asYaml);

        List<DataAction> daList = tp.getActionList();
        List<DataAction> daListJson = tpFromJson.getActionList();
        List<DataAction> daListYaml = tpFromYaml.getActionList();

        for (int i = 0; i < daList.size(); i++) {
            DataAction da1 = daList.get(i);
            DataAction da2 = daListJson.get(i);
            DataAction da3 = daListYaml.get(i);

//            System.out.println(i + "\t" + da1);

            assertEquals(da1, da2);
            assertEquals(da1, da3);
        }

        assertEquals(tp, tpFromJson);
        assertEquals(tp, tpFromYaml);

    }

}
