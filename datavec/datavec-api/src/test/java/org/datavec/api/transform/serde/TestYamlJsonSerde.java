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

package org.datavec.api.transform.serde;

import org.datavec.api.transform.*;
import org.datavec.api.transform.condition.BooleanCondition;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.*;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.ConvertFromSequence;
import org.datavec.api.transform.sequence.ConvertToSequence;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator;
import org.datavec.api.transform.sequence.comparator.StringComparator;
import org.datavec.api.transform.sequence.split.SequenceSplitTimeSeparation;
import org.datavec.api.transform.sequence.split.SplitMaxLengthSequence;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.datavec.api.transform.transform.categorical.IntegerToCategoricalTransform;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.datavec.api.transform.transform.column.DuplicateColumnsTransform;
import org.datavec.api.transform.transform.column.RemoveColumnsTransform;
import org.datavec.api.transform.transform.column.RenameColumnsTransform;
import org.datavec.api.transform.transform.column.ReorderColumnsTransform;
import org.datavec.api.transform.transform.condition.ConditionalCopyValueTransform;
import org.datavec.api.transform.transform.doubletransform.*;
import org.datavec.api.transform.transform.integer.IntegerColumnsMathOpTransform;
import org.datavec.api.transform.transform.integer.IntegerMathOpTransform;
import org.datavec.api.transform.transform.integer.ReplaceEmptyIntegerWithValueTransform;
import org.datavec.api.transform.transform.integer.ReplaceInvalidWithIntegerTransform;
import org.datavec.api.transform.transform.longtransform.LongColumnsMathOpTransform;
import org.datavec.api.transform.transform.longtransform.LongMathOpTransform;
import org.datavec.api.transform.transform.string.*;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.transform.transform.time.TimeMathOpTransform;
import org.datavec.api.writable.comparator.DoubleWritableComparator;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.junit.Test;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 20/07/2016.
 */
public class TestYamlJsonSerde {

    public static YamlSerializer y = new YamlSerializer();
    public static JsonSerializer j = new JsonSerializer();

    @Test
    public void testTransforms() {

        Map<String, String> map = new HashMap<>();
        map.put("A", "A1");
        map.put("B", "B1");

        Transform[] transforms = new Transform[] {new CategoricalToIntegerTransform("ColName"),
                        new CategoricalToOneHotTransform("ColName"),
                        new IntegerToCategoricalTransform("ColName", Arrays.asList("State0", "State1")),
                        new StringToCategoricalTransform("ColName", Arrays.asList("State0", "State1")),
                        new DuplicateColumnsTransform(Arrays.asList("Dup1", "Dup2"),
                                        Arrays.asList("NewName1", "NewName2")),
                        new RemoveColumnsTransform("R1", "R2"),
                        new RenameColumnsTransform(Arrays.asList("N1", "N2"), Arrays.asList("NewN1", "NewN2")),
                        new ReorderColumnsTransform("A", "B"),
                        new DoubleColumnsMathOpTransform("NewName", MathOp.Subtract, "A", "B"),
                        new DoubleMathOpTransform("ColName", MathOp.Multiply, 2.0),
                        new Log2Normalizer("ColName", 1.0, 0.0, 2.0), new MinMaxNormalizer("ColName", 0, 100, -1, 1),
                        new StandardizeNormalizer("ColName", 20, 5), new SubtractMeanNormalizer("ColName", 10),
                        new IntegerColumnsMathOpTransform("NewName", MathOp.Multiply, "A", "B"),
                        new IntegerMathOpTransform("ColName", MathOp.Add, 10),
                        new ReplaceEmptyIntegerWithValueTransform("Col", 3),
                        new ReplaceInvalidWithIntegerTransform("Col", 3),
                        new LongColumnsMathOpTransform("NewName", MathOp.Multiply, "A", "B"),
                        new LongMathOpTransform("Col", MathOp.ScalarMax, 5L),
                        new MapAllStringsExceptListTransform("Col", "NewVal", Arrays.asList("E1", "E2")),
                        new RemoveWhiteSpaceTransform("Col"), new ReplaceEmptyStringTransform("Col", "WasEmpty"),
                        new ReplaceStringTransform("Col_A", map),
                        new StringListToCategoricalSetTransform("Col", Arrays.asList("A", "B"), Arrays.asList("A", "B"),
                                        ","),
                        new StringMapTransform("Col", map),
                        new DeriveColumnsFromTimeTransform.Builder("TimeColName")
                                        .addIntegerDerivedColumn("Hour", DateTimeFieldType.hourOfDay())
                                        .addStringDerivedColumn("DateTime", "YYYY-MM-dd hh:mm:ss", DateTimeZone.UTC)
                                        .build(),
                        new StringToTimeTransform("TimeCol", "YYYY-MM-dd hh:mm:ss", DateTimeZone.UTC),
                        new TimeMathOpTransform("TimeCol", MathOp.Add, 1, TimeUnit.HOURS)};

        for (Transform t : transforms) {
            String yaml = y.serialize(t);
            String json = j.serialize(t);

            //            System.out.println(yaml);
//                        System.out.println(json);
            //            System.out.println();

//            Transform t2 = y.deserializeTransform(yaml);
            Transform t3 = j.deserializeTransform(json);
//            assertEquals(t, t2);
            assertEquals(t, t3);
        }


        String tArrAsYaml = y.serialize(transforms);
        String tArrAsJson = j.serialize(transforms);
        String tListAsYaml = y.serializeTransformList(Arrays.asList(transforms));
        String tListAsJson = j.serializeTransformList(Arrays.asList(transforms));

        //        System.out.println("\n\n\n\n");
        //        System.out.println(tListAsYaml);

        List<Transform> lFromYaml = y.deserializeTransformList(tListAsYaml);
        List<Transform> lFromJson = j.deserializeTransformList(tListAsJson);

        assertEquals(Arrays.asList(transforms), y.deserializeTransformList(tArrAsYaml));
        assertEquals(Arrays.asList(transforms), j.deserializeTransformList(tArrAsJson));
        assertEquals(Arrays.asList(transforms), lFromYaml);
        assertEquals(Arrays.asList(transforms), lFromJson);
    }


    @Test
    public void testTransformsRegexBrackets() {

        Schema schema = new Schema.Builder().addColumnString("someCol").addColumnString("otherCol").build();

        Transform[] transforms = new Transform[] {
                        new ConditionalCopyValueTransform("someCol", "otherCol",
                                        new StringRegexColumnCondition("someCol", "\\d")),
                        new ConditionalCopyValueTransform("someCol", "otherCol",
                                        new StringRegexColumnCondition("someCol", "\\D+")),
                        new ConditionalCopyValueTransform("someCol", "otherCol",
                                        new StringRegexColumnCondition("someCol", "\".*\"")),
                        new ConditionalCopyValueTransform("someCol", "otherCol",
                                        new StringRegexColumnCondition("someCol", "[]{}()][}{)("))};

        for (Transform t : transforms) {
            String json = j.serialize(t);

            Transform t3 = j.deserializeTransform(json);
            assertEquals(t, t3);

            TransformProcess tp = new TransformProcess.Builder(schema).transform(t).build();

            String tpJson = j.serialize(tp);

            TransformProcess fromJson = TransformProcess.fromJson(tpJson);

            assertEquals(tp, fromJson);
        }


        String tArrAsYaml = y.serialize(transforms);
        String tArrAsJson = j.serialize(transforms);
        String tListAsYaml = y.serializeTransformList(Arrays.asList(transforms));
        String tListAsJson = j.serializeTransformList(Arrays.asList(transforms));

        List<Transform> lFromYaml = y.deserializeTransformList(tListAsYaml);
        List<Transform> lFromJson = j.deserializeTransformList(tListAsJson);

        assertEquals(Arrays.asList(transforms), y.deserializeTransformList(tArrAsYaml));
        assertEquals(Arrays.asList(transforms), j.deserializeTransformList(tArrAsJson));
        assertEquals(Arrays.asList(transforms), lFromYaml);
        assertEquals(Arrays.asList(transforms), lFromJson);
    }


    @Test
    public void testFilters() {
        Filter[] filters = new Filter[] {new FilterInvalidValues("A", "B"),
                        new ConditionFilter(new DoubleColumnCondition("Col", ConditionOp.GreaterOrEqual, 10.0))};

        for (Filter f : filters) {
            String yaml = y.serialize(f);
            String json = j.serialize(f);

            //            System.out.println(yaml);
            //            System.out.println(json);
            //            System.out.println();

            Filter t2 = y.deserializeFilter(yaml);
            Filter t3 = j.deserializeFilter(json);
            assertEquals(f, t2);
            assertEquals(f, t3);
        }

        String arrAsYaml = y.serialize(filters);
        String arrAsJson = j.serialize(filters);
        String listAsYaml = y.serializeFilterList(Arrays.asList(filters));
        String listAsJson = j.serializeFilterList(Arrays.asList(filters));

        //        System.out.println("\n\n\n\n");
        //        System.out.println(listAsYaml);

        List<Filter> lFromYaml = y.deserializeFilterList(listAsYaml);
        List<Filter> lFromJson = j.deserializeFilterList(listAsJson);

        assertEquals(Arrays.asList(filters), y.deserializeFilterList(arrAsYaml));
        assertEquals(Arrays.asList(filters), j.deserializeFilterList(arrAsJson));
        assertEquals(Arrays.asList(filters), lFromYaml);
        assertEquals(Arrays.asList(filters), lFromJson);
    }

    @Test
    public void testConditions() {
        Set<String> setStr = new HashSet<>();
        setStr.add("A");
        setStr.add("B");

        Set<Double> setD = new HashSet<>();
        setD.add(1.0);
        setD.add(2.0);

        Set<Integer> setI = new HashSet<>();
        setI.add(1);
        setI.add(2);

        Set<Long> setL = new HashSet<>();
        setL.add(1L);
        setL.add(2L);

        Condition[] conditions = new Condition[] {new CategoricalColumnCondition("Col", ConditionOp.Equal, "A"),
                        new CategoricalColumnCondition("Col", ConditionOp.NotInSet, setStr),
                        new DoubleColumnCondition("Col", ConditionOp.Equal, 1.0),
                        new DoubleColumnCondition("Col", ConditionOp.InSet, setD),

                        new IntegerColumnCondition("Col", ConditionOp.Equal, 1),
                        new IntegerColumnCondition("Col", ConditionOp.InSet, setI),

                        new LongColumnCondition("Col", ConditionOp.Equal, 1),
                        new LongColumnCondition("Col", ConditionOp.InSet, setL),

                        new NullWritableColumnCondition("Col"),

                        new StringColumnCondition("Col", ConditionOp.NotEqual, "A"),
                        new StringColumnCondition("Col", ConditionOp.InSet, setStr),

                        new TimeColumnCondition("Col", ConditionOp.Equal, 1L),
                        new TimeColumnCondition("Col", ConditionOp.InSet, setL),

                        new StringRegexColumnCondition("Col", "Regex"),

                        BooleanCondition.OR(
                                        BooleanCondition.AND(
                                                        new CategoricalColumnCondition("Col", ConditionOp.Equal, "A"),
                                                        new LongColumnCondition("Col2", ConditionOp.Equal, 1)),
                                        BooleanCondition.NOT(new TimeColumnCondition("Col3", ConditionOp.Equal, 1L)))};

        for (Condition c : conditions) {
            String yaml = y.serialize(c);
            String json = j.serialize(c);

//                        System.out.println(yaml);
//                        System.out.println(json);
            //            System.out.println();

            Condition t2 = y.deserializeCondition(yaml);
            Condition t3 = j.deserializeCondition(json);
            assertEquals(c, t2);
            assertEquals(c, t3);
        }

        String arrAsYaml = y.serialize(conditions);
        String arrAsJson = j.serialize(conditions);
        String listAsYaml = y.serializeConditionList(Arrays.asList(conditions));
        String listAsJson = j.serializeConditionList(Arrays.asList(conditions));

        //        System.out.println("\n\n\n\n");
        //        System.out.println(listAsYaml);

        List<Condition> lFromYaml = y.deserializeConditionList(listAsYaml);
        List<Condition> lFromJson = j.deserializeConditionList(listAsJson);

        assertEquals(Arrays.asList(conditions), y.deserializeConditionList(arrAsYaml));
        assertEquals(Arrays.asList(conditions), j.deserializeConditionList(arrAsJson));
        assertEquals(Arrays.asList(conditions), lFromYaml);
        assertEquals(Arrays.asList(conditions), lFromJson);
    }

    @Test
    public void testReducer() {
        IAssociativeReducer[] reducers = new IAssociativeReducer[] {new Reducer.Builder(ReduceOp.TakeFirst)
                        .keyColumns("KeyCol").stdevColumns("Stdev").minColumns("min").countUniqueColumns("B").build()};

        for (IAssociativeReducer r : reducers) {
            String yaml = y.serialize(r);
            String json = j.serialize(r);

            //            System.out.println(yaml);
            //            System.out.println(json);
            //            System.out.println();

            IAssociativeReducer t2 = y.deserializeReducer(yaml);
            IAssociativeReducer t3 = j.deserializeReducer(json);
            assertEquals(r, t2);
            assertEquals(r, t3);
        }

        String arrAsYaml = y.serialize(reducers);
        String arrAsJson = j.serialize(reducers);
        String listAsYaml = y.serializeReducerList(Arrays.asList(reducers));
        String listAsJson = j.serializeReducerList(Arrays.asList(reducers));

        //        System.out.println("\n\n\n\n");
        //        System.out.println(listAsYaml);

        List<IAssociativeReducer> lFromYaml = y.deserializeReducerList(listAsYaml);
        List<IAssociativeReducer> lFromJson = j.deserializeReducerList(listAsJson);

        assertEquals(Arrays.asList(reducers), y.deserializeReducerList(arrAsYaml));
        assertEquals(Arrays.asList(reducers), j.deserializeReducerList(arrAsJson));
        assertEquals(Arrays.asList(reducers), lFromYaml);
        assertEquals(Arrays.asList(reducers), lFromJson);
    }

    @Test
    public void testSequenceComparator() {
        SequenceComparator[] comparators = new SequenceComparator[] {new NumericalColumnComparator("Col", true),
                        new StringComparator("Col")};

        for (SequenceComparator f : comparators) {
            String yaml = y.serialize(f);
            String json = j.serialize(f);

            //            System.out.println(yaml);
            //            System.out.println(json);
            //            System.out.println();

            SequenceComparator t2 = y.deserializeSequenceComparator(yaml);
            SequenceComparator t3 = j.deserializeSequenceComparator(json);
            assertEquals(f, t2);
            assertEquals(f, t3);
        }

        String arrAsYaml = y.serialize(comparators);
        String arrAsJson = j.serialize(comparators);
        String listAsYaml = y.serializeSequenceComparatorList(Arrays.asList(comparators));
        String listAsJson = j.serializeSequenceComparatorList(Arrays.asList(comparators));

        //        System.out.println("\n\n\n\n");
        //        System.out.println(listAsYaml);

        List<SequenceComparator> lFromYaml = y.deserializeSequenceComparatorList(listAsYaml);
        List<SequenceComparator> lFromJson = j.deserializeSequenceComparatorList(listAsJson);

        assertEquals(Arrays.asList(comparators), y.deserializeSequenceComparatorList(arrAsYaml));
        assertEquals(Arrays.asList(comparators), j.deserializeSequenceComparatorList(arrAsJson));
        assertEquals(Arrays.asList(comparators), lFromYaml);
        assertEquals(Arrays.asList(comparators), lFromJson);
    }

    @Test
    public void testCalculateSortedRank() {
        CalculateSortedRank rank = new CalculateSortedRank("RankCol", "SortOnCol", new DoubleWritableComparator());

        String asYaml = y.serialize(rank);
        String asJson = j.serialize(rank);

        CalculateSortedRank yRank = y.deserializeSortedRank(asYaml);
        CalculateSortedRank jRank = j.deserializeSortedRank(asJson);

        assertEquals(rank, yRank);
        assertEquals(rank, jRank);
    }

    @Test
    public void testSequenceSplit() {
        SequenceSplit[] splits = new SequenceSplit[] {new SequenceSplitTimeSeparation("col", 1, TimeUnit.HOURS),
                        new SplitMaxLengthSequence(100, false)};

        for (SequenceSplit f : splits) {
            String yaml = y.serialize(f);
            String json = j.serialize(f);

            //            System.out.println(yaml);
            //            System.out.println(json);
            //            System.out.println();

            SequenceSplit t2 = y.deserializeSequenceSplit(yaml);
            SequenceSplit t3 = j.deserializeSequenceSplit(json);
            assertEquals(f, t2);
            assertEquals(f, t3);
        }
    }


    @Test
    public void testDataAction() {
        DataAction[] dataActions = new DataAction[] {new DataAction(new CategoricalToIntegerTransform("Col")),
                        new DataAction(new ConditionFilter(new DoubleColumnCondition("Col", ConditionOp.Equal, 1))),
                        new DataAction(new ConvertToSequence("KeyCol", new NumericalColumnComparator("Col", true))),
                        new DataAction(new ConvertFromSequence()),
                        new DataAction(new SequenceSplitTimeSeparation("TimeCol", 1, TimeUnit.HOURS)),
                        new DataAction(new Reducer.Builder(ReduceOp.TakeFirst).build()),
                        new DataAction(new CalculateSortedRank("NewCol", "SortCol", new DoubleWritableComparator()))};

        for (DataAction f : dataActions) {
            String yaml = y.serialize(f);
            String json = j.serialize(f);

            //            System.out.println(yaml);
            //            System.out.println(json);
            //            System.out.println();

            DataAction t2 = y.deserializeDataAction(yaml);
            DataAction t3 = j.deserializeDataAction(json);
            assertEquals(f, t2);
            assertEquals(f, t3);
        }

        String arrAsYaml = y.serialize(dataActions);
        String arrAsJson = j.serialize(dataActions);
        String listAsYaml = y.serializeDataActionList(Arrays.asList(dataActions));
        String listAsJson = j.serializeDataActionList(Arrays.asList(dataActions));

        //        System.out.println("\n\n\n\n");
        //        System.out.println(listAsYaml);

        List<DataAction> lFromYaml = y.deserializeDataActionList(listAsYaml);
        List<DataAction> lFromJson = j.deserializeDataActionList(listAsJson);

        assertEquals(Arrays.asList(dataActions), y.deserializeDataActionList(arrAsYaml));
        assertEquals(Arrays.asList(dataActions), j.deserializeDataActionList(arrAsJson));
        assertEquals(Arrays.asList(dataActions), lFromYaml);
        assertEquals(Arrays.asList(dataActions), lFromJson);
    }
}
