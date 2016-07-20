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

package org.datavec.api.transform.serde;

import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.condition.BooleanCondition;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.*;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.datavec.api.transform.transform.categorical.IntegerToCategoricalTransform;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.datavec.api.transform.transform.column.DuplicateColumnsTransform;
import org.datavec.api.transform.transform.column.RemoveColumnsTransform;
import org.datavec.api.transform.transform.column.RenameColumnsTransform;
import org.datavec.api.transform.transform.column.ReorderColumnsTransform;
import org.datavec.api.transform.transform.doubletransform.*;
import org.datavec.api.transform.transform.integer.IntegerColumnsMathOpTransform;
import org.datavec.api.transform.transform.integer.IntegerMathOpTransform;
import org.datavec.api.transform.transform.integer.ReplaceEmptyIntegerWithValueTransform;
import org.datavec.api.transform.transform.integer.ReplaceInvalidWithIntegerTransform;
import org.datavec.api.transform.transform.longtransform.LongColumnsMathOpTransform;
import org.datavec.api.transform.transform.longtransform.LongMathOpTransform;
import org.datavec.api.transform.transform.serde.JsonSerializer;
import org.datavec.api.transform.transform.serde.YamlSerializer;
import org.datavec.api.transform.transform.string.*;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.transform.transform.time.TimeMathOpTransform;
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
    public void testTransforms(){

        Map<String,String> map = new HashMap<>();
        map.put("A","A1");
        map.put("B","B1");

        Transform[] transforms = new Transform[]{
                new CategoricalToIntegerTransform("ColName"),
                new CategoricalToOneHotTransform("ColName"),
                new IntegerToCategoricalTransform("ColName", Arrays.asList("State0","State1")),
                new StringToCategoricalTransform("ColName", Arrays.asList("State0","State1")),
                new DuplicateColumnsTransform(Arrays.asList("Dup1","Dup2"), Arrays.asList("NewName1","NewName2")),
                new RemoveColumnsTransform("R1","R2"),
                new RenameColumnsTransform(Arrays.asList("N1","N2"), Arrays.asList("NewN1","NewN2")),
                new ReorderColumnsTransform("A","B"),
                new DoubleColumnsMathOpTransform("NewName", MathOp.Subtract, "A", "B"),
                new DoubleMathOpTransform("ColName", MathOp.Multiply, 2.0),
                new Log2Normalizer("ColName",1.0,0.0,2.0),
                new MinMaxNormalizer("ColName",0,100,-1,1),
                new StandardizeNormalizer("ColName",20,5),
                new SubtractMeanNormalizer("ColName",10),
                new IntegerColumnsMathOpTransform("NewName", MathOp.Multiply, "A", "B"),
                new IntegerMathOpTransform("ColName", MathOp.Add, 10),
                new ReplaceEmptyIntegerWithValueTransform("Col",3),
                new ReplaceInvalidWithIntegerTransform("Col",3),
                new LongColumnsMathOpTransform("NewName", MathOp.Multiply, "A", "B"),
                new LongMathOpTransform("Col", MathOp.ScalarMax, 5L),
                new MapAllStringsExceptListTransform("Col", "NewVal", Arrays.asList("E1","E2")),
                new RemoveWhiteSpaceTransform("Col"),
                new ReplaceEmptyStringTransform("Col","WasEmpty"),
                new StringListToCategoricalSetTransform("Col",Arrays.asList("A","B"), Arrays.asList("A","B"), ","),
                new StringMapTransform("Col",map),
                new DeriveColumnsFromTimeTransform.Builder("TimeColName").addIntegerDerivedColumn("Hour", DateTimeFieldType.hourOfDay())
                    .addStringDerivedColumn("DateTime","YYYY-MM-dd hh:mm:ss", DateTimeZone.UTC).build(),
                new StringToTimeTransform("TimeCol", "YYYY-MM-dd hh:mm:ss", DateTimeZone.UTC),
                new TimeMathOpTransform("TimeCol", MathOp.Add, 1, TimeUnit.HOURS)
        };

        for(Transform t : transforms){
            String yaml = y.serialize(t);
            String json = j.serialize(t);

            System.out.println(yaml);
            System.out.println(json);
            System.out.println();

            Transform t2 = y.deserializeTransform(yaml);
            Transform t3 = j.deserializeTransform(json);
            assertEquals(t,t2);
            assertEquals(t,t3);
        }


        String tArrAsYaml = y.serialize(transforms);
        String tArrAsJson = j.serialize(transforms);
        String tListAsYaml = y.serializeTransformList(Arrays.asList(transforms));
        String tListAsJson = j.serializeTransformList(Arrays.asList(transforms));

        System.out.println("\n\n\n\n");
        System.out.println(tListAsYaml);

        List<Transform> lFromYaml = y.deserializeTransformList(tListAsYaml);
        List<Transform> lFromJson = j.deserializeTransformList(tListAsJson);

        assertEquals(Arrays.asList(transforms), y.deserializeTransformList(tArrAsYaml));
        assertEquals(Arrays.asList(transforms), j.deserializeTransformList(tArrAsJson));
        assertEquals(Arrays.asList(transforms), lFromYaml);
        assertEquals(Arrays.asList(transforms), lFromJson);
    }


    @Test
    public void testFilters(){
        Filter[] filters = new Filter[]{
                new FilterInvalidValues("A","B"),
                new ConditionFilter(new DoubleColumnCondition("Col", ConditionOp.GreaterOrEqual, 10.0))
        };

        for(Filter f : filters){
            String yaml = y.serialize(f);
            String json = j.serialize(f);

            System.out.println(yaml);
            System.out.println(json);
            System.out.println();

            Filter t2 = y.deserializeFilter(yaml);
            Filter t3 = j.deserializeFilter(json);
            assertEquals(f,t2);
            assertEquals(f,t3);
        }

        String arrAsYaml = y.serialize(filters);
        String arrAsJson = j.serialize(filters);
        String listAsYaml = y.serializeFilterList(Arrays.asList(filters));
        String listAsJson = j.serializeFilterList(Arrays.asList(filters));

        System.out.println("\n\n\n\n");
        System.out.println(listAsYaml);

        List<Filter> lFromYaml = y.deserializeFilterList(listAsYaml);
        List<Filter> lFromJson = j.deserializeFilterList(listAsJson);

        assertEquals(Arrays.asList(filters), y.deserializeFilterList(arrAsYaml));
        assertEquals(Arrays.asList(filters), j.deserializeFilterList(arrAsJson));
        assertEquals(Arrays.asList(filters), lFromYaml);
        assertEquals(Arrays.asList(filters), lFromJson);
    }

    @Test
    public void testConditions(){
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

        Condition[] conditions = new Condition[]{
                new CategoricalColumnCondition("Col",ConditionOp.Equal, "A"),
                new CategoricalColumnCondition("Col",ConditionOp.NotInSet, setStr),
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
                                new CategoricalColumnCondition("Col",ConditionOp.Equal, "A"),
                                new LongColumnCondition("Col2", ConditionOp.Equal, 1)),
                        BooleanCondition.NOT(new TimeColumnCondition("Col3", ConditionOp.Equal, 1L)))
        };

        for(Condition c : conditions){
            String yaml = y.serialize(c);
            String json = j.serialize(c);

            System.out.println(yaml);
            System.out.println(json);
            System.out.println();

            Condition t2 = y.deserializeCondition(yaml);
            Condition t3 = j.deserializeCondition(json);
            assertEquals(c,t2);
            assertEquals(c,t3);
        }

        String arrAsYaml = y.serialize(conditions);
        String arrAsJson = j.serialize(conditions);
        String listAsYaml = y.serializeConditionList(Arrays.asList(conditions));
        String listAsJson = j.serializeConditionList(Arrays.asList(conditions));

        System.out.println("\n\n\n\n");
        System.out.println(listAsYaml);

        List<Condition> lFromYaml = y.deserializeConditionList(listAsYaml);
        List<Condition> lFromJson = j.deserializeConditionList(listAsJson);

        assertEquals(Arrays.asList(conditions), y.deserializeConditionList(arrAsYaml));
        assertEquals(Arrays.asList(conditions), j.deserializeConditionList(arrAsJson));
        assertEquals(Arrays.asList(conditions), lFromYaml);
        assertEquals(Arrays.asList(conditions), lFromJson);
    }

}
