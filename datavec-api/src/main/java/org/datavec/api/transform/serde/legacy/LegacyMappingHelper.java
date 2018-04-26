/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.transform.serde.legacy;

import org.datavec.api.transform.Transform;
import org.datavec.api.transform.analysis.columns.*;
import org.datavec.api.transform.condition.BooleanCondition;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.column.*;
import org.datavec.api.transform.condition.sequence.SequenceLengthCondition;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.ndarray.NDArrayColumnsMathOpTransform;
import org.datavec.api.transform.ndarray.NDArrayDistanceTransform;
import org.datavec.api.transform.ndarray.NDArrayMathFunctionTransform;
import org.datavec.api.transform.ndarray.NDArrayScalarOpTransform;
import org.datavec.api.transform.sequence.ReduceSequenceTransform;
import org.datavec.api.transform.sequence.trim.SequenceTrimTransform;
import org.datavec.api.transform.sequence.window.ReduceSequenceByWindowTransform;
import org.datavec.api.transform.transform.categorical.*;
import org.datavec.api.transform.transform.column.*;
import org.datavec.api.transform.transform.condition.ConditionalCopyValueTransform;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransform;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransformWithDefault;
import org.datavec.api.transform.transform.doubletransform.*;
import org.datavec.api.transform.transform.integer.*;
import org.datavec.api.transform.transform.longtransform.LongColumnsMathOpTransform;
import org.datavec.api.transform.transform.longtransform.LongMathOpTransform;
import org.datavec.api.transform.transform.nlp.TextToCharacterIndexTransform;
import org.datavec.api.transform.transform.parse.ParseDoubleTransform;
import org.datavec.api.transform.transform.sequence.SequenceDifferenceTransform;
import org.datavec.api.transform.transform.sequence.SequenceMovingWindowReduceTransform;
import org.datavec.api.transform.transform.sequence.SequenceOffsetTransform;
import org.datavec.api.transform.transform.string.*;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.transform.transform.time.TimeMathOpTransform;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

import java.util.HashMap;
import java.util.Map;

public class LegacyMappingHelper {
    
    private static Map<String,String> getLegacyMappingTransform(){
        
        //The following classes all used their class short name
        Map<String,String> m = new HashMap<>();
        m.put("CategoricalToIntegerTransform", CategoricalToIntegerTransform.class.getName());
        m.put("CategoricalToOneHotTransform", CategoricalToOneHotTransform.class.getName());
        m.put("IntegerToCategoricalTransform", IntegerToCategoricalTransform.class.getName());
        m.put("StringToCategoricalTransform", StringToCategoricalTransform.class.getName());
        m.put("DuplicateColumnsTransform", DuplicateColumnsTransform.class.getName());
        m.put("RemoveColumnsTransform", RemoveColumnsTransform.class.getName());
        m.put("RenameColumnsTransform", RenameColumnsTransform.class.getName());
        m.put("ReorderColumnsTransform", ReorderColumnsTransform.class.getName());
        m.put("ConditionalCopyValueTransform", ConditionalCopyValueTransform.class.getName());
        m.put("ConditionalReplaceValueTransform", ConditionalReplaceValueTransform.class.getName());
        m.put("ConditionalReplaceValueTransformWithDefault", ConditionalReplaceValueTransformWithDefault.class.getName());
        m.put("DoubleColumnsMathOpTransform", DoubleColumnsMathOpTransform.class.getName());
        m.put("DoubleMathOpTransform", DoubleMathOpTransform.class.getName());
        m.put("Log2Normalizer", Log2Normalizer.class.getName());
        m.put("MinMaxNormalizer", MinMaxNormalizer.class.getName());
        m.put("StandardizeNormalizer", StandardizeNormalizer.class.getName());
        m.put("SubtractMeanNormalizer", SubtractMeanNormalizer.class.getName());
        m.put("IntegerColumnsMathOpTransform", IntegerColumnsMathOpTransform.class.getName());
        m.put("IntegerMathOpTransform", IntegerMathOpTransform.class.getName());
        m.put("ReplaceEmptyIntegerWithValueTransform", ReplaceEmptyIntegerWithValueTransform.class.getName());
        m.put("ReplaceInvalidWithIntegerTransform", ReplaceInvalidWithIntegerTransform.class.getName());
        m.put("LongColumnsMathOpTransform", LongColumnsMathOpTransform.class.getName());
        m.put("LongMathOpTransform", LongMathOpTransform.class.getName());
        m.put("MapAllStringsExceptListTransform", MapAllStringsExceptListTransform.class.getName());
        m.put("RemoveWhiteSpaceTransform", RemoveWhiteSpaceTransform.class.getName());
        m.put("ReplaceEmptyStringTransform", ReplaceEmptyStringTransform.class.getName());
        m.put("ReplaceStringTransform", ReplaceStringTransform.class.getName());
        m.put("StringListToCategoricalSetTransform", StringListToCategoricalSetTransform.class.getName());
        m.put("StringMapTransform", StringMapTransform.class.getName());
        m.put("DeriveColumnsFromTimeTransform", DeriveColumnsFromTimeTransform.class.getName());
        m.put("StringToTimeTransform", StringToTimeTransform.class.getName());
        m.put("TimeMathOpTransform", TimeMathOpTransform.class.getName());
        m.put("ReduceSequenceByWindowTransform", ReduceSequenceByWindowTransform.class.getName());
        m.put("DoubleMathFunctionTransform", DoubleMathFunctionTransform.class.getName());
        m.put("AddConstantColumnTransform", AddConstantColumnTransform.class.getName());
        m.put("RemoveAllColumnsExceptForTransform", RemoveAllColumnsExceptForTransform.class.getName());
        m.put("ParseDoubleTransform", ParseDoubleTransform.class.getName());
        m.put("ConvertToStringTransform", ConvertToString.class.getName());
        m.put("AppendStringColumnTransform", AppendStringColumnTransform.class.getName());
        m.put("SequenceDifferenceTransform", SequenceDifferenceTransform.class.getName());
        m.put("ReduceSequenceTransform", ReduceSequenceTransform.class.getName());
        m.put("SequenceMovingWindowReduceTransform", SequenceMovingWindowReduceTransform.class.getName());
        m.put("IntegerToOneHotTransform", IntegerToOneHotTransform.class.getName());
        m.put("SequenceTrimTransform", SequenceTrimTransform.class.getName());
        m.put("SequenceOffsetTransform", SequenceOffsetTransform.class.getName());
        m.put("NDArrayColumnsMathOpTransform", NDArrayColumnsMathOpTransform.class.getName());
        m.put("NDArrayDistanceTransform", NDArrayDistanceTransform.class.getName());
        m.put("NDArrayMathFunctionTransform", NDArrayMathFunctionTransform.class.getName());
        m.put("NDArrayScalarOpTransform", NDArrayScalarOpTransform.class.getName());
        m.put("ChangeCaseStringTransform", ChangeCaseStringTransform.class.getName());
        m.put("ConcatenateStringColumns", ConcatenateStringColumns.class.getName());
        m.put("StringListToCountsNDArrayTransform", StringListToCountsNDArrayTransform.class.getName());
        m.put("StringListToIndicesNDArrayTransform", StringListToIndicesNDArrayTransform.class.getName());
        m.put("PivotTransform", PivotTransform.class.getName());
        m.put("TextToCharacterIndexTransform", TextToCharacterIndexTransform.class.getName());

        return m;
    }

    private static Map<String,String> getLegacyMappingColumnAnalysis(){
        Map<String,String> m = new HashMap<>();
        m.put("BytesAnalysis", BytesAnalysis.class.getName());
        m.put("CategoricalAnalysis", CategoricalAnalysis.class.getName());
        m.put("DoubleAnalysis", DoubleAnalysis.class.getName());
        m.put("IntegerAnalysis", IntegerAnalysis.class.getName());
        m.put("LongAnalysis", LongAnalysis.class.getName());
        m.put("StringAnalysis", StringAnalysis.class.getName());
        m.put("TimeAnalysis", TimeAnalysis.class.getName());
        return m;
    }

    private static Map<String,String> getLegacyMappingCondition(){
        Map<String,String> m = new HashMap<>();
        m.put("TrivialColumnCondition", TrivialColumnCondition.class.getName());
        m.put("CategoricalColumnCondition", CategoricalColumnCondition.class.getName());
        m.put("DoubleColumnCondition", DoubleColumnCondition.class.getName());
        m.put("IntegerColumnCondition", IntegerColumnCondition.class.getName());
        m.put("LongColumnCondition", LongColumnCondition.class.getName());
        m.put("NullWritableColumnCondition", NullWritableColumnCondition.class.getName());
        m.put("StringColumnCondition", StringColumnCondition.class.getName());
        m.put("TimeColumnCondition", TimeColumnCondition.class.getName());
        m.put("StringRegexColumnCondition", StringRegexColumnCondition.class.getName());
        m.put("BooleanCondition", BooleanCondition.class.getName());
        m.put("NaNColumnCondition", NaNColumnCondition.class.getName());
        m.put("InfiniteColumnCondition", InfiniteColumnCondition.class.getName());
        m.put("SequenceLengthCondition", SequenceLengthCondition.class.getName());
        return m;
    }

    public static void main(String[] args) {
        String s = "@JsonSubTypes.Type(value = TrivialColumnCondition.class, name = \"TrivialColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = CategoricalColumnCondition.class, name = \"CategoricalColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = DoubleColumnCondition.class, name = \"DoubleColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = IntegerColumnCondition.class, name = \"IntegerColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = LongColumnCondition.class, name = \"LongColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = NullWritableColumnCondition.class, name = \"NullWritableColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = StringColumnCondition.class, name = \"StringColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = TimeColumnCondition.class, name = \"TimeColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = StringRegexColumnCondition.class, name = \"StringRegexColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = BooleanCondition.class, name = \"BooleanCondition\"),\n" +
                "                @JsonSubTypes.Type(value = NaNColumnCondition.class, name = \"NaNColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = InfiniteColumnCondition.class, name = \"InfiniteColumnCondition\"),\n" +
                "                @JsonSubTypes.Type(value = SequenceLengthCondition.class, name = \"SequenceLengthCondition\")";

        String[] str = s.split("\n");
        for(String s2 : str){
            String[] str2 = s2.split(",");
            int first = str2[0].indexOf(" = ");
            int second = str2[0].indexOf(".class");

            String className = str2[0].substring(first+3, second);

            int a = str2[1].indexOf("\"");
            int b = str2[1].indexOf("\"", a+1);
            String oldName = str2[1].substring(a+1,b);

            System.out.println("m.put(\"" + oldName + "\", " + className + ".class.getName());");
        }
    }

    @JsonDeserialize(using = LegacyTransformDeserializer.class)
    public static class TransformHelper { }

    public static class LegacyTransformDeserializer extends GenericLegacyDeserializer<Transform> {
        public LegacyTransformDeserializer() {
            super(Transform.class, getLegacyMappingTransform());
        }
    }

    @JsonDeserialize(using = LegacyColumnAnalysisDeserializer.class)
    public static class ColumnAnalysisHelper { }

    public static class LegacyColumnAnalysisDeserializer extends GenericLegacyDeserializer<ColumnAnalysis> {
        public LegacyColumnAnalysisDeserializer() {
            super(ColumnAnalysis.class, getLegacyMappingColumnAnalysis());
        }
    }

    @JsonDeserialize(using = LegacyConditionDeserializer.class)
    public static class ConditionHelper { }

    public static class LegacyConditionDeserializer extends GenericLegacyDeserializer<Condition> {
        public LegacyConditionDeserializer() {
            super(Condition.class, getLegacyMappingCondition());
        }
    }
}
