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

package org.datavec.api.transform.serde.legacy;

import org.datavec.api.transform.Transform;
import org.datavec.api.transform.analysis.columns.*;
import org.datavec.api.transform.condition.BooleanCondition;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.column.*;
import org.datavec.api.transform.condition.sequence.SequenceLengthCondition;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.filter.InvalidNumColumns;
import org.datavec.api.transform.metadata.*;
import org.datavec.api.transform.ndarray.NDArrayColumnsMathOpTransform;
import org.datavec.api.transform.ndarray.NDArrayDistanceTransform;
import org.datavec.api.transform.ndarray.NDArrayMathFunctionTransform;
import org.datavec.api.transform.ndarray.NDArrayScalarOpTransform;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.sequence.ReduceSequenceTransform;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator;
import org.datavec.api.transform.sequence.comparator.StringComparator;
import org.datavec.api.transform.sequence.split.SequenceSplitTimeSeparation;
import org.datavec.api.transform.sequence.split.SplitMaxLengthSequence;
import org.datavec.api.transform.sequence.trim.SequenceTrimTransform;
import org.datavec.api.transform.sequence.window.OverlappingTimeWindowFunction;
import org.datavec.api.transform.sequence.window.ReduceSequenceByWindowTransform;
import org.datavec.api.transform.sequence.window.TimeWindowFunction;
import org.datavec.api.transform.sequence.window.WindowFunction;
import org.datavec.api.transform.stringreduce.IStringReducer;
import org.datavec.api.transform.stringreduce.StringReducer;
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
import org.datavec.api.transform.transform.nlp.TextToTermIndexSequenceTransform;
import org.datavec.api.transform.transform.parse.ParseDoubleTransform;
import org.datavec.api.transform.transform.sequence.SequenceDifferenceTransform;
import org.datavec.api.transform.transform.sequence.SequenceMovingWindowReduceTransform;
import org.datavec.api.transform.transform.sequence.SequenceOffsetTransform;
import org.datavec.api.transform.transform.string.*;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.transform.transform.time.TimeMathOpTransform;
import org.datavec.api.writable.*;
import org.datavec.api.writable.comparator.*;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

import java.util.HashMap;
import java.util.Map;

public class LegacyMappingHelper {

    public static Map<String,String> legacyMappingForClass(Class c){
        //Need to be able to get the map - and they need to be mutable...
        switch (c.getSimpleName()){
            case "Transform":
                return getLegacyMappingImageTransform();
            case "ColumnAnalysis":
                return getLegacyMappingColumnAnalysis();
            case "Condition":
                return getLegacyMappingCondition();
            case "Filter":
                return getLegacyMappingFilter();
            case "ColumnMetaData":
                return mapColumnMetaData;
            case "CalculateSortedRank":
                return mapCalculateSortedRank;
            case "Schema":
                return mapSchema;
            case "SequenceComparator":
                return mapSequenceComparator;
            case "SequenceSplit":
                return mapSequenceSplit;
            case "WindowFunction":
                return mapWindowFunction;
            case "IStringReducer":
                return mapIStringReducer;
            case "Writable":
                return mapWritable;
            case "WritableComparator":
                return mapWritableComparator;
            case "ImageTransform":
                return mapImageTransform;
            default:
                //Should never happen
                throw new IllegalArgumentException("No legacy mapping available for class " + c.getName());
        }
    }

    private static Map<String,String> mapTransform;
    private static Map<String,String> mapColumnAnalysis;
    private static Map<String,String> mapCondition;
    private static Map<String,String> mapFilter;
    private static Map<String,String> mapColumnMetaData;
    private static Map<String,String> mapCalculateSortedRank;
    private static Map<String,String> mapSchema;
    private static Map<String,String> mapSequenceComparator;
    private static Map<String,String> mapSequenceSplit;
    private static Map<String,String> mapWindowFunction;
    private static Map<String,String> mapIStringReducer;
    private static Map<String,String> mapWritable;
    private static Map<String,String> mapWritableComparator;
    private static Map<String,String> mapImageTransform;
    
    private static synchronized Map<String,String> getLegacyMappingTransform(){

        if(mapTransform == null) {
            //The following classes all used their class short name
            Map<String, String> m = new HashMap<>();
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

            //The following never had subtype annotations, and hence will have had the default name:
            m.put(TextToTermIndexSequenceTransform.class.getSimpleName(), TextToTermIndexSequenceTransform.class.getName());
            m.put(ConvertToInteger.class.getSimpleName(), ConvertToInteger.class.getName());
            m.put(ConvertToDouble.class.getSimpleName(), ConvertToDouble.class.getName());

            mapTransform = m;
        }

        return mapTransform;
    }

    private static Map<String,String> getLegacyMappingColumnAnalysis(){
        if(mapColumnAnalysis == null) {
            Map<String, String> m = new HashMap<>();
            m.put("BytesAnalysis", BytesAnalysis.class.getName());
            m.put("CategoricalAnalysis", CategoricalAnalysis.class.getName());
            m.put("DoubleAnalysis", DoubleAnalysis.class.getName());
            m.put("IntegerAnalysis", IntegerAnalysis.class.getName());
            m.put("LongAnalysis", LongAnalysis.class.getName());
            m.put("StringAnalysis", StringAnalysis.class.getName());
            m.put("TimeAnalysis", TimeAnalysis.class.getName());

            //The following never had subtype annotations, and hence will have had the default name:
            m.put(NDArrayAnalysis.class.getSimpleName(), NDArrayAnalysis.class.getName());

            mapColumnAnalysis = m;
        }

        return mapColumnAnalysis;
    }

    private static Map<String,String> getLegacyMappingCondition(){
        if(mapCondition == null) {
            Map<String, String> m = new HashMap<>();
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

            //The following never had subtype annotations, and hence will have had the default name:
            m.put(InvalidValueColumnCondition.class.getSimpleName(), InvalidValueColumnCondition.class.getName());
            m.put(BooleanColumnCondition.class.getSimpleName(), BooleanColumnCondition.class.getName());

            mapCondition = m;
        }

        return mapCondition;
    }

    private static Map<String,String> getLegacyMappingFilter(){
        if(mapFilter == null) {
            Map<String, String> m = new HashMap<>();
            m.put("ConditionFilter", ConditionFilter.class.getName());
            m.put("FilterInvalidValues", FilterInvalidValues.class.getName());
            m.put("InvalidNumCols", InvalidNumColumns.class.getName());

            mapFilter = m;
        }
        return mapFilter;
    }

    private static Map<String,String> getLegacyMappingColumnMetaData(){
        if(mapColumnMetaData == null) {
            Map<String, String> m = new HashMap<>();
            m.put("Categorical", CategoricalMetaData.class.getName());
            m.put("Double", DoubleMetaData.class.getName());
            m.put("Float", FloatMetaData.class.getName());
            m.put("Integer", IntegerMetaData.class.getName());
            m.put("Long", LongMetaData.class.getName());
            m.put("String", StringMetaData.class.getName());
            m.put("Time", TimeMetaData.class.getName());
            m.put("NDArray", NDArrayMetaData.class.getName());

            //The following never had subtype annotations, and hence will have had the default name:
            m.put(BooleanMetaData.class.getSimpleName(), BooleanMetaData.class.getName());
            m.put(BinaryMetaData.class.getSimpleName(), BinaryMetaData.class.getName());

            mapColumnMetaData = m;
        }

        return mapColumnMetaData;
    }

    private static Map<String,String> getLegacyMappingCalculateSortedRank(){
        if(mapCalculateSortedRank == null) {
            Map<String, String> m = new HashMap<>();
            m.put("CalculateSortedRank", CalculateSortedRank.class.getName());
            mapCalculateSortedRank = m;
        }
        return mapCalculateSortedRank;
    }

    private static Map<String,String> getLegacyMappingSchema(){
        if(mapSchema == null) {
            Map<String, String> m = new HashMap<>();
            m.put("Schema", Schema.class.getName());
            m.put("SequenceSchema", SequenceSchema.class.getName());

            mapSchema = m;
        }
        return mapSchema;
    }

    private static Map<String,String> getLegacyMappingSequenceComparator(){
        if(mapSequenceComparator == null) {
            Map<String, String> m = new HashMap<>();
            m.put("NumericalColumnComparator", NumericalColumnComparator.class.getName());
            m.put("StringComparator", StringComparator.class.getName());

            mapSequenceComparator = m;
        }
        return mapSequenceComparator;
    }

    private static Map<String,String> getLegacyMappingSequenceSplit(){
        if(mapSequenceSplit == null) {
            Map<String, String> m = new HashMap<>();
            m.put("SequenceSplitTimeSeparation", SequenceSplitTimeSeparation.class.getName());
            m.put("SplitMaxLengthSequence", SplitMaxLengthSequence.class.getName());

            mapSequenceSplit = m;
        }
        return mapSequenceSplit;
    }

    private static Map<String,String> getLegacyMappingWindowFunction(){
        if(mapWindowFunction == null) {
            Map<String, String> m = new HashMap<>();
            m.put("TimeWindowFunction", TimeWindowFunction.class.getName());
            m.put("OverlappingTimeWindowFunction", OverlappingTimeWindowFunction.class.getName());

            mapWindowFunction = m;
        }
        return mapWindowFunction;
    }

    private static Map<String,String> getLegacyMappingIStringReducer(){
        if(mapIStringReducer == null) {
            Map<String, String> m = new HashMap<>();
            m.put("StringReducer", StringReducer.class.getName());

            mapIStringReducer = m;
        }
        return mapIStringReducer;
    }

    private static Map<String,String> getLegacyMappingWritable(){
        if (mapWritable == null) {
            Map<String, String> m = new HashMap<>();
            m.put("ArrayWritable", ArrayWritable.class.getName());
            m.put("BooleanWritable", BooleanWritable.class.getName());
            m.put("ByteWritable", ByteWritable.class.getName());
            m.put("DoubleWritable", DoubleWritable.class.getName());
            m.put("FloatWritable", FloatWritable.class.getName());
            m.put("IntWritable", IntWritable.class.getName());
            m.put("LongWritable", LongWritable.class.getName());
            m.put("NullWritable", NullWritable.class.getName());
            m.put("Text", Text.class.getName());
            m.put("BytesWritable", BytesWritable.class.getName());

            //The following never had subtype annotations, and hence will have had the default name:
            m.put(NDArrayWritable.class.getSimpleName(), NDArrayWritable.class.getName());

            mapWritable = m;
        }

        return mapWritable;
    }

    private static Map<String,String> getLegacyMappingWritableComparator(){
        if(mapWritableComparator == null) {
            Map<String, String> m = new HashMap<>();
            m.put("DoubleWritableComparator", DoubleWritableComparator.class.getName());
            m.put("FloatWritableComparator", FloatWritableComparator.class.getName());
            m.put("IntWritableComparator", IntWritableComparator.class.getName());
            m.put("LongWritableComparator", LongWritableComparator.class.getName());
            m.put("TextWritableComparator", TextWritableComparator.class.getName());

            //The following never had subtype annotations, and hence will have had the default name:
            m.put(ByteWritable.Comparator.class.getSimpleName(), ByteWritable.Comparator.class.getName());
            m.put(FloatWritable.Comparator.class.getSimpleName(), FloatWritable.Comparator.class.getName());
            m.put(IntWritable.Comparator.class.getSimpleName(), IntWritable.Comparator.class.getName());
            m.put(BooleanWritable.Comparator.class.getSimpleName(), BooleanWritable.Comparator.class.getName());
            m.put(LongWritable.Comparator.class.getSimpleName(), LongWritable.Comparator.class.getName());
            m.put(Text.Comparator.class.getSimpleName(), Text.Comparator.class.getName());
            m.put(LongWritable.DecreasingComparator.class.getSimpleName(), LongWritable.DecreasingComparator.class.getName());
            m.put(DoubleWritable.Comparator.class.getSimpleName(), DoubleWritable.Comparator.class.getName());

            mapWritableComparator = m;
        }

        return mapWritableComparator;
    }

    public static Map<String,String> getLegacyMappingImageTransform(){
        if(mapImageTransform == null) {
            Map<String, String> m = new HashMap<>();
            m.put("EqualizeHistTransform", "org.datavec.image.transform.EqualizeHistTransform");
            m.put("RotateImageTransform", "org.datavec.image.transform.RotateImageTransform");
            m.put("ColorConversionTransform", "org.datavec.image.transform.ColorConversionTransform");
            m.put("WarpImageTransform", "org.datavec.image.transform.WarpImageTransform");
            m.put("BoxImageTransform", "org.datavec.image.transform.BoxImageTransform");
            m.put("CropImageTransform", "org.datavec.image.transform.CropImageTransform");
            m.put("FilterImageTransform", "org.datavec.image.transform.FilterImageTransform");
            m.put("FlipImageTransform", "org.datavec.image.transform.FlipImageTransform");
            m.put("LargestBlobCropTransform", "org.datavec.image.transform.LargestBlobCropTransform");
            m.put("ResizeImageTransform", "org.datavec.image.transform.ResizeImageTransform");
            m.put("RandomCropTransform", "org.datavec.image.transform.RandomCropTransform");
            m.put("ScaleImageTransform", "org.datavec.image.transform.ScaleImageTransform");

            mapImageTransform = m;
        }
        return mapImageTransform;
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

    @JsonDeserialize(using = LegacyFilterDeserializer.class)
    public static class FilterHelper { }

    public static class LegacyFilterDeserializer extends GenericLegacyDeserializer<Filter> {
        public LegacyFilterDeserializer() {
            super(Filter.class, getLegacyMappingFilter());
        }
    }

    @JsonDeserialize(using = LegacyColumnMetaDataDeserializer.class)
    public static class ColumnMetaDataHelper { }

    public static class LegacyColumnMetaDataDeserializer extends GenericLegacyDeserializer<ColumnMetaData> {
        public LegacyColumnMetaDataDeserializer() {
            super(ColumnMetaData.class, getLegacyMappingColumnMetaData());
        }
    }

    @JsonDeserialize(using = LegacyCalculateSortedRankDeserializer.class)
    public static class CalculateSortedRankHelper { }

    public static class LegacyCalculateSortedRankDeserializer extends GenericLegacyDeserializer<CalculateSortedRank> {
        public LegacyCalculateSortedRankDeserializer() {
            super(CalculateSortedRank.class, getLegacyMappingCalculateSortedRank());
        }
    }

    @JsonDeserialize(using = LegacySchemaDeserializer.class)
    public static class SchemaHelper { }

    public static class LegacySchemaDeserializer extends GenericLegacyDeserializer<Schema> {
        public LegacySchemaDeserializer() {
            super(Schema.class, getLegacyMappingSchema());
        }
    }

    @JsonDeserialize(using = LegacySequenceComparatorDeserializer.class)
    public static class SequenceComparatorHelper { }

    public static class LegacySequenceComparatorDeserializer extends GenericLegacyDeserializer<SequenceComparator> {
        public LegacySequenceComparatorDeserializer() {
            super(SequenceComparator.class, getLegacyMappingSequenceComparator());
        }
    }

    @JsonDeserialize(using = LegacySequenceSplitDeserializer.class)
    public static class SequenceSplitHelper { }

    public static class LegacySequenceSplitDeserializer extends GenericLegacyDeserializer<SequenceSplit> {
        public LegacySequenceSplitDeserializer() {
            super(SequenceSplit.class, getLegacyMappingSequenceSplit());
        }
    }

    @JsonDeserialize(using = LegacyWindowFunctionDeserializer.class)
    public static class WindowFunctionHelper { }

    public static class LegacyWindowFunctionDeserializer extends GenericLegacyDeserializer<WindowFunction> {
        public LegacyWindowFunctionDeserializer() {
            super(WindowFunction.class, getLegacyMappingWindowFunction());
        }
    }


    @JsonDeserialize(using = LegacyIStringReducerDeserializer.class)
    public static class IStringReducerHelper { }

    public static class LegacyIStringReducerDeserializer extends GenericLegacyDeserializer<IStringReducer> {
        public LegacyIStringReducerDeserializer() {
            super(IStringReducer.class, getLegacyMappingIStringReducer());
        }
    }


    @JsonDeserialize(using = LegacyWritableDeserializer.class)
    public static class WritableHelper { }

    public static class LegacyWritableDeserializer extends GenericLegacyDeserializer<Writable> {
        public LegacyWritableDeserializer() {
            super(Writable.class, getLegacyMappingWritable());
        }
    }

    @JsonDeserialize(using = LegacyWritableComparatorDeserializer.class)
    public static class WritableComparatorHelper { }

    public static class LegacyWritableComparatorDeserializer extends GenericLegacyDeserializer<WritableComparator> {
        public LegacyWritableComparatorDeserializer() {
            super(WritableComparator.class, getLegacyMappingWritableComparator());
        }
    }
}
