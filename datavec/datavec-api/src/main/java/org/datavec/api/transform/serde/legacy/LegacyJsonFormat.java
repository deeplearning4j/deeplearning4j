package org.datavec.api.transform.serde.legacy;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
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
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * This class defines a set of Jackson Mixins - which are a way of using a proxy class with annotations to override
 * the existing annotations.
 * In 1.0.0-beta, we switched how subtypes were handled in JSON ser/de: from "wrapper object" to "@class field".
 * We use these mixins to allow us to still load the old format
 *
 * @author Alex Black
 */
public class LegacyJsonFormat {

    private LegacyJsonFormat(){ }

    /**
     * Get a mapper (minus general config) suitable for loading old format JSON - 1.0.0-alpha and before
     * @return Object mapper
     */
    public static ObjectMapper legacyMapper(){
        ObjectMapper om = new ObjectMapper();
        om.addMixIn(Schema.class, SchemaMixin.class);
        om.addMixIn(ColumnMetaData.class, ColumnMetaDataMixin.class);
        om.addMixIn(Transform.class, TransformMixin.class);
        om.addMixIn(Condition.class, ConditionMixin.class);
        om.addMixIn(Writable.class, WritableMixin.class);
        om.addMixIn(Filter.class, FilterMixin.class);
        om.addMixIn(SequenceComparator.class, SequenceComparatorMixin.class);
        om.addMixIn(SequenceSplit.class, SequenceSplitMixin.class);
        om.addMixIn(WindowFunction.class, WindowFunctionMixin.class);
        om.addMixIn(CalculateSortedRank.class, CalculateSortedRankMixin.class);
        om.addMixIn(WritableComparator.class, WritableComparatorMixin.class);
        om.addMixIn(ColumnAnalysis.class, ColumnAnalysisMixin.class);
        om.addMixIn(IStringReducer.class, IStringReducerMixin.class);
        return om;
    }


    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes({@JsonSubTypes.Type(value = Schema.class, name = "Schema"),
            @JsonSubTypes.Type(value = SequenceSchema.class, name = "SequenceSchema")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class SchemaMixin { }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes({@JsonSubTypes.Type(value = BinaryMetaData.class, name = "Binary"),
            @JsonSubTypes.Type(value = BooleanMetaData.class, name = "Boloean"),
            @JsonSubTypes.Type(value = CategoricalMetaData.class, name = "Categorical"),
            @JsonSubTypes.Type(value = DoubleMetaData.class, name = "Double"),
            @JsonSubTypes.Type(value = FloatMetaData.class, name = "Float"),
            @JsonSubTypes.Type(value = IntegerMetaData.class, name = "Integer"),
            @JsonSubTypes.Type(value = LongMetaData.class, name = "Long"),
            @JsonSubTypes.Type(value = NDArrayMetaData.class, name = "NDArray"),
            @JsonSubTypes.Type(value = StringMetaData.class, name = "String"),
            @JsonSubTypes.Type(value = TimeMetaData.class, name = "Time")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class ColumnMetaDataMixin { }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = CalculateSortedRank.class, name = "CalculateSortedRank"),
            @JsonSubTypes.Type(value = CategoricalToIntegerTransform.class, name = "CategoricalToIntegerTransform"),
            @JsonSubTypes.Type(value = CategoricalToOneHotTransform.class, name = "CategoricalToOneHotTransform"),
            @JsonSubTypes.Type(value = IntegerToCategoricalTransform.class, name = "IntegerToCategoricalTransform"),
            @JsonSubTypes.Type(value = StringToCategoricalTransform.class, name = "StringToCategoricalTransform"),
            @JsonSubTypes.Type(value = DuplicateColumnsTransform.class, name = "DuplicateColumnsTransform"),
            @JsonSubTypes.Type(value = RemoveColumnsTransform.class, name = "RemoveColumnsTransform"),
            @JsonSubTypes.Type(value = RenameColumnsTransform.class, name = "RenameColumnsTransform"),
            @JsonSubTypes.Type(value = ReorderColumnsTransform.class, name = "ReorderColumnsTransform"),
            @JsonSubTypes.Type(value = ConditionalCopyValueTransform.class, name = "ConditionalCopyValueTransform"),
            @JsonSubTypes.Type(value = ConditionalReplaceValueTransform.class, name = "ConditionalReplaceValueTransform"),
            @JsonSubTypes.Type(value = ConditionalReplaceValueTransformWithDefault.class, name = "ConditionalReplaceValueTransformWithDefault"),
            @JsonSubTypes.Type(value = DoubleColumnsMathOpTransform.class, name = "DoubleColumnsMathOpTransform"),
            @JsonSubTypes.Type(value = DoubleMathOpTransform.class, name = "DoubleMathOpTransform"),
            @JsonSubTypes.Type(value = Log2Normalizer.class, name = "Log2Normalizer"),
            @JsonSubTypes.Type(value = MinMaxNormalizer.class, name = "MinMaxNormalizer"),
            @JsonSubTypes.Type(value = StandardizeNormalizer.class, name = "StandardizeNormalizer"),
            @JsonSubTypes.Type(value = SubtractMeanNormalizer.class, name = "SubtractMeanNormalizer"),
            @JsonSubTypes.Type(value = IntegerColumnsMathOpTransform.class, name = "IntegerColumnsMathOpTransform"),
            @JsonSubTypes.Type(value = IntegerMathOpTransform.class, name = "IntegerMathOpTransform"),
            @JsonSubTypes.Type(value = ReplaceEmptyIntegerWithValueTransform.class, name = "ReplaceEmptyIntegerWithValueTransform"),
            @JsonSubTypes.Type(value = ReplaceInvalidWithIntegerTransform.class, name = "ReplaceInvalidWithIntegerTransform"),
            @JsonSubTypes.Type(value = LongColumnsMathOpTransform.class, name = "LongColumnsMathOpTransform"),
            @JsonSubTypes.Type(value = LongMathOpTransform.class, name = "LongMathOpTransform"),
            @JsonSubTypes.Type(value = MapAllStringsExceptListTransform.class, name = "MapAllStringsExceptListTransform"),
            @JsonSubTypes.Type(value = RemoveWhiteSpaceTransform.class, name = "RemoveWhiteSpaceTransform"),
            @JsonSubTypes.Type(value = ReplaceEmptyStringTransform.class, name = "ReplaceEmptyStringTransform"),
            @JsonSubTypes.Type(value = ReplaceStringTransform.class, name = "ReplaceStringTransform"),
            @JsonSubTypes.Type(value = StringListToCategoricalSetTransform.class, name = "StringListToCategoricalSetTransform"),
            @JsonSubTypes.Type(value = StringMapTransform.class, name = "StringMapTransform"),
            @JsonSubTypes.Type(value = DeriveColumnsFromTimeTransform.class, name = "DeriveColumnsFromTimeTransform"),
            @JsonSubTypes.Type(value = StringToTimeTransform.class, name = "StringToTimeTransform"),
            @JsonSubTypes.Type(value = TimeMathOpTransform.class, name = "TimeMathOpTransform"),
            @JsonSubTypes.Type(value = ReduceSequenceByWindowTransform.class, name = "ReduceSequenceByWindowTransform"),
            @JsonSubTypes.Type(value = DoubleMathFunctionTransform.class, name = "DoubleMathFunctionTransform"),
            @JsonSubTypes.Type(value = AddConstantColumnTransform.class, name = "AddConstantColumnTransform"),
            @JsonSubTypes.Type(value = RemoveAllColumnsExceptForTransform.class, name = "RemoveAllColumnsExceptForTransform"),
            @JsonSubTypes.Type(value = ParseDoubleTransform.class, name = "ParseDoubleTransform"),
            @JsonSubTypes.Type(value = ConvertToString.class, name = "ConvertToStringTransform"),
            @JsonSubTypes.Type(value = AppendStringColumnTransform.class, name = "AppendStringColumnTransform"),
            @JsonSubTypes.Type(value = SequenceDifferenceTransform.class, name = "SequenceDifferenceTransform"),
            @JsonSubTypes.Type(value = ReduceSequenceTransform.class, name = "ReduceSequenceTransform"),
            @JsonSubTypes.Type(value = SequenceMovingWindowReduceTransform.class, name = "SequenceMovingWindowReduceTransform"),
            @JsonSubTypes.Type(value = IntegerToOneHotTransform.class, name = "IntegerToOneHotTransform"),
            @JsonSubTypes.Type(value = SequenceTrimTransform.class, name = "SequenceTrimTransform"),
            @JsonSubTypes.Type(value = SequenceOffsetTransform.class, name = "SequenceOffsetTransform"),
            @JsonSubTypes.Type(value = NDArrayColumnsMathOpTransform.class, name = "NDArrayColumnsMathOpTransform"),
            @JsonSubTypes.Type(value = NDArrayDistanceTransform.class, name = "NDArrayDistanceTransform"),
            @JsonSubTypes.Type(value = NDArrayMathFunctionTransform.class, name = "NDArrayMathFunctionTransform"),
            @JsonSubTypes.Type(value = NDArrayScalarOpTransform.class, name = "NDArrayScalarOpTransform"),
            @JsonSubTypes.Type(value = ChangeCaseStringTransform.class, name = "ChangeCaseStringTransform"),
            @JsonSubTypes.Type(value = ConcatenateStringColumns.class, name = "ConcatenateStringColumns"),
            @JsonSubTypes.Type(value = StringListToCountsNDArrayTransform.class, name = "StringListToCountsNDArrayTransform"),
            @JsonSubTypes.Type(value = StringListToIndicesNDArrayTransform.class, name = "StringListToIndicesNDArrayTransform"),
            @JsonSubTypes.Type(value = PivotTransform.class, name = "PivotTransform"),
            @JsonSubTypes.Type(value = TextToCharacterIndexTransform.class, name = "TextToCharacterIndexTransform")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class TransformMixin { }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = TrivialColumnCondition.class, name = "TrivialColumnCondition"),
            @JsonSubTypes.Type(value = CategoricalColumnCondition.class, name = "CategoricalColumnCondition"),
            @JsonSubTypes.Type(value = DoubleColumnCondition.class, name = "DoubleColumnCondition"),
            @JsonSubTypes.Type(value = IntegerColumnCondition.class, name = "IntegerColumnCondition"),
            @JsonSubTypes.Type(value = LongColumnCondition.class, name = "LongColumnCondition"),
            @JsonSubTypes.Type(value = NullWritableColumnCondition.class, name = "NullWritableColumnCondition"),
            @JsonSubTypes.Type(value = StringColumnCondition.class, name = "StringColumnCondition"),
            @JsonSubTypes.Type(value = TimeColumnCondition.class, name = "TimeColumnCondition"),
            @JsonSubTypes.Type(value = StringRegexColumnCondition.class, name = "StringRegexColumnCondition"),
            @JsonSubTypes.Type(value = BooleanCondition.class, name = "BooleanCondition"),
            @JsonSubTypes.Type(value = NaNColumnCondition.class, name = "NaNColumnCondition"),
            @JsonSubTypes.Type(value = InfiniteColumnCondition.class, name = "InfiniteColumnCondition"),
            @JsonSubTypes.Type(value = SequenceLengthCondition.class, name = "SequenceLengthCondition")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class ConditionMixin { }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = ArrayWritable.class, name = "ArrayWritable"),
            @JsonSubTypes.Type(value = BooleanWritable.class, name = "BooleanWritable"),
            @JsonSubTypes.Type(value = ByteWritable.class, name = "ByteWritable"),
            @JsonSubTypes.Type(value = DoubleWritable.class, name = "DoubleWritable"),
            @JsonSubTypes.Type(value = FloatWritable.class, name = "FloatWritable"),
            @JsonSubTypes.Type(value = IntWritable.class, name = "IntWritable"),
            @JsonSubTypes.Type(value = LongWritable.class, name = "LongWritable"),
            @JsonSubTypes.Type(value = NullWritable.class, name = "NullWritable"),
            @JsonSubTypes.Type(value = Text.class, name = "Text"),
            @JsonSubTypes.Type(value = BytesWritable.class, name = "BytesWritable")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class WritableMixin { }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = ConditionFilter.class, name = "ConditionFilter"),
            @JsonSubTypes.Type(value = FilterInvalidValues.class, name = "FilterInvalidValues"),
            @JsonSubTypes.Type(value = InvalidNumColumns.class, name = "InvalidNumCols")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class FilterMixin { }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = NumericalColumnComparator.class, name = "NumericalColumnComparator"),
            @JsonSubTypes.Type(value = StringComparator.class, name = "StringComparator")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class SequenceComparatorMixin { }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = SequenceSplitTimeSeparation.class, name = "SequenceSplitTimeSeparation"),
            @JsonSubTypes.Type(value = SplitMaxLengthSequence.class, name = "SplitMaxLengthSequence")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class SequenceSplitMixin { }

    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = TimeWindowFunction.class, name = "TimeWindowFunction"),
            @JsonSubTypes.Type(value = OverlappingTimeWindowFunction.class, name = "OverlappingTimeWindowFunction")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class WindowFunctionMixin { }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = CalculateSortedRank.class, name = "CalculateSortedRank")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class CalculateSortedRankMixin { }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = DoubleWritableComparator.class, name = "DoubleWritableComparator"),
            @JsonSubTypes.Type(value = FloatWritableComparator.class, name = "FloatWritableComparator"),
            @JsonSubTypes.Type(value = IntWritableComparator.class, name = "IntWritableComparator"),
            @JsonSubTypes.Type(value = LongWritableComparator.class, name = "LongWritableComparator"),
            @JsonSubTypes.Type(value = TextWritableComparator.class, name = "TextWritableComparator")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class WritableComparatorMixin { }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = BytesAnalysis.class, name = "BytesAnalysis"),
            @JsonSubTypes.Type(value = CategoricalAnalysis.class, name = "CategoricalAnalysis"),
            @JsonSubTypes.Type(value = DoubleAnalysis.class, name = "DoubleAnalysis"),
            @JsonSubTypes.Type(value = IntegerAnalysis.class, name = "IntegerAnalysis"),
            @JsonSubTypes.Type(value = LongAnalysis.class, name = "LongAnalysis"),
            @JsonSubTypes.Type(value = StringAnalysis.class, name = "StringAnalysis"),
            @JsonSubTypes.Type(value = TimeAnalysis.class, name = "TimeAnalysis")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class ColumnAnalysisMixin{ }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
    @JsonSubTypes(value = {@JsonSubTypes.Type(value = StringReducer.class, name = "StringReducer")})
    @NoArgsConstructor(access = AccessLevel.PRIVATE)
    public static class IStringReducerMixin{ }
}
