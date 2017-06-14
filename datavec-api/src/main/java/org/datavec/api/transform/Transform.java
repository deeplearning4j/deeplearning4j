/*-
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

package org.datavec.api.transform;

import org.datavec.api.transform.sequence.ReduceSequenceTransform;
import org.datavec.api.transform.sequence.trim.SequenceTrimTransform;
import org.datavec.api.transform.transform.column.*;
import org.datavec.api.transform.transform.integer.*;
import org.datavec.api.transform.transform.parse.ParseDoubleTransform;
import org.datavec.api.transform.transform.sequence.SequenceDifferenceTransform;
import org.datavec.api.transform.transform.sequence.SequenceMovingWindowReduceTransform;
import org.datavec.api.transform.transform.sequence.SequenceOffsetTransform;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.window.ReduceSequenceByWindowTransform;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.datavec.api.transform.transform.categorical.IntegerToCategoricalTransform;
import org.datavec.api.transform.transform.categorical.StringToCategoricalTransform;
import org.datavec.api.transform.transform.condition.ConditionalCopyValueTransform;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransform;
import org.datavec.api.transform.transform.doubletransform.*;
import org.datavec.api.transform.transform.longtransform.LongColumnsMathOpTransform;
import org.datavec.api.transform.transform.longtransform.LongMathOpTransform;
import org.datavec.api.transform.transform.string.*;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.transform.transform.time.TimeMathOpTransform;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

/**A Transform converts an example to another example, or a sequence to another sequence
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value = {
                @JsonSubTypes.Type(value = CategoricalToIntegerTransform.class, name = "CategoricalToIntegerTransform"),
                @JsonSubTypes.Type(value = CategoricalToOneHotTransform.class, name = "CategoricalToOneHotTransform"),
                @JsonSubTypes.Type(value = IntegerToCategoricalTransform.class, name = "IntegerToCategoricalTransform"),
                @JsonSubTypes.Type(value = StringToCategoricalTransform.class, name = "StringToCategoricalTransform"),
                @JsonSubTypes.Type(value = DuplicateColumnsTransform.class, name = "DuplicateColumnsTransform"),
                @JsonSubTypes.Type(value = RemoveColumnsTransform.class, name = "RemoveColumnsTransform"),
                @JsonSubTypes.Type(value = RenameColumnsTransform.class, name = "RenameColumnsTransform"),
                @JsonSubTypes.Type(value = ReorderColumnsTransform.class, name = "ReorderColumnsTransform"),
                @JsonSubTypes.Type(value = ConditionalCopyValueTransform.class, name = "ConditionalCopyValueTransform"),
                @JsonSubTypes.Type(value = ConditionalReplaceValueTransform.class,
                                name = "ConditionalReplaceValueTransform"),
                @JsonSubTypes.Type(value = DoubleColumnsMathOpTransform.class, name = "DoubleColumnsMathOpTransform"),
                @JsonSubTypes.Type(value = DoubleMathOpTransform.class, name = "DoubleMathOpTransform"),
                @JsonSubTypes.Type(value = Log2Normalizer.class, name = "Log2Normalizer"),
                @JsonSubTypes.Type(value = MinMaxNormalizer.class, name = "MinMaxNormalizer"),
                @JsonSubTypes.Type(value = StandardizeNormalizer.class, name = "StandardizeNormalizer"),
                @JsonSubTypes.Type(value = SubtractMeanNormalizer.class, name = "SubtractMeanNormalizer"),
                @JsonSubTypes.Type(value = IntegerColumnsMathOpTransform.class, name = "IntegerColumnsMathOpTransform"),
                @JsonSubTypes.Type(value = IntegerMathOpTransform.class, name = "IntegerMathOpTransform"),
                @JsonSubTypes.Type(value = ReplaceEmptyIntegerWithValueTransform.class,
                                name = "ReplaceEmptyIntegerWithValueTransform"),
                @JsonSubTypes.Type(value = ReplaceInvalidWithIntegerTransform.class,
                                name = "ReplaceInvalidWithIntegerTransform"),
                @JsonSubTypes.Type(value = LongColumnsMathOpTransform.class, name = "LongColumnsMathOpTransform"),
                @JsonSubTypes.Type(value = LongMathOpTransform.class, name = "LongMathOpTransform"),
                @JsonSubTypes.Type(value = MapAllStringsExceptListTransform.class,
                                name = "MapAllStringsExceptListTransform"),
                @JsonSubTypes.Type(value = RemoveWhiteSpaceTransform.class, name = "RemoveWhiteSpaceTransform"),
                @JsonSubTypes.Type(value = ReplaceEmptyStringTransform.class, name = "ReplaceEmptyStringTransform"),
                @JsonSubTypes.Type(value = ReplaceStringTransform.class, name = "ReplaceStringTransform"),
                @JsonSubTypes.Type(value = StringListToCategoricalSetTransform.class,
                                name = "StringListToCategoricalSetTransform"),
                @JsonSubTypes.Type(value = StringMapTransform.class, name = "StringMapTransform"),
                @JsonSubTypes.Type(value = DeriveColumnsFromTimeTransform.class,
                                name = "DeriveColumnsFromTimeTransform"),
                @JsonSubTypes.Type(value = StringToTimeTransform.class, name = "StringToTimeTransform"),
                @JsonSubTypes.Type(value = TimeMathOpTransform.class, name = "TimeMathOpTransform"),
                @JsonSubTypes.Type(value = ReduceSequenceByWindowTransform.class,
                                name = "ReduceSequenceByWindowTransform"),
                @JsonSubTypes.Type(value = DoubleMathFunctionTransform.class, name = "DoubleMathFunctionTransform"),
                @JsonSubTypes.Type(value = AddConstantColumnTransform.class, name = "AddConstantColumnTransform"),
                @JsonSubTypes.Type(value = RemoveAllColumnsExceptForTransform.class,
                                name = "RemoveAllColumnsExceptForTransform"),
                @JsonSubTypes.Type(value = ParseDoubleTransform.class, name = "ParseDoubleTransform"),
                @JsonSubTypes.Type(value = ConvertToString.class, name = "ConvertToStringTransform"),
                @JsonSubTypes.Type(value = AppendStringColumnTransform.class, name = "AppendStringColumnTransform"),
                @JsonSubTypes.Type(value = SequenceDifferenceTransform.class, name = "SequenceDifferenceTransform"),
                @JsonSubTypes.Type(value = ReduceSequenceTransform.class, name = "ReduceSequenceTransform"),
                @JsonSubTypes.Type(value = SequenceMovingWindowReduceTransform.class, name = "SequenceMovingWindowReduceTransform"),
                @JsonSubTypes.Type(value = IntegerToOneHotTransform.class, name = "IntegerToOneHotTransform"),
                @JsonSubTypes.Type(value = SequenceTrimTransform.class, name = "SequenceTrimTransform"),
                @JsonSubTypes.Type(value = SequenceOffsetTransform.class, name = "SequenceOffsetTransform")
})
public interface Transform extends Serializable, ColumnOp {

    /**
     * Transform a writable
     * in to another writable
     * @param writables the record to transform
     * @return the transformed writable
     */
    List<Writable> map(List<Writable> writables);

    /** Transform a sequence */
    List<List<Writable>> mapSequence(List<List<Writable>> sequence);

    /**
     * Transform an object
     * in to another object
     * @param input the record to transform
     * @return the transformed writable
     */
    Object map(Object input);

    /** Transform a sequence */
    Object mapSequence(Object sequence);


}
