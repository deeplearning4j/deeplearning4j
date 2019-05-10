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

package org.datavec.api.transform.reduce;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.ops.*;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 * Various utilities for performing reductions
 *
 * @author Alex Black
 */
public class AggregableReductionUtils {

    private AggregableReductionUtils() {}


    public static IAggregableReduceOp<Writable, List<Writable>> reduceColumn(List<ReduceOp> op, ColumnType type,
                    boolean ignoreInvalid, ColumnMetaData metaData) {
        switch (type) {
            case Integer:
                return reduceIntColumn(op, ignoreInvalid, metaData);
            case Long:
                return reduceLongColumn(op, ignoreInvalid, metaData);
            case Float:
                return reduceFloatColumn(op, ignoreInvalid, metaData);
            case Double:
                return reduceDoubleColumn(op, ignoreInvalid, metaData);
            case String:
            case Categorical:
                return reduceStringOrCategoricalColumn(op, ignoreInvalid, metaData);
            case Time:
                return reduceTimeColumn(op, ignoreInvalid, metaData);
            case Bytes:
                return reduceBytesColumn(op, ignoreInvalid, metaData);
            default:
                throw new UnsupportedOperationException("Unknown or not implemented column type: " + type);
        }
    }

    public static IAggregableReduceOp<Writable, List<Writable>> reduceIntColumn(List<ReduceOp> lop,
                    boolean ignoreInvalid, ColumnMetaData metaData) {

        List<IAggregableReduceOp<Integer, Writable>> res = new ArrayList<>(lop.size());
        for (int i = 0; i < lop.size(); i++) {
            switch (lop.get(i)) {
                case Prod:
                    res.add(new AggregatorImpls.AggregableProd<Integer>());
                    break;
                case Min:
                    res.add(new AggregatorImpls.AggregableMin<Integer>());
                    break;
                case Max:
                    res.add(new AggregatorImpls.AggregableMax<Integer>());
                    break;
                case Range:
                    res.add(new AggregatorImpls.AggregableRange<Integer>());
                    break;
                case Sum:
                    res.add(new AggregatorImpls.AggregableSum<Integer>());
                    break;
                case Mean:
                    res.add(new AggregatorImpls.AggregableMean<Integer>());
                    break;
                case Stdev:
                    res.add(new AggregatorImpls.AggregableStdDev<Integer>());
                    break;
                case UncorrectedStdDev:
                    res.add(new AggregatorImpls.AggregableUncorrectedStdDev<Integer>());
                    break;
                case Variance:
                    res.add(new AggregatorImpls.AggregableVariance<Integer>());
                    break;
                case PopulationVariance:
                    res.add(new AggregatorImpls.AggregablePopulationVariance<Integer>());
                    break;
                case Count:
                    res.add(new AggregatorImpls.AggregableCount<Integer>());
                    break;
                case CountUnique:
                    res.add(new AggregatorImpls.AggregableCountUnique<Integer>());
                    break;
                case TakeFirst:
                    res.add(new AggregatorImpls.AggregableFirst<Integer>());
                    break;
                case TakeLast:
                    res.add(new AggregatorImpls.AggregableLast<Integer>());
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown or not implemented op: " + lop.get(i));
            }
        }
        IAggregableReduceOp<Writable, List<Writable>> thisOp = new IntWritableOp<>(new AggregableMultiOp<>(res));
        if (ignoreInvalid)
            return new AggregableCheckingOp<>(thisOp, metaData);
        else
            return thisOp;
    }

    public static IAggregableReduceOp<Writable, List<Writable>> reduceLongColumn(List<ReduceOp> lop,
                    boolean ignoreInvalid, ColumnMetaData metaData) {

        List<IAggregableReduceOp<Long, Writable>> res = new ArrayList<>(lop.size());
        for (int i = 0; i < lop.size(); i++) {
            switch (lop.get(i)) {
                case Prod:
                    res.add(new AggregatorImpls.AggregableProd<Long>());
                    break;
                case Min:
                    res.add(new AggregatorImpls.AggregableMin<Long>());
                    break;
                case Max:
                    res.add(new AggregatorImpls.AggregableMax<Long>());
                    break;
                case Range:
                    res.add(new AggregatorImpls.AggregableRange<Long>());
                    break;
                case Sum:
                    res.add(new AggregatorImpls.AggregableSum<Long>());
                    break;
                case Stdev:
                    res.add(new AggregatorImpls.AggregableStdDev<Long>());
                    break;
                case UncorrectedStdDev:
                    res.add(new AggregatorImpls.AggregableUncorrectedStdDev<Long>());
                    break;
                case Variance:
                    res.add(new AggregatorImpls.AggregableVariance<Long>());
                    break;
                case PopulationVariance:
                    res.add(new AggregatorImpls.AggregablePopulationVariance<Long>());
                    break;
                case Mean:
                    res.add(new AggregatorImpls.AggregableMean<Long>());
                    break;
                case Count:
                    res.add(new AggregatorImpls.AggregableCount<Long>());
                    break;
                case CountUnique:
                    res.add(new AggregatorImpls.AggregableCountUnique<Long>());
                    break;
                case TakeFirst:
                    res.add(new AggregatorImpls.AggregableFirst<Long>());
                    break;
                case TakeLast:
                    res.add(new AggregatorImpls.AggregableLast<Long>());
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown or not implemented op: " + lop.get(i));
            }
        }
        IAggregableReduceOp<Writable, List<Writable>> thisOp = new LongWritableOp<>(new AggregableMultiOp<>(res));
        if (ignoreInvalid)
            return new AggregableCheckingOp<>(thisOp, metaData);
        else
            return thisOp;
    }

    public static IAggregableReduceOp<Writable, List<Writable>> reduceFloatColumn(List<ReduceOp> lop,
                    boolean ignoreInvalid, ColumnMetaData metaData) {

        List<IAggregableReduceOp<Float, Writable>> res = new ArrayList<>(lop.size());
        for (int i = 0; i < lop.size(); i++) {
            switch (lop.get(i)) {
                case Prod:
                    res.add(new AggregatorImpls.AggregableProd<Float>());
                    break;
                case Min:
                    res.add(new AggregatorImpls.AggregableMin<Float>());
                    break;
                case Max:
                    res.add(new AggregatorImpls.AggregableMax<Float>());
                    break;
                case Range:
                    res.add(new AggregatorImpls.AggregableRange<Float>());
                    break;
                case Sum:
                    res.add(new AggregatorImpls.AggregableSum<Float>());
                    break;
                case Mean:
                    res.add(new AggregatorImpls.AggregableMean<Float>());
                    break;
                case Stdev:
                    res.add(new AggregatorImpls.AggregableStdDev<Float>());
                    break;
                case UncorrectedStdDev:
                    res.add(new AggregatorImpls.AggregableUncorrectedStdDev<Float>());
                    break;
                case Variance:
                    res.add(new AggregatorImpls.AggregableVariance<Float>());
                    break;
                case PopulationVariance:
                    res.add(new AggregatorImpls.AggregablePopulationVariance<Float>());
                    break;
                case Count:
                    res.add(new AggregatorImpls.AggregableCount<Float>());
                    break;
                case CountUnique:
                    res.add(new AggregatorImpls.AggregableCountUnique<Float>());
                    break;
                case TakeFirst:
                    res.add(new AggregatorImpls.AggregableFirst<Float>());
                    break;
                case TakeLast:
                    res.add(new AggregatorImpls.AggregableLast<Float>());
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown or not implemented op: " + lop.get(i));
            }
        }
        IAggregableReduceOp<Writable, List<Writable>> thisOp = new FloatWritableOp<>(new AggregableMultiOp<>(res));
        if (ignoreInvalid)
            return new AggregableCheckingOp<>(thisOp, metaData);
        else
            return thisOp;
    }

    public static IAggregableReduceOp<Writable, List<Writable>> reduceDoubleColumn(List<ReduceOp> lop,
                    boolean ignoreInvalid, ColumnMetaData metaData) {

        List<IAggregableReduceOp<Double, Writable>> res = new ArrayList<>(lop.size());
        for (int i = 0; i < lop.size(); i++) {
            switch (lop.get(i)) {
                case Prod:
                    res.add(new AggregatorImpls.AggregableProd<Double>());
                    break;
                case Min:
                    res.add(new AggregatorImpls.AggregableMin<Double>());
                    break;
                case Max:
                    res.add(new AggregatorImpls.AggregableMax<Double>());
                    break;
                case Range:
                    res.add(new AggregatorImpls.AggregableRange<Double>());
                    break;
                case Sum:
                    res.add(new AggregatorImpls.AggregableSum<Double>());
                    break;
                case Mean:
                    res.add(new AggregatorImpls.AggregableMean<Double>());
                    break;
                case Stdev:
                    res.add(new AggregatorImpls.AggregableStdDev<Double>());
                    break;
                case UncorrectedStdDev:
                    res.add(new AggregatorImpls.AggregableUncorrectedStdDev<Double>());
                    break;
                case Variance:
                    res.add(new AggregatorImpls.AggregableVariance<Double>());
                    break;
                case PopulationVariance:
                    res.add(new AggregatorImpls.AggregablePopulationVariance<Double>());
                    break;
                case Count:
                    res.add(new AggregatorImpls.AggregableCount<Double>());
                    break;
                case CountUnique:
                    res.add(new AggregatorImpls.AggregableCountUnique<Double>());
                    break;
                case TakeFirst:
                    res.add(new AggregatorImpls.AggregableFirst<Double>());
                    break;
                case TakeLast:
                    res.add(new AggregatorImpls.AggregableLast<Double>());
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown or not implemented op: " + lop.get(i));
            }
        }
        IAggregableReduceOp<Writable, List<Writable>> thisOp = new DoubleWritableOp<>(new AggregableMultiOp<>(res));
        if (ignoreInvalid)
            return new AggregableCheckingOp<>(thisOp, metaData);
        else
            return thisOp;
    }

    public static IAggregableReduceOp<Writable, List<Writable>> reduceStringOrCategoricalColumn(List<ReduceOp> lop,
                    boolean ignoreInvalid, ColumnMetaData metaData) {

        List<IAggregableReduceOp<String, Writable>> res = new ArrayList<>(lop.size());
        for (int i = 0; i < lop.size(); i++) {
            switch (lop.get(i)) {
                case Count:
                    res.add(new AggregatorImpls.AggregableCount<String>());
                    break;
                case CountUnique:
                    res.add(new AggregatorImpls.AggregableCountUnique<String>());
                    break;
                case TakeFirst:
                    res.add(new AggregatorImpls.AggregableFirst<String>());
                    break;
                case TakeLast:
                    res.add(new AggregatorImpls.AggregableLast<String>());
                    break;
                case Append:
                    res.add(new StringAggregatorImpls.AggregableStringAppend());
                    break;
                case Prepend:
                    res.add(new StringAggregatorImpls.AggregableStringPrepend());
                    break;
                default:
                    throw new UnsupportedOperationException("Cannot execute op \"" + lop.get(i)
                                    + "\" on String/Categorical column "
                                    + "(can only perform Append, Prepend, Count, CountUnique, TakeFirst and TakeLast ops on categorical columns)");
            }
        }

        IAggregableReduceOp<Writable, List<Writable>> thisOp = new StringWritableOp<>(new AggregableMultiOp<>(res));
        if (ignoreInvalid)
            return new AggregableCheckingOp<>(thisOp, metaData);
        else
            return thisOp;
    }

    public static IAggregableReduceOp<Writable, List<Writable>> reduceTimeColumn(List<ReduceOp> lop,
                    boolean ignoreInvalid, ColumnMetaData metaData) {

        List<IAggregableReduceOp<Long, Writable>> res = new ArrayList<>(lop.size());
        for (int i = 0; i < lop.size(); i++) {
            switch (lop.get(i)) {
                case Min:
                    res.add(new AggregatorImpls.AggregableMin<Long>());
                    break;
                case Max:
                    res.add(new AggregatorImpls.AggregableMax<Long>());
                    break;
                case Range:
                    res.add(new AggregatorImpls.AggregableRange<Long>());
                    break;
                case Mean:
                    res.add(new AggregatorImpls.AggregableMean<Long>());
                    break;
                case Stdev:
                    res.add(new AggregatorImpls.AggregableStdDev<Long>());
                    break;
                case Count:
                    res.add(new AggregatorImpls.AggregableCount<Long>());
                    break;
                case CountUnique:
                    res.add(new AggregatorImpls.AggregableCountUnique<Long>());
                    break;
                case TakeFirst:
                    res.add(new AggregatorImpls.AggregableFirst<Long>());
                    break;
                case TakeLast:
                    res.add(new AggregatorImpls.AggregableLast<Long>());
                    break;
                default:
                    throw new UnsupportedOperationException(
                                    "Reduction op \"" + lop.get(i) + "\" not supported on time columns");
            }
        }
        IAggregableReduceOp<Writable, List<Writable>> thisOp = new LongWritableOp<>(new AggregableMultiOp<>(res));
        if (ignoreInvalid)
            return new AggregableCheckingOp<>(thisOp, metaData);
        else
            return thisOp;
    }

    public static IAggregableReduceOp<Writable, List<Writable>> reduceBytesColumn(List<ReduceOp> lop,
                    boolean ignoreInvalid, ColumnMetaData metaData) {

        List<IAggregableReduceOp<Byte, Writable>> res = new ArrayList<>(lop.size());
        for (int i = 0; i < lop.size(); i++) {
            switch (lop.get(i)) {
                case TakeFirst:
                    res.add(new AggregatorImpls.AggregableFirst<Byte>());
                    break;
                case TakeLast:
                    res.add(new AggregatorImpls.AggregableLast<Byte>());
                    break;
                default:
                    throw new UnsupportedOperationException("Cannot execute op \"" + lop.get(i) + "\" on Bytes column "
                                    + "(can only perform TakeFirst and TakeLast ops on bytes columns)");
            }
        }
        IAggregableReduceOp<Writable, List<Writable>> thisOp = new ByteWritableOp<>(new AggregableMultiOp<>(res));
        if (ignoreInvalid)
            return new AggregableCheckingOp<>(thisOp, metaData);
        else
            return thisOp;
    }


}
