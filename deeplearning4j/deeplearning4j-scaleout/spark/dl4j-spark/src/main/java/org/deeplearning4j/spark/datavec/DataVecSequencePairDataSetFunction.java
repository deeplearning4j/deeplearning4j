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

package org.deeplearning4j.spark.datavec;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.io.WritableConverter;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.FeatureUtil;
import scala.Tuple2;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

/**Map {@code Tuple2<Collection<Collection<Writable>>,Collection<Collection<Writable>>} objects (out of a TWO datavec-spark
 *  sequence record reader functions) to  DataSet objects for Spark training.
 * Analogous to {@link SequenceRecordReaderDataSetIterator}, but in the context of Spark.
 * Supports loading data from a TWO sources only; hence supports many-to-one and one-to-many situations.
 * see {@link DataVecSequenceDataSetFunction} for the single file version
 * @author Alex Black
 */
public class DataVecSequencePairDataSetFunction
                implements Function<Tuple2<List<List<Writable>>, List<List<Writable>>>, DataSet>, Serializable {
    /**Alignment mode for dealing with input/labels of differing lengths (for example, one-to-many and many-to-one type situations).
     * For example, might have 10 time steps total but only one label at end for sequence classification.<br>
     * <b>EQUAL_LENGTH</b>: Default. Assume that label and input time series are of equal length<br>
     * <b>ALIGN_START</b>: Align the label/input time series at the first time step, and zero pad either the labels or
     * the input at the end (pad whichever is shorter)<br>
     * <b>ALIGN_END</b>: Align the label/input at the last time step, zero padding either the input or the labels as required<br>
     */
    public enum AlignmentMode {
        EQUAL_LENGTH, ALIGN_START, ALIGN_END
    }

    private final boolean regression;
    private final int numPossibleLabels;
    private final AlignmentMode alignmentMode;
    private final DataSetPreProcessor preProcessor;
    private final WritableConverter converter;

    /** Constructor for equal length and no conversion of labels (i.e., regression or already in one-hot representation).
     * No data set proprocessor or writable converter
     */
    public DataVecSequencePairDataSetFunction() {
        this(-1, true);
    }

    /**Constructor for equal length, no data set preprocessor or writable converter
     * @see #DataVecSequencePairDataSetFunction(int, boolean, AlignmentMode, DataSetPreProcessor, WritableConverter)
     */
    public DataVecSequencePairDataSetFunction(int numPossibleLabels, boolean regression) {
        this(numPossibleLabels, regression, AlignmentMode.EQUAL_LENGTH);
    }

    /**Constructor for data with a specified alignment mode, no data set preprocessor or writable converter
     * @see #DataVecSequencePairDataSetFunction(int, boolean, AlignmentMode, DataSetPreProcessor, WritableConverter)
     */
    public DataVecSequencePairDataSetFunction(int numPossibleLabels, boolean regression, AlignmentMode alignmentMode) {
        this(numPossibleLabels, regression, alignmentMode, null, null);
    }

    /**
     * @param numPossibleLabels Number of classes for classification  (not used if regression = true)
     * @param regression False for classification, true for regression
     * @param alignmentMode Alignment mode for data. See {@link DataVecSequencePairDataSetFunction.AlignmentMode}
     * @param preProcessor DataSetPreprocessor (may be null)
     * @param converter WritableConverter (may be null)
     */
    public DataVecSequencePairDataSetFunction(int numPossibleLabels, boolean regression, AlignmentMode alignmentMode,
                    DataSetPreProcessor preProcessor, WritableConverter converter) {
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
        this.alignmentMode = alignmentMode;
        this.preProcessor = preProcessor;
        this.converter = converter;
    }


    @Override
    public DataSet call(Tuple2<List<List<Writable>>, List<List<Writable>>> input) throws Exception {
        List<List<Writable>> featuresSeq = input._1();
        List<List<Writable>> labelsSeq = input._2();

        int featuresLength = featuresSeq.size();
        int labelsLength = labelsSeq.size();


        Iterator<List<Writable>> fIter = featuresSeq.iterator();
        Iterator<List<Writable>> lIter = labelsSeq.iterator();

        INDArray inputArr = null;
        INDArray outputArr = null;

        int[] idx = new int[3];
        int i = 0;
        while (fIter.hasNext()) {
            List<Writable> step = fIter.next();
            if (i == 0) {
                int[] inShape = new int[] {1, step.size(), featuresLength};
                inputArr = Nd4j.create(inShape);
            }
            Iterator<Writable> timeStepIter = step.iterator();
            int f = 0;
            idx[1] = 0;
            while (timeStepIter.hasNext()) {
                Writable current = timeStepIter.next();
                if (converter != null)
                    current = converter.convert(current);
                try {
                    inputArr.putScalar(idx, current.toDouble());
                } catch (UnsupportedOperationException e) {
                    // This isn't a scalar, so check if we got an array already
                    if (current instanceof NDArrayWritable) {
                        inputArr.get(NDArrayIndex.point(idx[0]), NDArrayIndex.all(), NDArrayIndex.point(idx[2]))
                                        .putRow(0, ((NDArrayWritable) current).get());
                    } else {
                        throw e;
                    }
                }
                idx[1] = ++f;
            }
            idx[2] = ++i;
        }

        idx = new int[3];
        i = 0;
        while (lIter.hasNext()) {
            List<Writable> step = lIter.next();
            if (i == 0) {
                int[] outShape = new int[] {1, (regression ? step.size() : numPossibleLabels), labelsLength};
                outputArr = Nd4j.create(outShape);
            }
            Iterator<Writable> timeStepIter = step.iterator();
            int f = 0;
            idx[1] = 0;
            if (regression) {
                //Load all values without modification
                while (timeStepIter.hasNext()) {
                    Writable current = timeStepIter.next();
                    if (converter != null)
                        current = converter.convert(current);
                    outputArr.putScalar(idx, current.toDouble());
                    idx[1] = ++f;
                }
            } else {
                //Expect a single value (index) -> convert to one-hot vector
                Writable value = timeStepIter.next();
                int labelClassIdx = value.toInt();
                INDArray line = FeatureUtil.toOutcomeVector(labelClassIdx, numPossibleLabels);
                outputArr.tensorAlongDimension(i, 1).assign(line); //1d from [1,nOut,timeSeriesLength] -> tensor i along dimension 1 is at time i
            }

            idx[2] = ++i;
        }

        DataSet ds;
        if (alignmentMode == AlignmentMode.EQUAL_LENGTH || featuresLength == labelsLength) {
            ds = new DataSet(inputArr, outputArr);
        } else if (alignmentMode == AlignmentMode.ALIGN_END) {
            if (featuresLength > labelsLength) {
                //Input longer, pad output
                INDArray newOutput = Nd4j.create(1, outputArr.size(1), featuresLength);
                newOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                                NDArrayIndex.interval(featuresLength - labelsLength, featuresLength)).assign(outputArr);
                //Need an output mask array, but not an input mask array
                INDArray outputMask = Nd4j.create(1, featuresLength);
                for (int j = featuresLength - labelsLength; j < featuresLength; j++)
                    outputMask.putScalar(j, 1.0);
                ds = new DataSet(inputArr, newOutput, Nd4j.ones(outputMask.shape()), outputMask);
            } else {
                //Output longer, pad input
                INDArray newInput = Nd4j.create(1, inputArr.size(1), labelsLength);
                newInput.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                                NDArrayIndex.interval(labelsLength - featuresLength, labelsLength)).assign(inputArr);
                //Need an input mask array, but not an output mask array
                INDArray inputMask = Nd4j.create(1, labelsLength);
                for (int j = labelsLength - featuresLength; j < labelsLength; j++)
                    inputMask.putScalar(j, 1.0);
                ds = new DataSet(newInput, outputArr, inputMask, Nd4j.ones(inputMask.shape()));
            }
        } else if (alignmentMode == AlignmentMode.ALIGN_START) {
            if (featuresLength > labelsLength) {
                //Input longer, pad output
                INDArray newOutput = Nd4j.create(1, outputArr.size(1), featuresLength);
                newOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.interval(0, labelsLength))
                                .assign(outputArr);
                //Need an output mask array, but not an input mask array
                INDArray outputMask = Nd4j.create(1, featuresLength);
                for (int j = 0; j < labelsLength; j++)
                    outputMask.putScalar(j, 1.0);
                ds = new DataSet(inputArr, newOutput, Nd4j.ones(outputMask.shape()), outputMask);
            } else {
                //Output longer, pad input
                INDArray newInput = Nd4j.create(1, inputArr.size(1), labelsLength);
                newInput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.interval(0, featuresLength))
                                .assign(inputArr);
                //Need an input mask array, but not an output mask array
                INDArray inputMask = Nd4j.create(1, labelsLength);
                for (int j = 0; j < featuresLength; j++)
                    inputMask.putScalar(j, 1.0);
                ds = new DataSet(newInput, outputArr, inputMask, Nd4j.ones(inputMask.shape()));
            }
        } else {
            throw new UnsupportedOperationException("Invalid alignment mode: " + alignmentMode);
        }


        if (preProcessor != null)
            preProcessor.preProcess(ds);
        return ds;
    }
}
