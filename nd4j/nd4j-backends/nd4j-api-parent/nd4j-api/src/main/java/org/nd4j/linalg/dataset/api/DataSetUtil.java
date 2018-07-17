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

package org.nd4j.linalg.dataset.api;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * Created by susaneraly on 9/20/16.
 */
@Slf4j
public class DataSetUtil {
    public static INDArray tailor2d(@NonNull DataSet dataSet, boolean areFeatures) {
        return tailor2d(areFeatures ? dataSet.getFeatures() : dataSet.getLabels(),
                        areFeatures ? dataSet.getFeaturesMaskArray() : dataSet.getLabelsMaskArray());
    }

    public static INDArray tailor2d(@NonNull INDArray data, INDArray mask) {
        switch (data.rank()) {
            case 1:
            case 2:
                return data;
            case 3:
                return tailor3d2d(data, mask);
            case 4:
                return tailor4d2d(data);
            default:
                throw new RuntimeException("Unsupported data rank");
        }
    }

    /**
     * @deprecated
     */
    public static INDArray tailor3d2d(DataSet dataset, boolean areFeatures) {
        INDArray data = areFeatures ? dataset.getFeatures() : dataset.getLabels();
        INDArray mask = areFeatures ? dataset.getFeaturesMaskArray() : dataset.getLabelsMaskArray();
        return tailor3d2d(data, mask);
    }

    public static INDArray tailor3d2d(@NonNull INDArray data, INDArray mask) {
        //Check mask shapes:
        if (mask != null) {
            if (data.size(0) != mask.size(0) || data.size(2) != mask.size(1)) {
                throw new IllegalArgumentException(
                                "Invalid mask array/data combination: got data with shape [minibatch, vectorSize, timeSeriesLength] = "
                                                + Arrays.toString(data.shape())
                                                + "; got mask with shape [minibatch,timeSeriesLength] = "
                                                + Arrays.toString(mask.shape())
                                                + "; minibatch and timeSeriesLength dimensions must match");
            }
        }


        if (data.ordering() != 'f' || data.isView() || !Shape.strideDescendingCAscendingF(data)) {
            data = data.dup('f');
        }
        //F order: strides are like [1, miniBatch, minibatch*size] - i.e., each time step array is contiguous in memory
        //This can be reshaped to 2d with a no-copy op
        //Same approach as RnnToFeedForwardPreProcessor in DL4J
        //I.e., we're effectively stacking time steps for all examples

        val shape = data.shape();
        INDArray as2d;
        if (shape[0] == 1) {
            as2d = data.tensorAlongDimension(0, 1, 2).permutei(1, 0); //Edge case: miniBatchSize==1
        } else if (shape[2] == 1) {
            as2d = data.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        } else {
            INDArray permuted = data.permute(0, 2, 1); //Permute, so we get correct order after reshaping
            as2d = permuted.reshape('f', shape[0] * shape[2], shape[1]);
        }

        if (mask == null) {
            return as2d;
        }

        //With stride 1 along the examples (dimension 0), we are concatenating time series - same as the
        if (mask.ordering() != 'f' || mask.isView() || !Shape.strideDescendingCAscendingF(mask)) {
            mask = mask.dup('f');
        }

        INDArray mask1d = mask.reshape('f', new long[] {mask.length(), 1});

        //Assume masks are 0s and 1s: then sum == number of elements
        int numElements = mask.sumNumber().intValue();
        if (numElements == mask.length()) {
            return as2d; //All are 1s
        }
        if (numElements == 0) {
            return null;
        }

        int[] rowsToPull = new int[numElements];
        float[] floatMask1d = mask1d.data().asFloat();
        int currCount = 0;
        for (int i = 0; i < floatMask1d.length; i++) {
            if (floatMask1d[i] != 0.0f) {
                rowsToPull[currCount++] = i;
            }
        }

        INDArray subset = Nd4j.pullRows(as2d, 1, rowsToPull); //Tensor along dimension 1 == rows
        return subset;
    }

    public static INDArray tailor4d2d(DataSet dataset, boolean areFeatures) {
        return tailor4d2d(areFeatures ? dataset.getFeatures() : dataset.getLabels());
    }

    public static INDArray tailor4d2d(@NonNull INDArray data) {
        long instances = data.size(0);
        long channels = data.size(1);
        long height = data.size(2);
        long width = data.size(3);

        INDArray in2d = Nd4j.create(channels, height * width * instances);

        long tads = data.tensorssAlongDimension(3, 2, 0);
        for (int i = 0; i < tads; i++) {
            INDArray thisTAD = data.tensorAlongDimension(i, 3, 2, 0);
            in2d.putRow(i, Nd4j.toFlattened(thisTAD));
        }
        return in2d.transposei();
    }

    public static void setMaskedValuesToZero(INDArray data, INDArray mask) {
        if (mask == null || data.rank() != 3)
            return;

        Nd4j.getExecutioner().exec(new BroadcastMulOp(data, mask, data, 0, 2));
    }

    /**
     * Merge all of the features arrays into one minibatch.
     *
     * @param featuresToMerge     features to merge. Note that first index is the input array (example) index, the second
     *                            index is the input array.
     *                            Thus to merge 10 examples with 3 input arrays each, featuresToMerge will be indexed
     *                            like featuresToMerge[0..9][0..2]
     * @param featureMasksToMerge May be null. If non-null: feature masks to merge
     * @return Merged features, and feature masks. Note that feature masks may be added automatically, if required - even
     * if no feature masks were present originally
     */
    public static Pair<INDArray[], INDArray[]> mergeFeatures(@NonNull INDArray[][] featuresToMerge, INDArray[][] featureMasksToMerge) {
        int nInArrs = featuresToMerge[0].length;
        INDArray[] outF = new INDArray[nInArrs];
        INDArray[] outM = null;

        for (int i = 0; i < nInArrs; i++) {
            Pair<INDArray, INDArray> p = mergeFeatures(featuresToMerge, featureMasksToMerge, i);
            outF[i] = p.getFirst();
            if (p.getSecond() != null) {
                if (outM == null) {
                    outM = new INDArray[nInArrs];
                }
                outM[i] = p.getSecond();
            }
        }
        return new Pair<>(outF, outM);
    }

    /**
     * Merge the specified features and mask arrays (i.e., concatenate the examples)
     *
     * @param featuresToMerge     Features to merge
     * @param featureMasksToMerge Mask arrays to merge. May be null
     * @return Merged features and mask. Mask may be null
     */
    public static Pair<INDArray, INDArray> mergeFeatures(@NonNull INDArray[] featuresToMerge,
                    INDArray[] featureMasksToMerge) {
        int rankFeatures = featuresToMerge[0].rank();

        switch (rankFeatures) {
            case 2:
                return DataSetUtil.merge2d(featuresToMerge, featureMasksToMerge);
            case 3:
                return DataSetUtil.mergeTimeSeries(featuresToMerge, featureMasksToMerge);
            case 4:
                return DataSetUtil.merge4d(featuresToMerge, featureMasksToMerge);
            default:
                throw new IllegalStateException("Cannot merge examples: features rank must be in range 2 to 4"
                                + " inclusive. First example features shape: "
                                + Arrays.toString(featureMasksToMerge[0].shape()));
        }
    }

    /**
     * Extract out the specified column, and merge the specified features and mask arrays (i.e., concatenate the examples)
     *
     * @param featuresToMerge     Features to merge. Will use featuresToMerge[all][inOutIdx]
     * @param featureMasksToMerge Mask arrays to merge. May be null
     * @return Merged features and mask. Mask may be null
     */
    public static Pair<INDArray, INDArray> mergeFeatures(INDArray[][] featuresToMerge, INDArray[][] featureMasksToMerge,
                    int inOutIdx) {
        Pair<INDArray[], INDArray[]> p = selectColumnFromMDSData(featuresToMerge, featureMasksToMerge, inOutIdx);
        return mergeFeatures(p.getFirst(), p.getSecond());
    }

    /**
     * Merge the specified labels and label mask arrays (i.e., concatenate the examples)
     *
     * @param labelsToMerge     Features to merge
     * @param labelMasksToMerge Mask arrays to merge. May be null
     * @return Merged features and mask. Mask may be null
     */
    public static Pair<INDArray, INDArray> mergeLabels(INDArray[] labelsToMerge, INDArray[] labelMasksToMerge) {
        int rankFeatures = labelsToMerge[0].rank();

        switch (rankFeatures) {
            case 2:
                return DataSetUtil.merge2d(labelsToMerge, labelMasksToMerge);
            case 3:
                return DataSetUtil.mergeTimeSeries(labelsToMerge, labelMasksToMerge);
            case 4:
                return DataSetUtil.merge4d(labelsToMerge, labelMasksToMerge);
            default:
                throw new ND4JIllegalStateException("Cannot merge examples: labels rank must be in range 2 to 4"
                                + " inclusive. First example features shape: "
                                + Arrays.toString(labelsToMerge[0].shape()));
        }
    }

    /**
     * Extract out the specified column, and merge the specified label and label mask arrays
     * (i.e., concatenate the examples)
     *
     * @param labelsToMerge     Features to merge. Will use featuresToMerge[all][inOutIdx]
     * @param labelMasksToMerge Mask arrays to merge. May be null
     * @return Merged features and mask. Mask may be null
     */
    public static Pair<INDArray, INDArray> mergeLabels(@NonNull INDArray[][] labelsToMerge,
                    INDArray[][] labelMasksToMerge, int inOutIdx) {
        Pair<INDArray[], INDArray[]> p = selectColumnFromMDSData(labelsToMerge, labelMasksToMerge, inOutIdx);
        return mergeLabels(p.getFirst(), p.getSecond());
    }

    private static Pair<INDArray[], INDArray[]> selectColumnFromMDSData(@NonNull INDArray[][] arrays,
                    INDArray[][] masks, int inOutIdx) {
        INDArray[] a = new INDArray[arrays.length];
        INDArray[] m = new INDArray[a.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = arrays[i][inOutIdx];
            if (masks != null && masks[i] != null) {
                m[i] = masks[i][inOutIdx];
            }
        }
        return new Pair<>(a, m);
    }

    /**
     * Merge the specified 2d arrays and masks. See {@link #mergeFeatures(INDArray[], INDArray[])}
     * and {@link #mergeLabels(INDArray[], INDArray[])}
     *
     * @param arrays   Arrays to merge
     * @param masks    Mask arrays to merge
     * @param inOutIdx Index to extract out before merging
     * @return Merged arrays and mask
     */
    public static Pair<INDArray, INDArray> merge2d(@NonNull INDArray[][] arrays, INDArray[][] masks, int inOutIdx) {
        Pair<INDArray[], INDArray[]> p = selectColumnFromMDSData(arrays, masks, inOutIdx);
        return merge2d(p.getFirst(), p.getSecond());
    }

    /**
     * Merge the specified 2d arrays and masks. See {@link #mergeFeatures(INDArray[], INDArray[])}
     * and {@link #mergeLabels(INDArray[], INDArray[])}
     *
     * @param arrays   Arrays to merge
     * @param masks    Mask arrays to merge
     * @return Merged arrays and mask
     */
    public static Pair<INDArray, INDArray> merge2d(INDArray[] arrays, INDArray[] masks) {
        long cols = arrays[0].columns();

        INDArray[] temp = new INDArray[arrays.length];
        boolean hasMasks = false;
        for (int i = 0; i < arrays.length; i++) {
            if (arrays[i].columns() != cols) {
                throw new IllegalStateException("Cannot merge 2d arrays with different numbers of columns (firstNCols="
                                + cols + ", ithNCols=" + arrays[i].columns() + ")");
            }

            temp[i] = arrays[i];

            if (masks != null && masks[i] != null && masks[i] != null) {
                hasMasks = true;
            }
        }

        INDArray out = Nd4j.specialConcat(0, temp);
        INDArray outMask = null;
        if (hasMasks) {
            outMask = DataSetUtil.mergePerOutputMasks2d(out.shape(), arrays, masks);
        }

        return new Pair<>(out, outMask);
    }


    public static INDArray mergePerOutputMasks2d(long[] outShape, INDArray[][] arrays, INDArray[][] masks,
                    int inOutIdx) {
        Pair<INDArray[], INDArray[]> p = selectColumnFromMDSData(arrays, masks, inOutIdx);
        return mergePerOutputMasks2d(outShape, p.getFirst(), p.getSecond());
    }

    public static INDArray mergePerOutputMasks2d(long[] outShape, INDArray[] arrays, INDArray[] masks) {
        val numExamplesPerArr = new long[arrays.length];
        for (int i = 0; i < numExamplesPerArr.length; i++) {
            numExamplesPerArr[i] = arrays[i].size(0);
        }

        INDArray outMask = Nd4j.ones(outShape); //Initialize to 'all present' (1s)

        int rowsSoFar = 0;
        for (int i = 0; i < masks.length; i++) {
            long thisRows = numExamplesPerArr[i]; //Mask itself may be null -> all present, but may include multiple examples
            if (masks[i] == null) {
                continue;
            }

            outMask.put(new INDArrayIndex[] {NDArrayIndex.interval(rowsSoFar, rowsSoFar + thisRows),
                            NDArrayIndex.all()}, masks[i]);
            rowsSoFar += thisRows;
        }
        return outMask;
    }

    /**
     * Merge the specified time series (3d) arrays and masks. See {@link #mergeFeatures(INDArray[], INDArray[])}
     * and {@link #mergeLabels(INDArray[], INDArray[])}
     *
     * @param arrays   Arrays to merge
     * @param masks    Mask arrays to merge
     * @param inOutIdx Index to extract out before merging
     * @return Merged arrays and mask
     */
    public static Pair<INDArray, INDArray> mergeTimeSeries(INDArray[][] arrays, INDArray[][] masks, int inOutIdx) {
        Pair<INDArray[], INDArray[]> p = selectColumnFromMDSData(arrays, masks, inOutIdx);
        return mergeTimeSeries(p.getFirst(), p.getSecond());
    }

    /**
     * Merge the specified time series (3d) arrays and masks. See {@link #mergeFeatures(INDArray[], INDArray[])}
     * and {@link #mergeLabels(INDArray[], INDArray[])}
     *
     * @param arrays   Arrays to merge
     * @param masks    Mask arrays to merge
     * @return Merged arrays and mask
     */
    public static Pair<INDArray, INDArray> mergeTimeSeries(INDArray[] arrays, INDArray[] masks) {
        //Merge time series data, and handle masking etc for different length arrays

        //Complications with time series:
        //(a) They may have different lengths (if so: need input + output masking arrays)
        //(b) Even if they are all the same length, they may have masking arrays (if so: merge the masking arrays too)
        //(c) Furthermore: mask arrays can be per-time-step (2d) or per output (3d). Per-input masks (3d feature masks)
        //    are not supported, however

        long firstLength = arrays[0].size(2);
        long size = arrays[0].size(1);
        long maxLength = firstLength;

        boolean hasMask = false;
        int maskRank = -1;
        boolean lengthsDiffer = false;
        int totalExamples = 0;
        for (int i = 0; i < arrays.length; i++) {
            totalExamples += arrays[i].size(0);
            long thisLength = arrays[i].size(2);
            maxLength = Math.max(maxLength, thisLength);
            if (thisLength != firstLength)
                lengthsDiffer = true;
            if (masks != null && masks[i] != null && masks[i] != null) {
                maskRank = masks[i].rank();
                hasMask = true;
            }

            if (arrays[i].size(1) != size) {
                throw new IllegalStateException(
                                "Cannot merge time series with different size for dimension 1 (first shape: "
                                                + Arrays.toString(arrays[0].shape()) + ", " + i + "th shape: "
                                                + Arrays.toString(arrays[i].shape()));
            }
        }

        boolean needMask = hasMask || lengthsDiffer;
        INDArray arr = Nd4j.create(totalExamples, size, maxLength);
        INDArray mask = (needMask && maskRank != 3 ? Nd4j.ones(totalExamples, maxLength) : null);

        //Now, merge the time series (and if necessary, mask arrays):
        int examplesSoFar = 0;
        if (!lengthsDiffer && !needMask) {
            //Simplest case: same length, no mask arrays
            for (int i = 0; i < arrays.length; i++) {
                long thisNExamples = arrays[i].size(0);
                arr.put(new INDArrayIndex[] {NDArrayIndex.interval(examplesSoFar, examplesSoFar + thisNExamples),
                                NDArrayIndex.all(), NDArrayIndex.all()}, arrays[i]);
                examplesSoFar += thisNExamples;
            }
            return new Pair<>(arr, null);
        } else {
            //Either different length, or have mask arrays (or, both)
            if ((lengthsDiffer && !hasMask) || maskRank == 2) {
                //Standard per-example masking required
                for (int i = 0; i < arrays.length; i++) {
                    INDArray a = arrays[i];
                    long thisNExamples = a.size(0);
                    long thisLength = a.size(2);
                    arr.put(new INDArrayIndex[] {NDArrayIndex.interval(examplesSoFar, examplesSoFar + thisNExamples),
                                    NDArrayIndex.all(), NDArrayIndex.interval(0, thisLength)}, a);

                    if (masks != null && masks[i] != null && masks[i] != null) {
                        INDArray origMask = masks[i];
                        long maskLength = origMask.size(1);
                        mask.put(new INDArrayIndex[] {
                                        NDArrayIndex.interval(examplesSoFar, examplesSoFar + thisNExamples),
                                        NDArrayIndex.interval(0, maskLength)}, origMask);
                        if (maskLength < maxLength) {
                            //Set end mask array to zero...
                            mask.put(new INDArrayIndex[] {
                                            NDArrayIndex.interval(examplesSoFar, examplesSoFar + thisNExamples),
                                            NDArrayIndex.interval(maskLength, maxLength)},
                                            Nd4j.zeros(thisNExamples, maxLength - maskLength));
                        }
                    } else {
                        if (thisLength < maxLength) {
                            //Mask the end
                            mask.put(new INDArrayIndex[] {
                                            NDArrayIndex.interval(examplesSoFar, examplesSoFar + thisNExamples),
                                            NDArrayIndex.interval(thisLength, maxLength)},
                                            Nd4j.zeros(thisNExamples, maxLength - thisLength));
                        }
                    }

                    examplesSoFar += thisNExamples;
                }
            } else if (maskRank == 3) {
                //Per output masking required. May also be variable length
                mask = Nd4j.create(arr.shape());
                for (int i = 0; i < arrays.length; i++) {
                    INDArray m = masks[i];
                    INDArray a = arrays[i];
                    long thisNExamples = a.size(0);
                    long thisLength = a.size(2);
                    arr.put(new INDArrayIndex[] {NDArrayIndex.interval(examplesSoFar, examplesSoFar + thisNExamples),
                                    NDArrayIndex.all(), NDArrayIndex.interval(0, thisLength)}, a);

                    if (m == null) {
                        //This mask is null -> equivalent to "all present"
                        mask.get(NDArrayIndex.interval(examplesSoFar, examplesSoFar + thisNExamples),
                                        NDArrayIndex.all(), NDArrayIndex.interval(0, thisLength)).assign(1);
                    } else {
                        mask.put(new INDArrayIndex[] {
                                        NDArrayIndex.interval(examplesSoFar, examplesSoFar + thisNExamples),
                                        NDArrayIndex.all(), NDArrayIndex.interval(0, thisLength)}, m);
                    }

                    examplesSoFar += thisNExamples;
                }
            } else {
                throw new UnsupportedOperationException("Cannot merge time series with mask rank " + maskRank);
            }
        }

        return new Pair<>(arr, mask);
    }

    /**
     * Merge the specified 4d arrays and masks. See {@link #mergeFeatures(INDArray[], INDArray[])}
     * and {@link #mergeLabels(INDArray[], INDArray[])}
     *
     * @param arrays   Arrays to merge
     * @param masks    Mask arrays to merge
     * @param inOutIdx Index to extract out before merging
     * @return Merged arrays and mask
     */
    public static Pair<INDArray, INDArray> merge4d(INDArray[][] arrays, INDArray[][] masks, int inOutIdx) {
        Pair<INDArray[], INDArray[]> p = selectColumnFromMDSData(arrays, masks, inOutIdx);
        return merge4d(p.getFirst(), p.getSecond());
    }

    /**
     * Merge the specified 4d arrays and masks. See {@link #mergeFeatures(INDArray[], INDArray[])}
     * and {@link #mergeLabels(INDArray[], INDArray[])}
     *
     * @param arrays   Arrays to merge
     * @param masks    Mask arrays to merge
     * @return Merged arrays and mask
     */
    public static Pair<INDArray, INDArray> merge4d(INDArray[] arrays, INDArray[] masks) {
        //4d -> images. In principle: could have 2d mask arrays (per-example masks)

        int nExamples = 0;
        long[] shape = arrays[0].shape();
        INDArray[] temp = new INDArray[arrays.length];
        boolean hasMasks = false;
        for (int i = 0; i < arrays.length; i++) {
            nExamples += arrays[i].size(0);
            long[] thisShape = arrays[i].shape();
            if (thisShape.length != 4) {
                throw new IllegalStateException("Cannot merge 4d arrays with non 4d arrays");
            }
            for (int j = 1; j < 4; j++) {
                if (thisShape[j] != shape[j])
                    throw new IllegalStateException(
                                    "Cannot merge 4d arrays with different shape (other than # examples): "
                                                    + " data[0].shape = " + Arrays.toString(shape) + ", data[" + i
                                                    + "].shape = " + Arrays.toString(thisShape));
            }

            temp[i] = arrays[i];
            if (masks != null && masks[i] != null && masks[i] != null) {
                hasMasks = true;
                if (masks[i].rank() != 2) {
                    throw new UnsupportedOperationException("Cannot merged 4d arrays with masks that are not rank 2."
                                    + " Got mask array with rank: " + masks[i].rank());
                }
            }
        }

        INDArray out = Nd4j.specialConcat(0, temp);
        INDArray outMask = null;
        if (hasMasks) {
            outMask = DataSetUtil.mergePerOutputMasks2d(out.shape(), arrays, masks);
        }

        return new Pair<>(out, outMask);
    }
}
