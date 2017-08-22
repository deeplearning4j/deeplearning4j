package org.datavec.api.timeseries.util;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * Simple utils for converting {@link Writable} s
 * lists to {@link INDArray}
 *
 * @author Adam Gibson
 */
public class TimeSeriesWritableUtils {

    /**
     * Unchecked exception, thrown to signify that a zero-length sequence data set was encountered.
     */
    public static class ZeroLengthSequenceException extends RuntimeException {
        public ZeroLengthSequenceException() {
            this("");
        }

        public ZeroLengthSequenceException(String type) {
            super(String.format("Encountered zero-length %ssequence", type.equals("") ? "" : type + " "));
        }
    }


    @AllArgsConstructor
    @Builder
    @Getter
    public static class RecordDetails {
        private int minValues;
        private int maxTSLength;

    }

    /**
     * Get the {@link RecordDetails}
     * detailing the length of the time series
     * @param record the input time series
     *               to get the details for
     * @return the record details for the record
     */
    public static RecordDetails getDetails(List<List<List<Writable>>> record) {
        int maxTimeSeriesLength = 0;
        for(List<List<Writable>> step : record) {
            maxTimeSeriesLength = Math.max(maxTimeSeriesLength,step.size());

        }

        return RecordDetails.builder()
                .minValues(record.size())
                .maxTSLength(maxTimeSeriesLength).build();
    }

    /**
     * Convert the writables
     * to a sequence (3d) data set,
     * and also return the
     * mask array (if necessary)
     * @param timeSeriesRecord the input time series
     *
     */
    public static Pair<INDArray, INDArray> convertWritablesSequence(List<List<List<Writable>>> timeSeriesRecord) {
        return convertWritablesSequence(timeSeriesRecord,getDetails(timeSeriesRecord));
    }

    /**
     * Convert the writables
     * to a sequence (3d) data set,
     * and also return the
     * mask array (if necessary)
     */
    public static Pair<INDArray, INDArray> convertWritablesSequence(List<List<List<Writable>>> list,
                                                                    RecordDetails details) {


        INDArray arr;

        if (list.get(0).size() == 0) {
            throw new ZeroLengthSequenceException("Zero length sequence encountered");
        }

        List<Writable> firstStep = list.get(0).get(0);

        int size = 0;
        //Need to account for NDArrayWritables etc in list:
        for (Writable w : firstStep) {
            if (w instanceof NDArrayWritable) {
                size += ((NDArrayWritable) w).get().size(1);
            } else {
                size++;
            }
        }


        arr = Nd4j.create(new int[] {details.getMinValues(), size, details.getMaxTSLength()}, 'f');

        boolean needMaskArray = false;
        for (List<List<Writable>> c : list) {
            if (c.size() < details.getMaxTSLength()) {
                needMaskArray = true;
                break;
            }
        }


        INDArray maskArray;
        if (needMaskArray) {
            maskArray = Nd4j.ones(details.getMinValues(), details.getMaxTSLength());
        } else {
            maskArray = null;
        }



        for (int i = 0; i < details.getMinValues(); i++) {
            List<List<Writable>> sequence = list.get(i);
            int t = 0;
            int k;
            for (List<Writable> timeStep : sequence) {
                k =  t++;

                //Convert entire reader contents, without modification
                Iterator<Writable> iter = timeStep.iterator();
                int j = 0;
                while (iter.hasNext()) {
                    Writable w = iter.next();

                    if (w instanceof NDArrayWritable) {
                        INDArray row = ((NDArrayWritable) w).get();

                        arr.put(new INDArrayIndex[] {NDArrayIndex.point(i),
                                NDArrayIndex.interval(j, j + row.length()), NDArrayIndex.point(k)}, row);
                        j += row.length();
                    } else {
                        arr.putScalar(i, j, k, w.toDouble());
                        j++;
                    }
                }



            }

            //For any remaining time steps: set mask array to 0 (just padding)
            if (needMaskArray) {
                //Masking array entries at end (for align start)
                int lastStep =  sequence.size();
                for (int t2 = lastStep; t2 < details.getMaxTSLength(); t2++) {
                    maskArray.putScalar(i, t2, 0.0);
                }

            }
        }

        return new Pair<>(arr, maskArray);
    }

}
