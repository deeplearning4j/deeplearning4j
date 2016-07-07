package org.datavec.api.transform.analysis.sequence;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

/**
 * Created by Alex on 12/03/2016.
 */
@AllArgsConstructor @Data @Builder
public class SequenceLengthAnalysis {

    private final long totalNumSequences;
    private final int minSeqLength;
    private final int maxSeqLength;
    private final long countZeroLength;
    private final long countOneLength;
    private final double meanLength;
    private final double[] histogramBuckets;
    private final long[] histogramBucketCounts;

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("SequenceLengthAnalysis(")
                .append("totalNumSequences=").append(totalNumSequences)
                .append(",minSeqLength=").append(minSeqLength)
                .append(",maxSeqLength=").append(maxSeqLength)
                .append(",countZeroLength=").append(countZeroLength)
                .append(",countOneLength=").append(countOneLength)
                .append(",meanLength=").append(meanLength)
                .append(")");
        return sb.toString();
    }

}
