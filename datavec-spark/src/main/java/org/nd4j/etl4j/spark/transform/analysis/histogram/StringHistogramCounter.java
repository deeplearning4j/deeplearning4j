package org.nd4j.etl4j.spark.transform.analysis.histogram;

import org.nd4j.etl4j.api.writable.Writable;

/**
 * A counter for building histograms (of String length) on a String column
 *
 * @author Alex Black
 */
public class StringHistogramCounter implements HistogramCounter {

    private final int minLength;
    private final int maxLength;
    private final int nBins;
    private final double[] bins;
    private final long[] binCounts;

    public StringHistogramCounter(int minLength, int maxLength, int nBins) {
        this.minLength = minLength;
        this.maxLength = maxLength;
        this.nBins = nBins;

        bins = new double[nBins+1]; //+1 because bins are defined by a range of values: bins[i] to bins[i+1]
        double step = ((double)(maxLength-minLength))/nBins;
        for( int i=0; i<bins.length; i++ ){
            if(i == bins.length-1) bins[i] = maxLength;
            else bins[i] = i * step;
        }

        binCounts = new long[nBins];
    }


    @Override
    public HistogramCounter add(Writable w) {
        double d = w.toString().length();

        //Not super efficient, but linear search on 20-50 items should be good enough
        int idx = -1;
        for( int i=0; i<nBins; i++ ){
            if(d >= bins[i] && d < bins[i]){
                idx = i;
                break;
            }
        }
        if(idx == -1) idx = nBins-1;

        binCounts[idx]++;

        return this;
    }

    @Override
    public StringHistogramCounter merge(HistogramCounter other) {
        if(other == null) return this;
        if(!(other instanceof StringHistogramCounter)) throw new IllegalArgumentException("Cannot merge " + other);

        StringHistogramCounter o = (StringHistogramCounter)other;

        if(minLength != o.minLength || maxLength != o.maxLength) throw new IllegalStateException("Min/max values differ: (" + minLength + "," + maxLength + ") "
            + " vs. (" + o.minLength + "," + o.maxLength + ")");
        if(nBins != o.nBins) throw new IllegalStateException("Different number of bins: " + nBins + " vs " + o.nBins);

        for( int i=0; i<nBins; i++ ){
            binCounts[i] += o.binCounts[i];
        }

        return this;
    }

    @Override
    public double[] getBins() {
        return bins;
    }

    @Override
    public long[] getCounts() {
        return binCounts;
    }
}
