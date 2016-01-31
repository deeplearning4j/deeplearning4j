package org.deeplearning4j.ui.weights;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author raver119@gmail.com
 */
@Data
public class HistogramBin implements Serializable {
    private transient INDArray sourceArray;
    private int numberOfBins;
    private int rounds;
    private transient INDArray bins;
    private double max;
    private double min;
    private Map<BigDecimal, AtomicInteger> data = new LinkedHashMap<>();

    private static final Logger log = LoggerFactory.getLogger(HistogramBin.class);

    /**
     * No-Args constructor should be used only for serialization/deserialization purposes.
     * In all other cases please use Histogram.Builder()
     */
    public HistogramBin() {

    }

    /**
     * Builds histogram bin for specified array
     * @param array
     */
    public HistogramBin(INDArray array) {

    }

    @JsonIgnore
    private synchronized void calcHistogram() {
        max = sourceArray.maxNumber().doubleValue();
        min = sourceArray.minNumber().doubleValue();

        bins = Nd4j.create(numberOfBins);
        final double binSize = (max - min) / (numberOfBins - 1);


        data = new LinkedHashMap<>();
        BigDecimal[] keys = new BigDecimal[numberOfBins];

        for (int x = 0; x < numberOfBins; x++) {
            BigDecimal pos = new BigDecimal((min + (x * binSize))).setScale(rounds, BigDecimal.ROUND_CEILING);
            data.put(pos, new AtomicInteger(0));
            keys[x] = pos;
        }

        for (int x = 0; x < sourceArray.length(); x++) {
            double d = sourceArray.getDouble(x);
            int bin = (int) ((d - min) / binSize);

            if (bin < 0) {
                bins.putScalar(0, bins.getDouble(0) + 1);
                data.get(keys[0]).incrementAndGet();
            } else if (bin >= numberOfBins) {
                bins.putScalar(numberOfBins - 1, bins.getDouble(numberOfBins-1) + 1);
                data.get(keys[numberOfBins-1]).incrementAndGet();
              } else {
                bins.putScalar(bin, bins.getDouble(bin) + 1);
                data.get(keys[bin]).incrementAndGet();
            }
        }
    }

    public static class Builder {
        private INDArray source;
        private int binCount;
        private int rounds = 2;

        /**
         * Build Histogram Builder instance for specified array
         * @param array
         */
        public Builder(INDArray array) {
            this.source = array;
        }

        /**
         * Sets number of numbers behind decimal part
         *
         * @param rounds
         * @return
         */
        public Builder setRounding(int rounds) {
            this.rounds = rounds;
            return this;
        }

        /**
         * Specifies number of bins for output histogram
         *
         * @param bins
         * @return
         */
        public Builder setBinCount(int bins) {
            this.binCount = bins;
            return this;
        }

        /**
         * Returns ready-to-use Histogram instance
         * @return
         */
        public HistogramBin build() {
            HistogramBin histogram = new HistogramBin();
            histogram.sourceArray = this.source;
            histogram.numberOfBins = this.binCount;
            histogram.rounds = this.rounds;

            histogram.calcHistogram();

            return histogram;
        }
    }
}
