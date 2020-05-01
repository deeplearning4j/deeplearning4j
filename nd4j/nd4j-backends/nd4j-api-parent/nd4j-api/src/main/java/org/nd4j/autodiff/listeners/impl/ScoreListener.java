package org.nd4j.autodiff.listeners.impl;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.ListenerResponse;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.text.DecimalFormat;

/**
 * A listener that reports scores and performance metrics for each epoch.<br>
 * At every N iterations, the following is reported:
 * (a) Epoch and iteration number<br>
 * (b) Loss value (total loss)<br>
 * (c) ETL time (if > 0) - this represents how long training was blocked waiting for data. Values consistently above 0 indicate an ETL bottleneck<br>
 * <br><br>
 * At the end of every epoch, the following is reported:<br>
 * (a) Epoch and iteration numbers<br>
 * (b) Number of batches and examples in the epoch<br>
 * (c) Average number of batches per second and examples per second<br>
 * (d) Total amount of time blocked on ETL during the epoch (including percentage, if > 0)<br>
 *
 * @author Alex Black
 */
@Slf4j
public class ScoreListener extends BaseListener {

    private final int frequency;
    private final boolean reportEpochs;
    private final boolean reportIterPerformance;

    private long epochExampleCount;
    private int epochBatchCount;
    private long etlTotalTimeEpoch;

    private long lastIterTime;
    private long etlTimeSumSinceLastReport;
    private long iterTimeSumSinceLastReport;
    private int examplesSinceLastReportIter;
    private long lastReportTime = -1;

    /**
     * Create a ScoreListener reporting every 10 iterations, and at the end of each epoch
     */
    public ScoreListener() {
        this(10, true);
    }

    /**
     * Create a ScoreListener reporting every N iterations, and at the end of each epoch
     */
    public ScoreListener(int frequency) {
        this(frequency, true);
    }

    /**
     * Create a ScoreListener reporting every N iterations, and optionally at the end of each epoch
     */
    public ScoreListener(int frequency, boolean reportEpochs) {
        this(frequency, reportEpochs, true);
    }

    public ScoreListener(int frequency, boolean reportEpochs, boolean reportIterPerformance) {
        Preconditions.checkArgument(frequency > 0, "ScoreListener frequency must be > 0, got %s", frequency);
        this.frequency = frequency;
        this.reportEpochs = reportEpochs;
        this.reportIterPerformance = reportIterPerformance;
    }


    @Override
    public boolean isActive(Operation operation) {
        return operation == Operation.TRAINING;
    }

    @Override
    public void epochStart(SameDiff sd, At at) {
        if (reportEpochs) {
            epochExampleCount = 0;
            epochBatchCount = 0;
            etlTotalTimeEpoch = 0;
        }
        lastReportTime = -1;
        examplesSinceLastReportIter = 0;
    }

    @Override
    public ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis) {
        if (reportEpochs) {
            double batchesPerSec = epochBatchCount / (epochTimeMillis / 1000.0);
            double examplesPerSec = epochExampleCount / (epochTimeMillis / 1000.0);
            double pcEtl = 100.0 * etlTotalTimeEpoch / (double) epochTimeMillis;
            String etl = formatDurationMs(etlTotalTimeEpoch) + " ETL time" + (etlTotalTimeEpoch > 0 ? "(" + format2dp(pcEtl) + " %)" : "");
            log.info("Epoch {} complete on iteration {} - {} batches ({} examples) in {} - {} batches/sec, {} examples/sec, {}",
                    at.epoch(), at.iteration(), epochBatchCount, epochExampleCount, formatDurationMs(epochTimeMillis),
                    format2dp(batchesPerSec), format2dp(examplesPerSec), etl);
        }

        return ListenerResponse.CONTINUE;
    }

    @Override
    public void iterationStart(SameDiff sd, At at, MultiDataSet data, long etlMs) {
        lastIterTime = System.currentTimeMillis();
        etlTimeSumSinceLastReport += etlMs;
        etlTotalTimeEpoch += etlMs;
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        iterTimeSumSinceLastReport += System.currentTimeMillis() - lastIterTime;
        epochBatchCount++;
        if (dataSet.numFeatureArrays() > 0 && dataSet.getFeatures(0) != null) {
            int n = (int) dataSet.getFeatures(0).size(0);
            examplesSinceLastReportIter += n;
            epochExampleCount += n;
        }

        if (at.iteration() > 0 && at.iteration() % frequency == 0) {
            double l = loss.totalLoss();
            String etl = "";
            if (etlTimeSumSinceLastReport > 0) {
                etl = "(" + formatDurationMs(etlTimeSumSinceLastReport) + " ETL";
                if (frequency == 1) {
                    etl += ")";
                } else {
                    etl += " in " + frequency + " iter)";
                }
            }

            if(!reportIterPerformance) {
                log.info("Loss at epoch {}, iteration {}: {}{}", at.epoch(), at.iteration(), format5dp(l), etl);
            } else {
                long time = System.currentTimeMillis();
                if(lastReportTime > 0){
                    double batchPerSec = 1000 * frequency / (double)(time - lastReportTime);
                    double exPerSec = 1000 * examplesSinceLastReportIter / (double)(time - lastReportTime);
                    log.info("Loss at epoch {}, iteration {}: {}{}, batches/sec: {}, examples/sec: {}", at.epoch(), at.iteration(), format5dp(l),
                            etl, format5dp(batchPerSec), format5dp(exPerSec));
                } else {
                    log.info("Loss at epoch {}, iteration {}: {}{}", at.epoch(), at.iteration(), format5dp(l), etl);
                }

                lastReportTime = time;
            }

            iterTimeSumSinceLastReport = 0;
            etlTimeSumSinceLastReport = 0;
            examplesSinceLastReportIter = 0;
        }
    }

    protected String formatDurationMs(long ms) {
        if (ms <= 100) {
            return ms + " ms";
        } else if (ms <= 60000L) {
            double sec = ms / 1000.0;
            return format2dp(sec) + " sec";
        } else if (ms <= 60 * 60000L) {
            double min = ms / 60_000.0;
            return format2dp(min) + " min";
        } else {
            double hr = ms / 360_000.0;
            return format2dp(hr) + " hr";
        }
    }

    protected static final ThreadLocal<DecimalFormat> DF_2DP = new ThreadLocal<>();
    protected static final ThreadLocal<DecimalFormat> DF_2DP_SCI = new ThreadLocal<>();

    protected String format2dp(double d) {
        if (d < 0.01) {
            DecimalFormat f = DF_2DP_SCI.get();
            if (f == null) {
                f = new DecimalFormat("0.00E0");
                DF_2DP.set(f);
            }
            return f.format(d);
        } else {
            DecimalFormat f = DF_2DP.get();
            if (f == null) {
                f = new DecimalFormat("#.00");
                DF_2DP.set(f);
            }
            return f.format(d);
        }
    }

    protected static final ThreadLocal<DecimalFormat> DF_5DP = new ThreadLocal<>();
    protected static final ThreadLocal<DecimalFormat> DF_5DP_SCI = new ThreadLocal<>();

    protected String format5dp(double d) {

        if (d < 1e-4 || d > 1e4) {
            //Use scientific
            DecimalFormat f = DF_5DP_SCI.get();
            if (f == null) {
                f = new DecimalFormat("0.00000E0");
                DF_5DP_SCI.set(f);
            }
            return f.format(d);
        } else {
            DecimalFormat f = DF_5DP.get();
            if (f == null) {
                f = new DecimalFormat("0.00000");
                DF_5DP.set(f);
            }
            return f.format(d);
        }
    }
}
