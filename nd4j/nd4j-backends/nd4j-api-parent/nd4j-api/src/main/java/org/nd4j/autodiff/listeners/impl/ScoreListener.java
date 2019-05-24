package org.nd4j.autodiff.listeners.impl;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.text.DecimalFormat;

@Slf4j
public class ScoreListener extends BaseListener {

    private final int frequency;
    private final boolean reportEpochs;

    private long epochStart;
    private long epochExampleCount;
    private int epochBatchCount;

    private long lastIterTime;
    private long iterTimeSumSinceLastReport;
    private int examplesSinceLastReportIter;

    public ScoreListener(){
        this(10, true);
    }

    public ScoreListener(int frequency){
        this(frequency, true);
    }

    public ScoreListener(int frequency, boolean reportEpochs){
        Preconditions.checkArgument(frequency > 0, "ScoreListener frequency must be > 0, got %s", frequency);
        this.frequency = frequency;
        this.reportEpochs = reportEpochs;
    }


    @Override
    public void epochStart(SameDiff sd, At at) {
        if(reportEpochs) {
            epochStart = System.currentTimeMillis();
            epochExampleCount = 0;
            epochBatchCount = 0;
        }
    }

    @Override
    public void epochEnd(SameDiff sd, At at) {
        if(reportEpochs){
            long epochDuration = System.currentTimeMillis() - epochStart;
            double batchesPerSec = epochBatchCount / (epochDuration / 1000.0);
            double examplesPerSec = epochExampleCount / (epochDuration / 1000.0);
            log.info("Epoch {} complete on iteration {} - {} batches ({} examples) in {} - {} batches/sec, {} examples/sec",
                    at.epoch(), at.iteration(), epochBatchCount, epochExampleCount, formatDurationMs(epochDuration), format2dp(batchesPerSec), format2dp(examplesPerSec));
        }
    }

    @Override
    public void iterationStart(SameDiff sd, At at, MultiDataSet data) {
        lastIterTime = System.currentTimeMillis();
    }

    @Override
    public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
        iterTimeSumSinceLastReport += System.currentTimeMillis() - lastIterTime;
        if(dataSet.numFeatureArrays() > 0 && dataSet.getFeatures(0) != null){
            examplesSinceLastReportIter += dataSet.getFeatures(0).size(0);
        }

        if(at.iteration() > 0 && at.iteration() % frequency == 0){
            double l = loss.totalLoss();
            log.info("Loss at epoch {}, iteration {}: {}", at.epoch(), at.iteration(), format5dp(l));
            iterTimeSumSinceLastReport = 0;
        }
    }

    protected String formatDurationMs(long ms){
        if(ms <= 100){
            return ms + " ms";
        } else if(ms <= 60000L){
            double sec = ms / 1000.0;
            return format2dp(sec) + " sec";
        } else if(ms <= 60*60000L){
            double min = ms / 60_000.0;
            return format2dp(min) + " min";
        } else {
            double hr = ms / 360_000.0;
            return format2dp(hr) + " hr";
        }
    }

    protected static final ThreadLocal<DecimalFormat> DF_2DP = new ThreadLocal<>();
    protected static final ThreadLocal<DecimalFormat> DF_2DP_SCI = new ThreadLocal<>();
    protected String format2dp(double d){
        if(d < 0.01){
            DecimalFormat f = DF_2DP_SCI.get();
            if(f == null) {
                f = new DecimalFormat("0.00E0");
                DF_2DP.set(f);
            }
            return f.format(d);
        } else {
            DecimalFormat f = DF_2DP.get();
            if(f == null){
                f = new DecimalFormat("#.00");
                DF_2DP.set(f);
            }
            return f.format(d);
        }
    }

    protected static final ThreadLocal<DecimalFormat> DF_4DP = new ThreadLocal<>();
    protected static final ThreadLocal<DecimalFormat> DF_4DP_SCI = new ThreadLocal<>();
    protected String format5dp(double d){

        if (d < 1e-4 || d > 1e4) {
            //Use scientific
            DecimalFormat f = DF_4DP_SCI.get();
            if (f == null) {
                f = new DecimalFormat("0.0000E0");
                DF_4DP_SCI.set(f);
            }
            return f.format(d);
        } else {
            DecimalFormat f = DF_4DP.get();
            if (f == null) {
                f = new DecimalFormat("#.0000");
                DF_4DP.set(f);
            }
            return f.format(d);
        }
    }
}
