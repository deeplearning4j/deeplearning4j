package org.datavec.api.transform.analysis;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.columns.*;
import org.datavec.api.transform.analysis.counter.*;
import org.datavec.api.transform.analysis.histogram.HistogramCounter;

import java.util.ArrayList;
import java.util.List;

public class DataVecAnalysisUtils {

    private DataVecAnalysisUtils(){ }


    public static void mergeCounters(List<ColumnAnalysis> columnAnalysis, List<HistogramCounter> histogramCounters){
        if(histogramCounters == null)
            return;

        //Merge analysis values and histogram values
        for (int i = 0; i < columnAnalysis.size(); i++) {
            HistogramCounter hc = histogramCounters.get(i);
            ColumnAnalysis ca = columnAnalysis.get(i);
            if (ca instanceof IntegerAnalysis) {
                ((IntegerAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((IntegerAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof DoubleAnalysis) {
                ((DoubleAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((DoubleAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof LongAnalysis) {
                ((LongAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((LongAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof TimeAnalysis) {
                ((TimeAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((TimeAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof StringAnalysis) {
                ((StringAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((StringAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            } else if (ca instanceof NDArrayAnalysis) {
                ((NDArrayAnalysis) ca).setHistogramBuckets(hc.getBins());
                ((NDArrayAnalysis) ca).setHistogramBucketCounts(hc.getCounts());
            }
        }
    }


    public static List<ColumnAnalysis> convertCounters(List<AnalysisCounter> counters, double[][] minsMaxes, List<ColumnType> columnTypes){
        int nColumns = columnTypes.size();

        List<ColumnAnalysis> list = new ArrayList<>();

        for (int i = 0; i < nColumns; i++) {
            ColumnType ct = columnTypes.get(i);

            switch (ct) {
                case String:
                    StringAnalysisCounter sac = (StringAnalysisCounter) counters.get(i);
                    list.add(new StringAnalysis.Builder().countTotal(sac.getCountTotal())
                            .minLength(sac.getMinLengthSeen()).maxLength(sac.getMaxLengthSeen())
                            .meanLength(sac.getMean()).sampleStdevLength(sac.getSampleStdev())
                            .sampleVarianceLength(sac.getSampleVariance()).build());
                    minsMaxes[i][0] = sac.getMinLengthSeen();
                    minsMaxes[i][1] = sac.getMaxLengthSeen();
                    break;
                case Integer:
                    IntegerAnalysisCounter iac = (IntegerAnalysisCounter) counters.get(i);
                    IntegerAnalysis ia = new IntegerAnalysis.Builder().min(iac.getMinValueSeen())
                            .max(iac.getMaxValueSeen()).mean(iac.getMean()).sampleStdev(iac.getSampleStdev())
                            .sampleVariance(iac.getSampleVariance()).countZero(iac.getCountZero())
                            .countNegative(iac.getCountNegative()).countPositive(iac.getCountPositive())
                            .countMinValue(iac.getCountMinValue()).countMaxValue(iac.getCountMaxValue())
                            .countTotal(iac.getCountTotal()).digest(iac.getDigest()).build();
                    list.add(ia);

                    minsMaxes[i][0] = iac.getMinValueSeen();
                    minsMaxes[i][1] = iac.getMaxValueSeen();

                    break;
                case Long:
                    LongAnalysisCounter lac = (LongAnalysisCounter) counters.get(i);

                    LongAnalysis la = new LongAnalysis.Builder().min(lac.getMinValueSeen()).max(lac.getMaxValueSeen())
                            .mean(lac.getMean()).sampleStdev(lac.getSampleStdev())
                            .sampleVariance(lac.getSampleVariance()).countZero(lac.getCountZero())
                            .countNegative(lac.getCountNegative()).countPositive(lac.getCountPositive())
                            .countMinValue(lac.getCountMinValue()).countMaxValue(lac.getCountMaxValue())
                            .countTotal(lac.getCountTotal()).digest(lac.getDigest()).build();

                    list.add(la);

                    minsMaxes[i][0] = lac.getMinValueSeen();
                    minsMaxes[i][1] = lac.getMaxValueSeen();

                    break;
                case Double:
                    DoubleAnalysisCounter dac = (DoubleAnalysisCounter) counters.get(i);
                    DoubleAnalysis da = new DoubleAnalysis.Builder().min(dac.getMinValueSeen())
                            .max(dac.getMaxValueSeen()).mean(dac.getMean()).sampleStdev(dac.getSampleStdev())
                            .sampleVariance(dac.getSampleVariance()).countZero(dac.getCountZero())
                            .countNegative(dac.getCountNegative()).countPositive(dac.getCountPositive())
                            .countMinValue(dac.getCountMinValue()).countMaxValue(dac.getCountMaxValue())
                            .countNaN(dac.getCountNaN()).digest(dac.getDigest()).countTotal(dac.getCountTotal()).build();
                    list.add(da);

                    minsMaxes[i][0] = dac.getMinValueSeen();
                    minsMaxes[i][1] = dac.getMaxValueSeen();

                    break;
                case Categorical:
                    CategoricalAnalysisCounter cac = (CategoricalAnalysisCounter) counters.get(i);
                    CategoricalAnalysis ca = new CategoricalAnalysis(cac.getCounts());
                    list.add(ca);

                    break;
                case Time:
                    LongAnalysisCounter lac2 = (LongAnalysisCounter) counters.get(i);

                    TimeAnalysis la2 = new TimeAnalysis.Builder().min(lac2.getMinValueSeen())
                            .max(lac2.getMaxValueSeen()).mean(lac2.getMean()).sampleStdev(lac2.getSampleStdev())
                            .sampleVariance(lac2.getSampleVariance()).countZero(lac2.getCountZero())
                            .countNegative(lac2.getCountNegative()).countPositive(lac2.getCountPositive())
                            .countMinValue(lac2.getCountMinValue()).countMaxValue(lac2.getCountMaxValue())
                            .countTotal(lac2.getCountTotal()).digest(lac2.getDigest()).build();

                    list.add(la2);

                    minsMaxes[i][0] = lac2.getMinValueSeen();
                    minsMaxes[i][1] = lac2.getMaxValueSeen();

                    break;
                case Bytes:
                    BytesAnalysisCounter bac = (BytesAnalysisCounter) counters.get(i);
                    list.add(new BytesAnalysis.Builder().countTotal(bac.getCountTotal()).build());
                    break;
                case NDArray:
                    NDArrayAnalysisCounter nac = (NDArrayAnalysisCounter) counters.get(i);
                    NDArrayAnalysis nda = nac.toAnalysisObject();
                    list.add(nda);

                    minsMaxes[i][0] = nda.getMinValue();
                    minsMaxes[i][1] = nda.getMaxValue();

                    break;
                default:
                    throw new IllegalStateException("Unknown column type: " + ct);
            }
        }

        return list;
    }

}
