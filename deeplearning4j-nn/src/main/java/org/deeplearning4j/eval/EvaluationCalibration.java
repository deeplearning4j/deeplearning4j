package org.deeplearning4j.eval;

import lombok.Getter;
import org.deeplearning4j.eval.curves.ReliabilityDiagram;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Tools for classifier calibration analysis:
 * - Residual plot
 * - Reliability diagram
 *
 * @author Alex Black
 */
@Getter
public class EvaluationCalibration extends BaseEvaluation<EvaluationCalibration> {

    public static final int DEFAULT_RELIABILITY_DIAG_NUM_BINS = 10;

    private final int numBins;
    private final boolean excludeEmptyBins;

    private INDArray rDiagBinPosCount;
    private INDArray rDiagBinTotalCount;
    private INDArray rDiagBinSumPredictions;

    public EvaluationCalibration(){
        this(DEFAULT_RELIABILITY_DIAG_NUM_BINS, true);
    }

    public EvaluationCalibration(int numBins){
        this(numBins, true);
    }

    public EvaluationCalibration(@JsonProperty("numBins") int numBins, @JsonProperty("excludeEmptyBins") boolean excludeEmptyBins){
        this.numBins = numBins;
        this.excludeEmptyBins = excludeEmptyBins;
    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray) {

        if (labels.rank() == 3) {
            evalTimeSeries(labels, networkPredictions, maskArray);
            return;
        }

        //Stats for the reliability diagram: one reliability diagram for each class
        // For each bin, we need: (a) the number of positive cases AND total cases, (b) the average probability

        int nClasses = labels.size(1);

        if(rDiagBinPosCount == null){
            rDiagBinPosCount = Nd4j.create(numBins, nClasses);
            rDiagBinTotalCount = Nd4j.create(numBins, nClasses);
            rDiagBinSumPredictions = Nd4j.create(numBins, nClasses);
        }


        //First: loop over classes, determine positive count and total count - for each bin
        double binSize = 1.0 / numBins;

        INDArray p = networkPredictions;
        INDArray l = labels;

        if(maskArray != null){
            //2 options: per-output masking, or
            if(maskArray.isColumnVector()){
                //Per-example masking
                l = l.mulColumnVector(maskArray);
            } else {
                l = l.mul(maskArray);
            }
        }

        for(int j = 0; j< numBins; j++ ){
            INDArray geqBinLower = p.gte(j*binSize);
            INDArray ltBinUpper;
            if(j == numBins-1){
                //Handle edge case
                ltBinUpper = p.lte(1.0);
            } else {
                ltBinUpper = p.lt((j+1)*binSize);
            }

            INDArray isMax = Nd4j.getExecutioner().execAndReturn(new IsMax(p.dup(), 1));

            //Calculate bit-mask over each entry - whether that entry is in the current bin or not
            INDArray currBinBitMask = geqBinLower.muli(ltBinUpper); //.muli(isMax);
            if(maskArray != null){
                if(maskArray.isColumnVector()){
                    currBinBitMask.muliColumnVector(maskArray);
                } else {
                    currBinBitMask.muli(maskArray);
                }
            }

            INDArray isPosLabelForBin = l.mul(currBinBitMask);
            INDArray maskedProbs = networkPredictions.mul(currBinBitMask);

            INDArray numPredictionsCurrBin = currBinBitMask.sum(0);

            rDiagBinSumPredictions.getRow(j).addi(maskedProbs.sum(0));
            rDiagBinPosCount.getRow(j).addi(isPosLabelForBin.sum(0));
            rDiagBinTotalCount.getRow(j).addi(numPredictionsCurrBin);
        }

    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray)null);
    }

    @Override
    public void merge(EvaluationCalibration other) {
        if(numBins != other.numBins){
            throw new UnsupportedOperationException("Cannot merge EvaluationCalibration instances with different numbers of bins");
        }

        if(other.rDiagBinPosCount == null){
            return;
        }

        if(rDiagBinPosCount == null){
            this.rDiagBinPosCount = other.rDiagBinPosCount;
            this.rDiagBinTotalCount = other.rDiagBinTotalCount;
            this.rDiagBinSumPredictions = other.rDiagBinSumPredictions;
        }

        this.rDiagBinPosCount.addi(other.rDiagBinPosCount);
        this.rDiagBinTotalCount.addi(other.rDiagBinTotalCount);
        this.rDiagBinSumPredictions.addi(other.rDiagBinSumPredictions);
    }

    @Override
    public void reset() {
        rDiagBinPosCount = null;
        rDiagBinTotalCount = null;
        rDiagBinSumPredictions = null;
    }

    @Override
    public String stats() {
        return "EvaluationCalibration(nBins=" + numBins + ")";
    }

    public ReliabilityDiagram getReliabilityDiagram(int classNum){

        INDArray totalCountBins = rDiagBinTotalCount.getColumn(classNum);
        INDArray countPositiveBins = rDiagBinPosCount.getColumn(classNum);

        double[] meanPredictionBins = rDiagBinSumPredictions.getColumn(classNum)
                .div(totalCountBins).data().asDouble();

        double[] fracPositives = countPositiveBins.div(totalCountBins).data().asDouble();

        if(excludeEmptyBins){
            MatchCondition condition = new MatchCondition(totalCountBins, Conditions.equals(0));
            int numZeroBins = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);
            if(numZeroBins != 0){
                double[] mpb = meanPredictionBins;
                double[] fp = fracPositives;

                meanPredictionBins = new double[totalCountBins.length() - numZeroBins];
                fracPositives = new double[meanPredictionBins.length];
                int j=0;
                for( int i=0; i<mpb.length; i++ ){
                    if(totalCountBins.getDouble(i) != 0){
                        meanPredictionBins[j] = mpb[i];
                        fracPositives[j] = fp[i];
                        j++;
                    }
                }
            }
        }

        return new ReliabilityDiagram(meanPredictionBins, fracPositives);
    }
}
