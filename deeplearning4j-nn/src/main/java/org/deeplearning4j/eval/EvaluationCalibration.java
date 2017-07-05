package org.deeplearning4j.eval;

import lombok.Getter;
import org.deeplearning4j.eval.curves.ReliabilityDiagram;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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

    private INDArray rDiagBinPosCount;
    private INDArray rDiagBinTotalCount;
    private INDArray rDiagBinSumPredictions;

    public EvaluationCalibration(){
        this(DEFAULT_RELIABILITY_DIAG_NUM_BINS);
    }

    public EvaluationCalibration(@JsonProperty("numBins") int numBins){
        this.numBins = numBins;
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
                l = l.mul(maskArray.getColumn(i));
            }
        }

        for(int j = 0; j< numBins; j++ ){
            INDArray geqBinLower = p.gte(j*binSize);
            INDArray ltBinUpper = p.lt((j+1)*binSize);

            //Calculate bit-mask over each entry - whether that entry is in the current bin or not
            INDArray currBinBitMask = geqBinLower.muli(ltBinUpper);
            if(maskArray != null){
                if(maskArray.isColumnVector()){
                    currBinBitMask.muliColumnVector(maskArray);
                } else {
                    currBinBitMask.muli(maskArray);
                }
            }

            INDArray isPosLabelForBin = labels.mul(currBinBitMask);
            INDArray maskedProbs = networkPredictions.mul(currBinBitMask);

            INDArray numPredictionsCurrBin = currBinBitMask.sum(0);

            rDiagBinSumPredictions.getRow(j).addi(maskedProbs.sum(0));
            rDiagBinPosCount.getRow(j).addi(isPosLabelForBin);
            rDiagBinTotalCount.getRow(j).addi(numPredictionsCurrBin);
        }

    }

    @Override
    public void eval(INDArray labels, INDArray networkPredictions) {
        eval(labels, networkPredictions, (INDArray)null);
    }

    @Override
    public void merge(EvaluationCalibration other) {

    }

    @Override
    public void reset() {

    }

    @Override
    public String stats() {
        return null;
    }

    public ReliabilityDiagram getReliabilityDiagram(int classNum){

        INDArray totalCountBins = rDiagBinTotalCount.getRow(classNum);
        INDArray countPositiveBins = rDiagBinPosCount.getRow(classNum);

        double[] meanPredictionBins = rDiagBinSumPredictions.getRow(classNum)
                .div(totalCountBins).data().asDouble();

        double[] fracPositives = countPositiveBins.div(totalCountBins).data().asDouble();

        return new ReliabilityDiagram(meanPredictionBins, fracPositives);
    }
}
