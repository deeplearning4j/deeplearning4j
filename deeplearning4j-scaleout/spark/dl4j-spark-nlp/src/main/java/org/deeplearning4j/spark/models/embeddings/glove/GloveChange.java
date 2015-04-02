package org.deeplearning4j.spark.models.embeddings.glove;

import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Adam Gibson
 */
public class GloveChange implements Serializable {
    private Map<Integer,INDArray> changed = new HashMap<>();
    private Map<Integer,INDArray> weightAdaGrad = new HashMap<>();
    private Map<Integer,Double> biasChange = new HashMap<>();
    private Map<Integer,Double> biasAdaGradChange = new HashMap<>();
    private VocabWord w1,w2;



    public GloveChange(VocabWord w1, VocabWord w2,GloveWeightLookupTable table) {
        this.w1 = w1;
        this.w2 = w2;
        changed.put(w1.getIndex(),table.getSyn0().slice(w1.getIndex()));
        changed.put(w2.getIndex(),table.getSyn0().slice(w2.getIndex()));
        biasChange.put(w1.getIndex(),table.getBias().getDouble(w1.getIndex()));
        biasChange.put(w2.getIndex(),table.getBias().getDouble(w2.getIndex()));
        biasAdaGradChange.put(w1.getIndex(),table.getBiasAdaGrad().getHistoricalGradient().getDouble(w1.getIndex()));
        biasAdaGradChange.put(w2.getIndex(),table.getBiasAdaGrad().getHistoricalGradient().getDouble(w2.getIndex()));
        weightAdaGrad.put(w1.getIndex(),table.getWeightAdaGrad().getHistoricalGradient().slice(w1.getIndex()));
        weightAdaGrad.put(w2.getIndex(),table.getWeightAdaGrad().getHistoricalGradient().slice(w2.getIndex()));

    }


    private double getDelta(double sub,double delta) {
        return sub - (sub + delta);
    }

    private void addDelta(INDArray add,INDArray update) {
        add.addi(update.sub(add));
    }
    /**
     * Apply the changes to the table
     * @param table
     */
    public void apply(GloveWeightLookupTable table) {
        //only apply the delta (not the original weights)
        for(VocabWord word : new VocabWord[]{w1,w2}) {
            table.getBias().putScalar(word.getIndex(),getDelta(table.getBias().getDouble(word.getIndex()),biasChange.get(word.getIndex())));
            table.getBiasAdaGrad().getHistoricalGradient().putScalar(word.getIndex(),getDelta(table.getBiasAdaGrad().getHistoricalGradient().getDouble(word.getIndex()), biasAdaGradChange.get(word.getIndex())));
            addDelta(table.getWeightAdaGrad().getHistoricalGradient().slice(word.getIndex()), changed.get(word.getIndex()));
            addDelta( table.getSyn0().slice(word.getIndex()),changed.get(word.getIndex()));
        }
    }




}
