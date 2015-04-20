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
    private VocabWord w1,w2;
    private INDArray w1Update,w2Update;
    private double w1BiasUpdate,w2BiasUpdate;
    private double error;

    public GloveChange(VocabWord w1, VocabWord w2, INDArray w1Update, INDArray w2Update, double w1BiasUpdate, double w2BiasUpdate,double error) {
        this.w1 = w1;
        this.w2 = w2;
        this.w1Update = w1Update;
        this.w2Update = w2Update;
        this.w1BiasUpdate = w1BiasUpdate;
        this.w2BiasUpdate = w2BiasUpdate;
        this.error = error;
    }

    /**
     * Apply the changes to the table
     * @param table
     */
    public void apply(GloveWeightLookupTable table) {
        table.getBias().putScalar(w1.getIndex(), table.getBias().getDouble(w1.getIndex()) - w1BiasUpdate);
        table.getBias().putScalar(w2.getIndex(),table.getBias().getDouble(w2.getIndex()) - w2BiasUpdate);
        table.getSyn0().slice(w1.getIndex()).subi(w1Update);
        table.getSyn0().slice(w2.getIndex()).sub(w2Update);
    }

    public VocabWord getW1() {
        return w1;
    }

    public void setW1(VocabWord w1) {
        this.w1 = w1;
    }

    public VocabWord getW2() {
        return w2;
    }

    public void setW2(VocabWord w2) {
        this.w2 = w2;
    }

    public INDArray getW1Update() {
        return w1Update;
    }

    public void setW1Update(INDArray w1Update) {
        this.w1Update = w1Update;
    }

    public INDArray getW2Update() {
        return w2Update;
    }

    public void setW2Update(INDArray w2Update) {
        this.w2Update = w2Update;
    }

    public double getW1BiasUpdate() {
        return w1BiasUpdate;
    }

    public void setW1BiasUpdate(double w1BiasUpdate) {
        this.w1BiasUpdate = w1BiasUpdate;
    }

    public double getW2BiasUpdate() {
        return w2BiasUpdate;
    }

    public void setW2BiasUpdate(double w2BiasUpdate) {
        this.w2BiasUpdate = w2BiasUpdate;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }
}
