/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.spark.models.embeddings.glove;

import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class GloveChange implements Serializable {
    private VocabWord w1, w2;
    private INDArray w1Update, w2Update;
    private double w1BiasUpdate, w2BiasUpdate;
    private double error;
    private INDArray w1History, w2History;
    private double w1BiasHistory, w2BiasHistory;

    public GloveChange(VocabWord w1, VocabWord w2, INDArray w1Update, INDArray w2Update, double w1BiasUpdate,
                    double w2BiasUpdate, double error, INDArray w1History, INDArray w2History, double w1BiasHistory,
                    double w2BiasHistory) {
        this.w1 = w1;
        this.w2 = w2;
        this.w1Update = w1Update;
        this.w2Update = w2Update;
        this.w1BiasUpdate = w1BiasUpdate;
        this.w2BiasUpdate = w2BiasUpdate;
        this.error = error;
        this.w1History = w1History;
        this.w2History = w2History;
        this.w1BiasHistory = w1BiasHistory;
        this.w2BiasHistory = w2BiasHistory;
    }

    /**
     * Apply the changes to the table
     * @param table
     */
    public void apply(GloveWeightLookupTable table) {
        table.getBias().putScalar(w1.getIndex(), table.getBias().getDouble(w1.getIndex()) - w1BiasUpdate);
        table.getBias().putScalar(w2.getIndex(), table.getBias().getDouble(w2.getIndex()) - w2BiasUpdate);
        table.getSyn0().slice(w1.getIndex()).subi(w1Update);
        table.getSyn0().slice(w2.getIndex()).subi(w2Update);
        table.getWeightAdaGrad().getHistoricalGradient().slice(w1.getIndex()).addi(w1History);
        table.getWeightAdaGrad().getHistoricalGradient().slice(w2.getIndex()).addi(w2History);
        table.getBiasAdaGrad().getHistoricalGradient().putScalar(w1.getIndex(),
                        table.getBiasAdaGrad().getHistoricalGradient().getDouble(w1.getIndex()) + w1BiasHistory);
        table.getBiasAdaGrad().getHistoricalGradient().putScalar(w2.getIndex(),
                        table.getBiasAdaGrad().getHistoricalGradient().getDouble(w2.getIndex()) + w1BiasHistory);

    }

    public INDArray getW1History() {
        return w1History;
    }

    public void setW1History(INDArray w1History) {
        this.w1History = w1History;
    }

    public INDArray getW2History() {
        return w2History;
    }

    public void setW2History(INDArray w2History) {
        this.w2History = w2History;
    }

    public double getW1BiasHistory() {
        return w1BiasHistory;
    }

    public void setW1BiasHistory(double w1BiasHistory) {
        this.w1BiasHistory = w1BiasHistory;
    }

    public double getW2BiasHistory() {
        return w2BiasHistory;
    }

    public void setW2BiasHistory(double w2BiasHistory) {
        this.w2BiasHistory = w2BiasHistory;
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

    @Override
    public String toString() {
        return w1.getIndex() + "," + w2.getIndex() + " error " + error;
    }

}
