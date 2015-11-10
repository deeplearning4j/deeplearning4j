package org.deeplearning4j.graph.models.DeepWalk;

import org.deeplearning4j.graph.api.VertexSequence;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.models.GraphVectors;

/**
 * Created by Alex on 10/11/2015.
 */
public class DeepWalk<V,E> implements GraphVectors<V,E> {



    private int vectorSize;
    private int windowSize;
    private int batchSize;
    private long seed;
    private double learningRate;


    public DeepWalk(){

    }

    public int getVectorSize(){
        return vectorSize;
    }

    public int getWindowSize(){
        return windowSize;
    }

    public int getBatchSize(){
        return batchSize;
    }

    public double getLearningRate(){
        return learningRate;
    }

    public void fit(GraphWalkIterator<V> iterator){

        int walkLength = iterator.walkLength();

        while(iterator.hasNext()){
            VertexSequence<V> sequence = iterator.next();

            //Skipgram model:
            int[] walk = new int[walkLength+1];
            int i=0;
            while(iterator.hasNext()) walk[i++] = sequence.next().vertexID();

        }

    }

    public static class Builder<V,E> {

        private int vectorSize;
        private int batchSize;
        private long seed = Long.MAX_VALUE;
        private double learningRate = 0.01;
        private int windowSize;

        public Builder<V,E> vectorSize(int size){
            this.vectorSize = vectorSize;
            return this;
        }

        public Builder<V,E> batchSize(int batchSize){
            this.batchSize = batchSize;
            return this;
        }

        public Builder<V,E> seed(long seed){
            this.seed = seed;
            return this;
        }

        public Builder<V,E> learningRate(double learningRate){
            this.learningRate = learningRate;
            return this;
        }

        public Builder<V,E> windowSize(int windowSize){
            this.windowSize = windowSize;
            return this;
        }


        public DeepWalk<V,E> build(){

            DeepWalk<V,E> dw = new DeepWalk<>();
            dw.vectorSize = vectorSize;
            dw.batchSize = batchSize;
            dw.seed = seed;
            dw.learningRate = learningRate;

            return dw;
        }
    }

}
