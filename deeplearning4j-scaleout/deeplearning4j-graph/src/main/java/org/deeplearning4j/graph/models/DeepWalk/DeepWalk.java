package org.deeplearning4j.graph.models.deepwalk;

import org.deeplearning4j.graph.api.Graph;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.api.VertexSequence;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.models.BinaryTree;
import org.deeplearning4j.graph.models.GraphVectors;
import org.deeplearning4j.graph.models.embeddings.GraphVectorLookupTable;
import org.deeplearning4j.graph.models.embeddings.InMemoryGraphLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Created by Alex on 10/11/2015.
 */
public class DeepWalk<V,E> implements GraphVectors<V,E> {
    private int vectorSize;
    private int windowSize;
    private int batchSize;
    private long seed;
    private double learningRate;
    private boolean initCalled = false;
    private BinaryTree tree;
    private GraphVectorLookupTable lookupTable;



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

    public void initialize(Graph<V,E> graph){
        int nVertices = graph.numVertices();
        int[] degrees = new int[nVertices];
        for( int i=0; i<nVertices; i++ ) degrees[i] = graph.getVertexDegree(i);
        initialize(degrees);
    }

    public void initialize(int[] graphVertexDegrees){
        GraphHuffman gh = new GraphHuffman(graphVertexDegrees.length);
        gh.buildTree(graphVertexDegrees);
        tree = gh;
        lookupTable = new InMemoryGraphLookupTable(graphVertexDegrees.length,vectorSize,gh,learningRate);
        initCalled = true;
    }

    public void fit(GraphWalkIterator<V> iterator){
        if(!initCalled) throw new UnsupportedOperationException("DeepWalk not initialized (call initialize before fit)");
        int walkLength = iterator.walkLength();

        while(iterator.hasNext()){
            VertexSequence<V> sequence = iterator.next();

            //Skipgram model:
            int[] walk = new int[walkLength+1];
            int i=0;
            while(sequence.hasNext()) walk[i++] = sequence.next().vertexID();

            skipGram(walk);
        }
    }

    private void skipGram(int[] walk){

        for(int mid = windowSize; mid < walk.length-windowSize; mid++ ){

            for(int pos=0; pos<2*windowSize; pos++){
                if(pos == mid) continue;

                //pair of vertices: walk[mid] -> walk[pos]
//                doIteration(walk[mid],walk[pos]);
                lookupTable.iterate(walk[mid],walk[pos]);
            }
        }
    }

//    private void doIteration(int vertexIn, int vertexOut){
//        lookupTable.iterate(vertexIn,vertexOut);
//    }


    @Override
    public Graph<V, E> getGraph() {
        return null;
    }

    @Override
    public int numVertices() {
        return 0;
    }

    @Override
    public INDArray getVertexVector(Vertex<V> vertex) {
        return null;
    }

    @Override
    public INDArray getVertexVector(int vertexIdx) {
        return lookupTable.getVector(vertexIdx);
    }

    @Override
    public Collection<Vertex<V>> verticesNearest(Vertex<V> vertex, int top) {
        return null;
    }

    @Override
    public double similarity(Vertex<V> vertex1, Vertex<V> vertex2) {
        return 0;
    }

    @Override
    public double similarity(int vertexIdx1, int vertexIdx2) {
        return 0;
    }

    public static class Builder<V,E> {

        private int vectorSize = 100;
        private int batchSize;
        private long seed = 12345;
        private double learningRate = 0.01;
        private int windowSize = 2;

        public Builder<V,E> vectorSize(int vectorSize){
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
            dw.windowSize = windowSize;
            dw.batchSize = batchSize;
            dw.seed = seed;
            dw.learningRate = learningRate;

            return dw;
        }
    }

}
