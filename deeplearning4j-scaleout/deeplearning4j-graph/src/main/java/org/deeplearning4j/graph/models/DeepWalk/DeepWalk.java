package org.deeplearning4j.graph.models.deepwalk;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.api.IVertexSequence;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.models.BinaryTree;
import org.deeplearning4j.graph.models.GraphVectors;
import org.deeplearning4j.graph.models.embeddings.GraphVectorLookupTable;
import org.deeplearning4j.graph.models.embeddings.InMemoryGraphLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**Implementation of the DeepWalk graph vectorization model, based on the paper
 * <i>DeepWalk: Online Learning of Social Representations</i> by Perozzi, Al-Rfou & Skiena (2014),
 * <a href="http://arxiv.org/abs/1403.6652">http://arxiv.org/abs/1403.6652</a><br>
 * Similar to word2vec in nature, DeepWalk is an unsupervised learning algorithm that learns a vector representation
 * of each vertex in a graph. Vector representations are learned using walks (usually random walks) on the vertices in
 * the graph.<br>
 * Once learned, these vector representations can then be used for purposes such as classification, clustering, similarity
 * search, etc on the graph<br>
 * @author Alex Black
 */
public class DeepWalk<V,E> implements GraphVectors<V,E> {
    private int vectorSize;
    private int windowSize;
    private int batchSize;
    private double learningRate;
    private boolean initCalled = false;
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

    /** Initialize the DeepWalk model with a given graph. */
    public void initialize(IGraph<V,E> graph){
        int nVertices = graph.numVertices();
        int[] degrees = new int[nVertices];
        for( int i=0; i<nVertices; i++ ) degrees[i] = graph.getVertexDegree(i);
        initialize(degrees);
    }

    /** Initialize the DeepWalk model with a list of vertex degrees for a graph.<br>
     * Specifically, graphVertexDegrees[i] represents the vertex degree of the ith vertex<br>
     * vertex degrees are used to construct a binary (Huffman) tree, which is in turn used in
     * the hierarchical softmax implementation
     * @param graphVertexDegrees degrees of each vertex
     */
    public void initialize(int[] graphVertexDegrees){
        GraphHuffman gh = new GraphHuffman(graphVertexDegrees.length);
        gh.buildTree(graphVertexDegrees);
        lookupTable = new InMemoryGraphLookupTable(graphVertexDegrees.length,vectorSize,gh,learningRate);
        initCalled = true;
    }

    /**Fit the DeepWalk model using a GraphWalkIterator. Note that {@link #initialize(IGraph)} or {@link #initialize(int[])}
     * <em>must</em> be called first.
     * @param iterator iterator for graph walks
     */
    public void fit(GraphWalkIterator<V> iterator){
        if(!initCalled) throw new UnsupportedOperationException("DeepWalk not initialized (call initialize before fit)");
        int walkLength = iterator.walkLength();

        while(iterator.hasNext()){
            IVertexSequence<V> sequence = iterator.next();

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
                lookupTable.iterate(walk[mid],walk[pos]);
            }
        }
    }


    @Override
    public IGraph<V, E> getGraph() {
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

    public GraphVectorLookupTable lookupTable(){
        return lookupTable;
    }


    public static class Builder<V,E> {

        private int vectorSize = 100;
        private int batchSize;
        private long seed = 12345;
        private double learningRate = 0.01;
        private int windowSize = 2;

        /** Sets the size of the vectors to be learned for each vertex in the graph */
        public Builder<V,E> vectorSize(int vectorSize){
            this.vectorSize = vectorSize;
            return this;
        }

        public Builder<V,E> batchSize(int batchSize){
            this.batchSize = batchSize;
            return this;
        }

        /** Set the learning rate */
        public Builder<V,E> learningRate(double learningRate){
            this.learningRate = learningRate;
            return this;
        }

        /** Sets the window size used in skipgram model */
        public Builder<V,E> windowSize(int windowSize){
            this.windowSize = windowSize;
            return this;
        }

        public DeepWalk<V,E> build(){
            DeepWalk<V,E> dw = new DeepWalk<>();
            dw.vectorSize = vectorSize;
            dw.windowSize = windowSize;
            dw.batchSize = batchSize;
            dw.learningRate = learningRate;

            return dw;
        }
    }
}
