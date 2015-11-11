package org.deeplearning4j.graph.models.embeddings;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.graph.models.BinaryTree;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class InMemoryGraphLookupTable implements GraphVectorLookupTable {

    protected int nVertices;
    protected int vectorSize;
    protected BinaryTree tree;
    protected INDArray vertexVectors;   //'input' vectors
    protected INDArray outWeights;      //'output' vectors
    protected double learningRate;  //TODO

    public InMemoryGraphLookupTable(int nVertices, int vectorSize, BinaryTree tree, double learningRate ){
        this.nVertices = nVertices;
        this.vectorSize = vectorSize;
        this.tree = tree;
        this.learningRate = learningRate;
        this.vertexVectors = Nd4j.rand(nVertices,vectorSize).subi(0.5).divi(vectorSize);
        this.outWeights = Nd4j.rand(nVertices,vectorSize).subi(0.5).divi(vectorSize);
    }

    @Override
    public int vectorSize() {
        return vectorSize;
    }

    @Override
    public void resetWeights() {
        vertexVectors = Nd4j.rand(nVertices,vectorSize);
    }

    @Override
    public void iterate(int first, int second) {

        //Get vector for first vertex, as well as code:
        INDArray vec = vertexVectors.getRow(first);
        int codeLength = tree.getCodeLength(first);
        long code = tree.getCode(first);
        int[] innerNodesForVertex = tree.getPathInnerNodes(first);

        //Calculate probability - heirarchical softmax P(vertex_second|vertex_first)
        //http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        //http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf


        Level1 l1 = Nd4j.getBlasWrapper().level1();
        double prob = 1.0;
        INDArray accumError = Nd4j.create(vec.shape());
        for( int i=0; i<codeLength; i++ ){

            //Inner node:
            int innerNodeIdx = innerNodesForVertex[i];
            INDArray nwi = outWeights.getRow(innerNodeIdx);


            double dot = Nd4j.getBlasWrapper().dot(nwi,vec);

            boolean path = getBit(code, i);  //left or right?

            double sigmoidDot = sigmoid(dot);
            double innerProb = (path ? sigmoid(dot) : sigmoid(-dot));   //prob of going left or right at inner node

            prob *= innerProb;



            //Update rule for output (softmax) parameters:
            double a;
            if(path){
                //nwi.subi(learningRate*(sigmoidDot-1.0) * vec)
                a = -learningRate * (sigmoidDot-1);

                //Accumulate error
                l1.axpy(vec.length(),sigmoidDot-1,nwi,accumError);
//                errorSum += sigmoidDot - 1;
            } else {
                a = -learningRate * sigmoidDot;

                //Accumulate error
                l1.axpy(vec.length(),sigmoidDot,nwi,accumError);
//                errorSum += sigmoidDot;
            }
            l1.axpy(vec.length(),a,vec,nwi);
        }

        //Update vector representation of vertex given errors:
        l1.axpy(vec.length(),-learningRate,accumError,vec);
    }

    @Override
    public INDArray getVector(int idx) {
        return vertexVectors.getRow(idx);
    }

    private static double sigmoid(double in){
        return 1.0 / (1.0 + FastMath.exp(-in));
    }

    private boolean getBit(long in, int bitNum){
        long mask = 1L << bitNum;
        return (in & mask) != 0L;
    }
}
