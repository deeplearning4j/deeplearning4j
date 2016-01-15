package org.deeplearning4j.nn.graph.nodes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/** Merge node: take the activations out of 2 or more layers, and merge them.
 * So, MergeNode([m,nIn1],[m,nIn2]) -> [m,nIn1+nIn2]
 * For standard feed-forward activations (2d) and time series activations (3d)
 */
public class MergeNode implements GraphNode {

    private int[][] forwardPassShapes;
    private int fwdPassRank;

    @Override
    public INDArray forward(INDArray... activations) {
        if(activations.length == 1){
            //No-op case
            int[] shape = activations[0].shape();
            forwardPassShapes = new int[][]{Arrays.copyOf(shape,shape.length)};
            return activations[0];
        }

        forwardPassShapes = new int[activations.length][0];
        int nExamples = activations[0].size(0);
        int nOut = 0;
        for( int i=0; i<activations.length; i++ ){
            int[] currShape = activations[i].shape();
            forwardPassShapes[i] = Arrays.copyOf(currShape,currShape.length);
            if(currShape[0] != nExamples){
                throw new IllegalStateException("Cannot merge activations with different number of examples (activations[0] shape: "
                        + Arrays.toString(activations[0].shape()) + ", activations[" + i + "] shape: " + Arrays.toString(activations[i].shape()));
            }
            nOut += currShape[1];
        }

        INDArray out;
        if(activations[0].rank() == 2){
            //Standard inputs...
            fwdPassRank = 2;

            out = Nd4j.create(nExamples, nOut);

            int nOutCumulative = 0;
            for (INDArray activation : activations) {
                int[] currShape = activation.shape();
                out.get(NDArrayIndex.all(), NDArrayIndex.interval(nOutCumulative, nOutCumulative + currShape[1]))
                        .assign(activation);
                nOutCumulative += currShape[1];
            }

        } else if(activations[0].rank() == 3){
            //Time series inputs...
            fwdPassRank = 3;

            int tsLength = activations[0].size(2);
            out = Nd4j.create(nExamples, nOut, tsLength);

            int nOutCumulative = 0;
            for (INDArray activation : activations) {
                int[] currShape = activation.shape();
                out.get(NDArrayIndex.all(), NDArrayIndex.interval(nOutCumulative, nOutCumulative + currShape[1]), NDArrayIndex.all())
                        .assign(activation);
                nOutCumulative += currShape[1];
            }

        } else {
            throw new UnsupportedOperationException("Cannot merge activations with rank 4 or more");
        }

        return out;
    }

    @Override
    public INDArray[] backward(INDArray... epsilons) {
        if(epsilons == null || epsilons.length != 1) throw new IllegalArgumentException("Invalid input: expect one epsilon during forward pass");
        if(forwardPassShapes.length == 1){
            //No op case
            return new INDArray[]{epsilons[0]};
        }

        //Split the epsilons in the opposite way that the activations were merged
        INDArray[] out = new INDArray[forwardPassShapes.length];
        for( int i=0; i<out.length; i++ ) out[i] = Nd4j.create(forwardPassShapes[i]);

        if(fwdPassRank == 2){
            //Standard
            int cumulative = 0;
            for( int i=0; i<forwardPassShapes.length; i++ ){
                out[i].assign(epsilons[0].get(NDArrayIndex.all(),   //All rows
                        NDArrayIndex.interval(cumulative,cumulative+forwardPassShapes[i][1])));     //subset of columns
                cumulative += forwardPassShapes[i][1];
            }

        } else {
            //Time series
            int cumulative = 0;
            for( int i=0; i<forwardPassShapes.length; i++ ){
                out[i].assign(epsilons[0].get(NDArrayIndex.all(),   //All rows
                        NDArrayIndex.interval(cumulative,cumulative+forwardPassShapes[i][1]), //subset of columns
                        NDArrayIndex.all()));   //All time steps

                cumulative += forwardPassShapes[i][1];
            }
        }
        return out;
    }
}
