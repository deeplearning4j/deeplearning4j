package org.deeplearning4j.nn.graph.nodes;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/** Merge node: take the activations out of 2 or more layers, and merge (concatenate) them.<br>
 * Exactly how this happens depends on the type of input<br>
 * For feedforward and recurrent nets: merge along activations (dimension 2).<br>
 *     FeedForward: MergeNode([m,nIn1],[m,nIn2]) -> [m,nIn1+nIn2]<br>
 *     RNN: MergeNode([m,nIn1,T],[m,nIn2,T]) -> [m,nIn1+nIn2,T]. Note that time series lengths must be the same<br>
 * For convolutional nets: merge along depth (dimension 4).<br>
 *     CNNs: MergeNode([m,w,h,d1],[m,w,h,d2]) -> [m,w,h,d1+d2]). Examples (m), width (w) and height (h) must be the same<br>
 */
public class MergeNode implements GraphNode {

    private int[][] forwardPassShapes;
    private int fwdPassRank;

    @Override
    public boolean equals(Object o){
        return o instanceof MergeNode;
    }

    @Override
    public int hashCode(){
        return 987654321;
    }

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
        fwdPassRank = activations[0].rank();
        for( int i=0; i<activations.length; i++ ){
            int[] currShape = activations[i].shape();
            if(fwdPassRank != currShape.length){
                throw new IllegalStateException("Cannot merge activations with different ranks: first activations have rank " + fwdPassRank +
                    ", activations[" + i + "] have rank " + currShape.length + " (shape="+Arrays.toString(currShape)+")");
            }
            forwardPassShapes[i] = Arrays.copyOf(currShape,currShape.length);
            if(currShape[0] != nExamples){
                throw new IllegalStateException("Cannot merge activations with different number of examples (activations[0] shape: "
                        + Arrays.toString(activations[0].shape()) + ", activations[" + i + "] shape: " + Arrays.toString(activations[i].shape()));
            }

            nOut += currShape[1];   //Same dimension for all of CNNs, FF, RNNs
        }

        int nOutCumulative = 0;
        INDArray out;
        switch(activations[0].rank()) {
            case 2:
                //Standard feedforward inputs...
                out = Nd4j.create(nExamples, nOut);

                for (INDArray activation : activations) {
                    int[] currShape = activation.shape();
                    out.get(NDArrayIndex.all(), NDArrayIndex.interval(nOutCumulative, nOutCumulative + currShape[1]))
                            .assign(activation);
                    nOutCumulative += currShape[1];
                }
                break;
            case 3:
                //Time series inputs...
                int tsLength = activations[0].size(2);
                out = Nd4j.create(nExamples, nOut, tsLength);

                for (INDArray activation : activations) {
                    int[] currShape = activation.shape();
                    out.get(NDArrayIndex.all(), NDArrayIndex.interval(nOutCumulative, nOutCumulative + currShape[1]), NDArrayIndex.all())
                            .assign(activation);
                    nOutCumulative += currShape[1];
                }

                break;
            case 4:
                fwdPassRank = 4;
                int[] outShape = Arrays.copyOf(activations[0].shape(),4);
                outShape[1] = nOut;
                out = Nd4j.create(outShape);

                //Input activations: [minibatch,depth,width,height]
                for( INDArray activation : activations ){
                    out.get(NDArrayIndex.all(), NDArrayIndex.interval(nOutCumulative, nOutCumulative + activation.size(1)), NDArrayIndex.all(), NDArrayIndex.all())
                            .assign(activation);
                    nOutCumulative += activation.size(1);
                }

                break;
            default:
                throw new UnsupportedOperationException("Cannot merge activations with rank 4 or more");
        }

        return out;
    }

    @Override
    public INDArray[] backward(INDArray epsilon) {
        if(forwardPassShapes.length == 1){
            //No op case
            return new INDArray[]{epsilon};
        }

        //Split the epsilons in the opposite way that the activations were merged
        INDArray[] out = new INDArray[forwardPassShapes.length];
        for( int i=0; i<out.length; i++ ) out[i] = Nd4j.create(forwardPassShapes[i]);

        int cumulative = 0;
        switch(fwdPassRank){
            case 2:
                //Standard
                for( int i=0; i<forwardPassShapes.length; i++ ){
                    out[i].assign(epsilon.get(NDArrayIndex.all(),   //All rows
                            NDArrayIndex.interval(cumulative, cumulative + forwardPassShapes[i][1])));     //subset of columns
                    cumulative += forwardPassShapes[i][1];
                }
                break;
            case 3:
                for( int i=0; i<forwardPassShapes.length; i++ ){
                    out[i].assign(epsilon.get(NDArrayIndex.all(),   //All rows
                            NDArrayIndex.interval(cumulative, cumulative + forwardPassShapes[i][1]), //subset of columns
                            NDArrayIndex.all()));   //All time steps

                    cumulative += forwardPassShapes[i][1];
                }
                break;
            case 4:
                for( int i=0; i<forwardPassShapes.length; i++ ){
                    out[i].assign(epsilon.get(NDArrayIndex.all(),
                            NDArrayIndex.interval(cumulative, cumulative + forwardPassShapes[i][1]),   //Subset of depth
                            NDArrayIndex.all(),     //Width
                            NDArrayIndex.all()));    //height
                    cumulative += forwardPassShapes[i][1];
                }
                break;
            default:
                throw new RuntimeException("Invalid rank during forward pass (not 2, 3, 4)"); //Should never happen
        }

        return out;
    }

    @Override
    public MergeNode clone(){
        return new MergeNode();
    }
}
