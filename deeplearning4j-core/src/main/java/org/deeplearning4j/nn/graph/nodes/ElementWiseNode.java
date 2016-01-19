package org.deeplearning4j.nn.graph.nodes;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.nd4j.linalg.api.ndarray.INDArray;

/** Graph node for element-wise operations on activations (such as adding, subtracting, multiplying)
 *
 */
public class ElementWiseNode implements GraphNode {
    public enum Op {Add, Subtract, Product};

    private Op op;
    private int nInForwardPass;

    public ElementWiseNode(@JsonProperty("op") Op op){
        this.op = op;
    }

    public Op getOp() {
        return this.op;
    }

    public void setOp(Op op) {
        this.op = op;
    }

    public String toString() {
        return "ElementWiseNode(" + this.op + ")";
    }

    public boolean equals(Object o) {
        if (!(o instanceof ElementWiseNode)) return false;
        ElementWiseNode e = (ElementWiseNode)o;
        return op == e.op;
    }

    public int hashCode() {
        return op.hashCode();
    }

    @Override
    public INDArray forward(INDArray... activations) {
        nInForwardPass = activations.length;
        if(activations.length == 1) return activations[0];

        switch(op){
            case Add:
                INDArray sum = activations[0].dup();
                for( int i=1; i<activations.length; i++){
                    sum.addi(activations[i]);
                }
                return sum;
            case Subtract:
                if(activations.length != 2) throw new IllegalArgumentException("Subtraction node only supports 2 inputs");
                //TODO: maybe specify a convention: (first - (second + third + fourth + ...) etc?)
                //Or, maybe not. Can always build that in two steps anyway with an add and a binary subtract ops)
                return activations[0].sub(activations[1]);
            case Product:
                throw new UnsupportedOperationException("Not yet implemented");
            default:
                throw new UnsupportedOperationException("Unknown op: " + op);
        }
    }

    @Override
    public INDArray[] backward(INDArray epsilon) {
        if(nInForwardPass == 1) return new INDArray[]{epsilon};

        switch(op){
            case Add:
                //If x=sum_i a_i then dL/da_i = dL/dx * dx/da_i = dL/dx
                INDArray[] out = new INDArray[nInForwardPass];
                out[0] = epsilon;
                for( int i=1; i<nInForwardPass; i++ ) out[i] = out[0].dup();
                return out;
            case Subtract:
                INDArray[] out2 = new INDArray[2];
                out2[0] = epsilon;
                out2[1] = epsilon.mul(-1);
                return out2;
            case Product:
                throw new UnsupportedOperationException("Not yet implemented");
            default:
                throw new UnsupportedOperationException("Unknown op: " + op);
        }
    }

    @Override
    public ElementWiseNode clone(){
        return new ElementWiseNode(op);
    }
}
