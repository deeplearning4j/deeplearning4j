package org.deeplearning4j.nn.conf.graph;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.deeplearning4j.nn.graph.ComputationGraph;

/** An ElementWiseVertex is used to combine the activations of two or more layer in an element-wise manner<br>
 * For example, the activations may be combined by addition, subtraction or multiplication.
 * Addition may use an arbitrary number of input arrays. Note that in the case of subtraction, only two inputs may be used.
 * @author Alex Black
 */
@Data
public class ElementWiseVertex extends GraphVertex {

    public ElementWiseVertex(@JsonProperty("op") Op op) {
        this.op = op;
    }

    public enum Op {Add, Subtract, Product};

    protected Op op;

    @Override
    public ElementWiseVertex clone() {
        return new ElementWiseVertex(op);
    }

    @Override
    public boolean equals(Object o){
        if(!(o instanceof ElementWiseVertex)) return false;
        return ((ElementWiseVertex)o).op == op;
    }

    @Override
    public int hashCode(){
        return op.hashCode();
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        org.deeplearning4j.nn.graph.vertex.impl.ElementWiseVertex.Op op;
        switch(this.op){
            case Add:
                op = org.deeplearning4j.nn.graph.vertex.impl.ElementWiseVertex.Op.Add;
                break;
            case Subtract:
                op = org.deeplearning4j.nn.graph.vertex.impl.ElementWiseVertex.Op.Subtract;
                break;
            case Product:
                op = org.deeplearning4j.nn.graph.vertex.impl.ElementWiseVertex.Op.Product;
                break;
            default:
                throw new RuntimeException();
        }
        return new org.deeplearning4j.nn.graph.vertex.impl.ElementWiseVertex(graph,name,idx,op);
    }
}
