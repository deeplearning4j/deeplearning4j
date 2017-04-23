package org.nd4j.autodiff.functions;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;


/**
 * Created by agibsonccc on 4/12/17.
 */
@NoArgsConstructor
public abstract class AbstractBinaryReduceFunction<X extends  Field<X>> extends AbstractBinaryFunction<X> {
    protected int[] dimensions;


    public AbstractBinaryReduceFunction(Graph<NDArrayInformation, OpState> graph, DifferentialFunction<X> i_v1, DifferentialFunction<X> i_v2,int...dimensions) {
        super(graph, i_v1, i_v2);
        this.dimensions = dimensions;
    }



    @Override
    protected void addEdges(Graph<NDArrayInformation,OpState> graph,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName) {
        if(i_v1.getValue() instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) i_v1.getValue();
            //skip empty dimensions
            if(dimensions == null)
                return;
            addEdges(graph,i_v1,i_v2,opName,
                    OpState.OpType.ACCUMULATION,
                    Shape.getReducedShape(arrayField.getInput().getShape(),
                            dimensions));

        }

        else
            throw new UnsupportedOperationException("Only supporting array fields");
    }

    @Override
    public String doGetFormula(List<Variable<X>> variables) {
        return toString();
    }

    @Override
    public double getReal() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        return functionName() + "(" + larg() + "," + rarg() + ")";
    }


    @Override
    public DifferentialFunction<X> dup() {
        try {
            return getClass().getConstructor(graph.getClass(),larg().getClass(),rarg().getClass()).newInstance(graph,larg(),rarg());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
