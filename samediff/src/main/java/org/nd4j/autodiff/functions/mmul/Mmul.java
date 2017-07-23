package org.nd4j.autodiff.functions.mmul;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.linalg.api.shape.Shape;

/**
 *  Specialized matrix multiply operations.
 *  Many people know this as "gemm"
 *
 *
 */

public class Mmul<X extends Field<X>> extends TensorMmul<X> {


    public Mmul(SDGraph graph,
                DifferentialFunction<X> i_v1,
                DifferentialFunction<X> i_v2,
                DifferentialFunctionFactory<X> differentialFunctionFactory,
                int argNum) {
        super(graph,
                i_v1,
                i_v2,
                differentialFunctionFactory,new int[][] {
                {1},{0}
        },argNum);
    }



    @Override
    protected void addEdges(Graph<NDArrayInformation,OpState> graph,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName) {
        if(i_v1.getValue(true) instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) i_v1.getValue(true);
            ArrayField secondVal = (ArrayField) i_v2.getValue(true);
            //skip empty dimensions
            addEdges(graph,i_v1,i_v2,opName,
                    OpState.OpType.ACCUMULATION,
                    Shape.getMatrixMultiplyShape(arrayField.getInput().getShape(),secondVal.getInput().getShape()));

        }

        else
            throw new UnsupportedOperationException("Only supporting array fields");
    }



    /**
     * Get the value of this function
     *
     * @return
     */
    @Override
    protected X doGetValue() {
        return differentialFunctionFactory.getMFactory().mmul(larg(),rarg());
    }


    @Override
    public String functionName() {
        return "mmul";
    }


}
