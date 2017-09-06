package org.nd4j.autodiff.functions;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.shape.Shape;

import java.util.List;


/**
 * Created by agibsonccc on 4/12/17.
 */
@NoArgsConstructor
public abstract class AbstractBinaryReduceFunction extends AbstractBinaryFunction {
    protected int[] dimensions;


    public AbstractBinaryReduceFunction(SameDiff sameDiff,
                                        DifferentialFunction i_v1,
                                        DifferentialFunction i_v2,
                                        int...dimensions) {
        super(sameDiff, i_v1, i_v2);
        this.dimensions = dimensions;
        //note that the below won't trigger if dimensions are null
        //please don't remove this
        addEdges(sameDiff,i_v1,
                i_v2,functionName());
    }

    public AbstractBinaryReduceFunction(SameDiff sameDiff) {
        super(sameDiff);
    }


    @Override
    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            DifferentialFunction i_v2,
                            String opName) {
            ArrayField arrayField = i_v1.getValue(true);
            //skip empty dimensions
            if(dimensions == null)
                return;
            addEdges(sameDiff,i_v1,i_v2,opName,
                    OpState.OpType.ACCUMULATION,
                    Shape.getReducedShape(arrayField.getInput().getShape(),
                            dimensions));


    }

    @Override
    public String doGetFormula(List<Variable> variables) {
        return functionName();
    }



    @Override
    public String toString() {
        return functionName() + "(" + larg() + "," + rarg() + ")";
    }


    @Override
    public DifferentialFunction dup() {
        try {
            return getClass().getConstructor(sameDiff.getClass(),larg()
                    .getClass(),rarg().getClass()).newInstance(sameDiff,larg(),rarg());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
