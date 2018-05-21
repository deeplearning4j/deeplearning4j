package org.nd4j.linalg.api.ops.impl.transforms.comparison;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.Collections;
import java.util.List;

/**
 * This op allows us to (based on the passed in condition)
 * to return the element fulfilling the condition.
 * In numpy, this is equivalent to the boolean indexing like:
 * a[a > 2] which returns all elements in the array greater than 2
 * as a flat vector.
 *
 * This op interops with underlying libnd4j by leveraging the {@link Condition#condtionNum()}
 *
 * @author Adam Gibson
 */
public class Choose extends DynamicCustomOp {
    private Condition condition;

    public Choose(SameDiff sameDiff, SDVariable[] args, Condition condition) {
        super(null, sameDiff, args);
        if(condition == null) {
            throw new ND4JIllegalArgumentException("Must specify a condition.");
        }

        this.inPlace = true;
        this.inplaceCall = true;
        addIArgument(condition.condtionNum());
        this.condition = condition;
    }

    public Choose(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments) {
        super(opName, inputs, outputs, tArguments, iArguments);
    }

    public Choose(String opName, INDArray[] inputs, Condition condition) {
        super(opName, inputs, null);
        if(condition == null) {
            throw new ND4JIllegalArgumentException("Must specify a condition.");
        }

        addInputArgument(inputs);
        addIArgument(condition.condtionNum());
        addOutputArgument(Nd4j.create(inputs[0].length()),Nd4j.scalar(1.0));
    }

    /**
     *
     * @param inputs
     * @param condition
     */
    public Choose(INDArray[] inputs,Condition condition) {
        this(inputs, Collections.<Integer>emptyList(),Collections.<Double>emptyList(),condition);
    }

    /**
     * Note that iArgs (integer arguments) and  tArgs(double/float arguments)
     * may end up being used under the following conditions:
     * scalar operations (if a scalar is specified the you do not need to specify an ndarray)
     * otherwise, if an ndarray is needed as a second input then put it in the inputs
     *
     * Usually, you only need 1 input (the equivalent of the array you're trying to do indexing on)
     *
     * @param inputs the inputs in to the op
     * @param iArgs the integer arguments as needed
     * @param tArgs the arguments
     * @param condition the condition to filter on
     */
    public Choose(INDArray[] inputs,List<Integer> iArgs, List<Double> tArgs,Condition condition) {
        super(null, inputs, null);
        if(condition == null) {
            throw new ND4JIllegalArgumentException("Must specify a condition.");
        }

        if(!iArgs.isEmpty())
            addIArgument(Ints.toArray(iArgs));

        if(!tArgs.isEmpty())
            addTArgument(Doubles.toArray(tArgs));
        addIArgument(condition.condtionNum());
        addOutputArgument(Nd4j.create(inputs[0].shape(), inputs[0].ordering()),Nd4j.trueScalar(1.0));
    }

    public Choose(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(opName, sameDiff, args, inPlace);
    }

    public Choose(){
        //No-arg constructor for use in DifferentialFunctionClassHolder
    }

    @Override
    public String opName() {
        return "choose";
    }
}
