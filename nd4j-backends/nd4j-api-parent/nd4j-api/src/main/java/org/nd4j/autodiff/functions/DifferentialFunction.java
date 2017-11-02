package org.nd4j.autodiff.functions;

import com.rits.cloning.Cloner;
import lombok.*;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.Arrays;
import java.util.List;


@AllArgsConstructor
@Data
@NoArgsConstructor
public abstract class DifferentialFunction implements Differential {

    @Getter
    @Setter
    protected SameDiff sameDiff;
    @Getter
    protected OpState opState;
    @Getter
    @Setter
    protected int[] vertexId;
    @Setter
    protected NDArrayVertex vertex;
    @Getter
    @Setter
    protected DifferentialFunction gradient;
    @Getter
    @Setter
    protected boolean inPlace;
    @Getter
    @Setter
    protected boolean gradFunction;


    @Getter
    @Setter
    protected DifferentialFunction forwardFunction;

    @Getter
    @Setter
    protected int[] shape;

    @Getter
    @Setter
    protected DifferentialFunction[] args;


    @Getter
    @Setter
    protected Number scalarValue;


    @Getter
    @Setter
    protected int[] dimensions;

    protected Object[] extraArgs;


    /**
     * Get the output vertex ids for this function
     * @return the set of output vertex ids for this function.
     */
    public int[] getOutputVertexIds() {
        NDArrayVertex[] outputs = getVertices();
        int[] ret = new int[outputs.length];
        for(int i = 0; i < outputs.length; i++) {
            ret[i] = outputs[i].vertexID();
        }

        return ret;
    }

    /**
     * Return the output functions for this differential function.
     * @return
     */
    public DifferentialFunction[] outputFunctions() {
        return new DifferentialFunction[]{this};
    }

    /**
     * Get the vertices of the outputs.
     * @return
     */
    public NDArrayVertex[] getVertices() {
        NDArrayVertex[] ret = new NDArrayVertex[vertexId.length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (NDArrayVertex) sameDiff.graph().getVertex(vertexId[i]);
        }

        return ret;
    }

    /**
     * Get the vertex of the output.
     * @return
     */
    public NDArrayVertex getVertex() {
        if(vertexId == null || vertexId.length > 1)
            throw new IllegalStateException("Unable to obtain single vertex. Function has more than 1.");
        if(vertex == null && getSameDiff().graph().getVertices().containsKey(vertexId[0]))
            return (NDArrayVertex) getSameDiff().graph().getVertex(vertexId[0]);
        return vertex;
    }


    /**
     *
     * @param sameDiff
     * @param extraArgs
     */
    public DifferentialFunction(SameDiff sameDiff,boolean inPlace, Object[] extraArgs) {
        this.sameDiff = sameDiff;
        this.inPlace = inPlace;
        this.extraArgs = extraArgs;
    }


    /**
     *
     * @param sameDiff
     * @param extraArgs
     */
    public DifferentialFunction(SameDiff sameDiff, Object[] extraArgs) {
        this.sameDiff = sameDiff;
        this.extraArgs = extraArgs;
    }

    public DifferentialFunction(SameDiff sameDiff, DifferentialFunction[] args) {
        this(sameDiff,false,args);
    }

    public DifferentialFunction(SameDiff sameDiff, boolean inPlace, DifferentialFunction[] args) {
        this.sameDiff = sameDiff;
        this.inPlace = inPlace;
        this.args = args;
    }

    /**
     * Get the result shape for this function
     * @return
     */
    public int[] getResultShape() {
        if(opState == null || opState.getResults() == null || opState.getResults().length > 1)
            throw new IllegalStateException("Unable to get result shape with null op state");
        return opState.getResults()[0].getShape();
    }


    /**
     * Get the gradient for this function.
     * @return
     */
    public DifferentialFunction getGradient() {
        if(gradient == null)
            return null;
        return sameDiff.setupFunction(gradient);
    }


    /**
     * Get the output functions for this function
     * @return
     */
    public List<DifferentialFunction> outputs() {
        return Arrays.asList(this);
    }

    public  boolean isVariable() {
        return false;
    }




    public int depth() {
        return getVertex().getDepth();
    }


    /**
     * The actual implementation for automatic differentiation.
     *
     * @param f1
     * @return
     */
    public abstract List<DifferentialFunction> doDiff(List<DifferentialFunction> f1);

    /**
     * Shortcut for the {@link DifferentialFunctionFactory}
     * @return
     */
    public DifferentialFunctionFactory f() {
        return sameDiff.f();
    }



    @Override
    public abstract String toString();



    public boolean isConstant() {
        return false;
    }

    public  DifferentialFunction[] args() {
        return args;
    }

    public  DifferentialFunction arg() {
        return args[0];
    }


    @Override
    public  List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
        List<DifferentialFunction> vals = doDiff(i_v1);
        for(int i = 0; i < vals.size(); i++) {
            DifferentialFunction differentialFunction = sameDiff.setupFunction(vals.get(i));
            DifferentialFunction arg = sameDiff.setupFunction(args()[i]);
            DifferentialFunction grad = arg.getGradient();
            if(grad != null) {
                DifferentialFunction ret = f().addi(differentialFunction, grad);
                arg.setGradient(ret);
            }
            else
                arg.setGradient(differentialFunction);
            differentialFunction.setGradFunction(true);
        }

        return vals;
    }


    public String opName() {
        if(this instanceof  Op) {
            Op op = (Op) this;
            return op.name();
        }
        throw new UnsupportedOperationException();
    }


    public Op.Type opType() {
        throw new UnsupportedOperationException();
    }


    private INDArray getX() {
        INDArray ret =  sameDiff.getVariable(args()[0].getResult().getVarName()).getArr();
        return ret;
    }

    private INDArray getY() {
        if(args().length > 1) {
            SDVariable opId = args()[1].getResult();
            INDArray ret = sameDiff.getVariable(opId.getVarName()).getArr();
            return ret;
        }
        return null;
    }

    private INDArray getZ() {
        if(this.opState.isInPlace())
            return getX();
        SDVariable opId = opState.getResults()[0];
        INDArray ret =  sameDiff.getVariable(opId.getVarName()).getArr();
        return ret;
    }


    public void fillInArrays() {
        if(this instanceof Op){
            Op op = (Op) this;
            op.setX(getX());
            //y is often optional for many problems
            if(args().length > 1)
                op.setY(getY());
            op.setZ(getZ());
        }
        else
            throw new IllegalStateException("Unable to fill in arrays. Type must be an operation.");
    }

    /**
     * Get the result
     * @return
     */
    public SDVariable getResult() {
        if(opState == null || opState.getResults() == null) {
            throw new ND4JIllegalStateException("No op state for variable found for obtaining result.");
        }
        return opState.getResults()[0];
    }





    /**
     * The left argument for this function
     * @return
     */
    public DifferentialFunction larg() {
        if(args == null || args.length == 0)
            throw new ND4JIllegalStateException("No arguments found.");
        return args[0];
    }

    /**
     * The right argument for this function.
     * Note that this assumes that there are 2 args for this
     * function, if 2 are not set, it throws an
     * {@link ND4JIllegalStateException}
     * @return
     */
    public DifferentialFunction rarg() {
        if(args == null || args.length != 2)
            throw new ND4JIllegalStateException("In order to use this function, the numebr of arguments for this function must be 2.");
        return args[1];
    }


    /**
     * Duplicate this function
     * @return
     */
    public  DifferentialFunction dup() {
        Cloner cloner = new Cloner();
        return cloner.deepClone(this);
    }



    /**
     * Return the vertex id
     * of the result
     * of this equation.
     *
     * @return
     */
    public  int[] resultVertexId() {
        return vertexId;
    }


    /**
     * Calculate the output shape for this op
     * @return
     */
    public List<int[]> calculateOutputShape() {
        throw new UnsupportedOperationException();
    }

    /**
     * Set a forward function reference
     * and a gradient reference
     * for this function
     * @param gradient
     */
    public void setGradient(DifferentialFunction gradient) {
        DifferentialFunction functionRef = sameDiff.getFunctionForVertexId(vertexId);
        if(functionRef != this)
            functionRef.setGradient(gradient);
        this.gradient = sameDiff.setupFunction(gradient);
        this.gradient.setForwardFunction(this);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DifferentialFunction that = (DifferentialFunction) o;

        if (vertexId != that.vertexId) return false;
        if (opState != null ? !opState.equals(that.opState) : that.opState != null) return false;
        //if (gradient != null ? !gradient.equals(that.gradient) : that.gradient != null) return false;
        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (opState != null ? opState.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(vertexId);
        return result;
    }



    protected int fromBoolean(boolean bool) {
        return bool ? 1 : 0;
    }



}
