package org.nd4j.autodiff.functions;

import com.rits.cloning.Cloner;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;


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


    public DifferentialFunction(SameDiff sameDiff, OpState opState, int[] vertexId, boolean inPlace, boolean gradFunction, DifferentialFunction forwardFunction, int[] shape, DifferentialFunction[] args, Number scalarValue, int[] dimensions, Object[] extraArgs) {
        this.sameDiff = sameDiff;
        this.opState = opState;
        this.vertexId = vertexId;
        this.inPlace = inPlace;
        this.gradFunction = gradFunction;
        this.forwardFunction = forwardFunction;
        this.shape = shape;
        this.args = args;
        this.scalarValue = scalarValue;
        this.dimensions = dimensions;
        this.extraArgs = extraArgs;

    }


    protected void addAsNewVertexId(int[] vertexId) {
        SDVariable var = sameDiff.var(opName() + "-" + UUID.randomUUID().toString(),shape,new ZeroInitScheme('f'),vertexId,maxDepthForArgs() + 1);
        if(sameDiff.graph().getVertex(vertexId[0]) == null) {
            NDArrayVertex ndArrayVertex = new NDArrayVertex(sameDiff, var.vertexId[0], depth(), var);
            var.setVertexId(new int[]{ndArrayVertex.vertexID()});
        }


        this.vertexId = var.getVertexId();
        var.setOpState(opState);
        sameDiff.addVariable(var);
        sameDiff.putFunction(var.getVertexId(),this);

    }

    protected void addAsNewVertexId() {
        int vertexId = sameDiff.graph().getNextVertexId()  > sameDiff.graph().numVertices() ? sameDiff.graph().getNextVertexId() : sameDiff.graph().nextVertexId();
         addAsNewVertexId(new int[]{vertexId});
    }


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
     * Get the result shape for this function
     * @return
     */
    public int[] getResultShape() {
        return getResult().getResultShape();
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
        return sameDiff.getGraph().getVertex(vertexId[0]).depth();
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
            SDVariable var = sameDiff.getVariableForVertexId(arg.vertexId);
            DifferentialFunction grad = var.getGradient();
            if(grad != null) {
                DifferentialFunction ret = f().addi(differentialFunction, grad);
                sameDiff.updateVariableName(ret.getVertexId(),var.getVarName() + "-grad");
                sameDiff.setGradientForVertexId(var.vertexId,sameDiff.getVariableForVertexId(ret.vertexId));
                sameDiff.setForwardVariableForVertexId(ret.vertexId,var);
            }
            else {
                SDVariable gradVar = sameDiff.getVariableForVertexId(differentialFunction.getVertexId());
                sameDiff.setGradientForVertexId(var.vertexId, gradVar);
                sameDiff.setForwardVariableForVertexId(gradVar.vertexId,var);
            }
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
        INDArray ret =  args()[0].getResult().getArr();
        return ret;
    }

    private INDArray getY() {
        if(args().length > 1) {
            SDVariable opId = args()[1].getResult();
            INDArray ret = opId.getArr();
            return ret;
        }
        return null;
    }

    private INDArray getZ() {
        if(this.opState.isInPlace())
            return getX();
        SDVariable opId = getResult();
        INDArray ret = opId.getArr();
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
        return sameDiff.getVariableForVertexId(vertexId);
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


    public int maxDepthForArgs() {
        int depth = -1;
        for(DifferentialFunction arg : args()) {
            depth = Math.max(arg.depth(),depth);
        }

        return depth;
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
