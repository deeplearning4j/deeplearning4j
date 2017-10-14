package org.nd4j.autodiff.functions;

import com.google.common.base.Preconditions;
import com.rits.cloning.Cloner;
import lombok.*;

import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.Variable;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;


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
    protected int vertexId;
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
        return new NDArrayVertex[] {getVertex()};
    }

    /**
     * Get the vertex of the output.
     * @return
     */
    public NDArrayVertex getVertex() {
        if(vertex == null)
            return (NDArrayVertex) sameDiff.graph().getVertex(vertexId);
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


    public DifferentialFunction getGradient() {
        return gradient;
    }


    public List<DifferentialFunction> outputs() {
        List<Edge<OpState>> opStates =  sameDiff.graph().getEdgesOut(new int[]{vertexId});
        return Arrays.asList(opStates.get(0).getValue().getDifferentialFunction());
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
        return sameDiff.getFunctionFactory();
    }

    @Override
    public  String getFormula(List<Variable> variables) {
        sameDiff.getGraph().freeze();
        String ret = doGetFormula(variables);
        sameDiff.getGraph().unfreeze();
        return ret;
    }

    public  String doGetFormula(List<Variable> variables) {
        return null;
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
            DifferentialFunction grad = arg.getGradient() != null ? sameDiff.setupFunction(arg.getGradient()) : null;
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

    protected void validateDifferentialFunctionGraph(DifferentialFunction function) {
        Preconditions.checkState(function.getSameDiff() == this.getSameDiff(),"Function applications must be contained in same graph. The left " + function +" must match this function " + this);

    }





    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            DifferentialFunction i_v2,
                            String opName,
                            Op.Type opType,
                            int[] shape) {
        addEdges(sameDiff,
                i_v1,
                i_v2,
                opName,
                opType,
                shape,
                null);

    }

    private INDArray getX() {
        INDArray ret =  sameDiff.getVertexToArray().get(args()[0].getResult().getArrId());
        return ret;
    }

    private INDArray getY() {
        if(args().length > 1) {
            NDArrayInformation opId = args()[1].getResult();
            INDArray ret = sameDiff.getVertexToArray().get(opId.getArrId());
            return ret;
        }
        return null;
    }

    private INDArray getZ() {
        if(this.opState.isInPlace())
            return getX();
        NDArrayInformation opId = opState.getResults()[0];
        INDArray ret =  sameDiff.getVertexToArray().get(opId.getArrId());
        return ret;
    }


    public void fillInArrays() {
        if(this instanceof Op){
            Op op = (Op) this;
            op.setX(getX());
            //y is often optional for many problems
            if(getY() != null)
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
    public NDArrayInformation getResult() {
        if(opState == null || opState.getResults() == null) {
            return  sameDiff.getVertexIdxToInfo().get(resultVertexId());
        }
        return opState.getResults()[0];
    }

    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            DifferentialFunction i_v2,
                            String opName,
                            Op.Type opType,
                            int[] shape, Object[] extraArgs) {
        validateFunctionReference(i_v1);
        validateFunctionReference(i_v2);

        validateDifferentialFunctionGraph(i_v1);
        validateDifferentialFunctionGraph(i_v2);


        /**
         * getValue() generates invalid vertex ids
         * need to look at a way of getting the proper vertex
         * metadata
         *
         * Should be looking at a way to derive the vertex id
         * for each of these equations.
         *
         *
         *
         */
        int v1VertexId = i_v1.resultVertexId();
        int v2VertexId = i_v2.resultVertexId();
        NDArrayInformation arrInfo = inPlace ?  i_v1.getResult() : NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString())
                .id(opName +"(" + i_v1.getResult().getId() + "," + i_v2.getResult().getId() + ")")
                .shape(shape).build();
        //result
        if(vertex == null) {
            vertex = (NDArrayVertex) sameDiff.graph().getVertex(vertexId);
        }

        NDArrayVertex newVertex = new NDArrayVertex(
                sameDiff,
                sameDiff.getGraph().nextVertexId(),
                Math.max(i_v1   .getVertex().depth(),i_v2.getVertex().getDepth()) + 1,
                arrInfo);
        if(newVertex.vertexID() == v2VertexId || newVertex.vertexID() == v1VertexId)
            throw new ND4JIllegalStateException("Illegal vertex id specified in new vertex." +
                    " Perhaps a mismatched graph call? Another likely cause is applyGraph");
        this.vertexId = newVertex.vertexID();
        //add the result vertex
        sameDiff.getGraph().addVertex(newVertex);
        OpState opState,opState2;


        //ensure there's 2 vertices for when the 2 inputs are the same
        if(i_v1.equals(i_v2)) {
            NDArrayVertex dupVertex = new NDArrayVertex(sameDiff,sameDiff.getGraph().nextVertexId(),
                    Math.max(i_v1.getVertex().depth(),i_v2.getVertex().getDepth()) + 1,
                    arrInfo);
            //update vertex id
            v2VertexId = dupVertex.vertexID();
            sameDiff.getGraph().addVertex(dupVertex);
            opState = OpState.builder()
                    .opType(opType).inPlace(inPlace)
                    .differentialFunction(this)
                    .opName(opName)
                    .id(opName + "(" + dupVertex.getValue().getId() + " -> " + newVertex.getValue().getId() + ")")
                    .vertexIds(sameDiff.generateVertexIds(v2VertexId,newVertex.vertexID()))
                    .n(ArrayUtil.prod(shape))
                    .extraArgs(extraArgs)
                    .results(new NDArrayInformation[]{arrInfo})
                    .build();


        }
        else {
            opState =  OpState.builder()
                    .opType(opType)
                    .opName(opName).inPlace(inPlace)
                    .differentialFunction(this)
                    .id(opName + "(" + i_v1.getVertex().getValue().getId() + " -> " + newVertex.getValue().getId() + ")")
                    .vertexIds(sameDiff.generateVertexIds(v2VertexId,newVertex.vertexID()))
                    .n(ArrayUtil.prod(shape))
                    .extraArgs(extraArgs)
                    .results(new NDArrayInformation[]{arrInfo})
                    .build();
        }

        opState2 = OpState.builder()
                .opType(opType).inPlace(inPlace)
                .opName(opName)
                .results(new NDArrayInformation[]{arrInfo})
                .id(opName + "(" + i_v1.getVertex().getValue().getId() + " -> " + newVertex.getValue().getId() + ")")
                .vertexIds(sameDiff.generateVertexIds(v1VertexId,newVertex.vertexID()))
                .n(ArrayUtil.prod(shape))
                .extraArgs(extraArgs)
                .differentialFunction(this)
                .build();


        //add the first vertex no matter what as normal
        sameDiff.graph().addEdge(
                new int[]{v1VertexId},
                new int[]{newVertex.vertexID()},
                opState2,true);

        sameDiff.graph().addEdge(
                new int[]{v2VertexId},
                new int[]{newVertex.vertexID()},
                opState
                ,true);
        newVertex.setOpState(opState2);
        arrInfo.setOwner(opState2);

        this.opState = opState;




    }


    /**
     * Resolve the type of this
     * ndarray based on the op.
     * @return
     */
    public Op.Type resolveType() {
        if(!(this instanceof  Op))
            throw new IllegalStateException("Unable to resolve type. Must be an op");
        if(this instanceof ScalarOp)
            return Op.Type.SCALAR;
        else if(this instanceof ShapeOp)
            return Op.Type.SHAPE;
        else if(this instanceof TransformOp)
            return Op.Type.TRANSFORM;
        else if(this instanceof BroadcastOp)
            return Op.Type.BROADCAST;
        else if(this instanceof Accumulation) {
            Accumulation accumulation = (Accumulation) this;
            if(accumulation.y() != null)
                return Op.Type.REDUCE3;
            else
                return Op.Type.REDUCE;
        }
        else if(this instanceof Variance)
            return Op.Type.VARIANCE;
        else if(this instanceof IndexAccumulation)
            return Op.Type.INDEXREDUCE;

        throw new IllegalStateException("No type found for class " + getClass().getName());

    }


    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            DifferentialFunction i_v2,
                            String opName) {
        validateDifferentialFunctionGraph(i_v1);
        validateDifferentialFunctionGraph(i_v2);
        validateFunctionReference(i_v1);
        validateFunctionReference(i_v2);


        addEdges(sameDiff,
                i_v1,
                i_v2,
                opName,
                resolveType(),
               i_v1.getResultShape());


    }


    public DifferentialFunction larg() {
        return args[0];
    }

    public DifferentialFunction rarg() {
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


    protected void validateFunctionReference(List<DifferentialFunction> reference) {
        for(int i = 0; i < reference.size(); i++) {
            validateFunctionReference(reference.get(i));
        }

    }
    protected void validateFunctionReference(DifferentialFunction reference) {
        if(sameDiff.getFunctionInstances().containsKey(reference.getVertexId())) {
            DifferentialFunction get = sameDiff.getFunctionInstances()
                    .get(reference.getVertexId());
            Preconditions.checkState(reference.equals(get), "Found invalid reference " + reference + " for vertex id "
                    + reference.getVertexId());
        }


    }

    /**
     * Return the vertex id
     * of the result
     * of this equation.
     *
     * @return
     */
    public  int resultVertexId() {
        return vertexId;
    }





    /**
     * Add nodes to the graph
     * @param sameDiff
     * @param i_v1
     * @param opName
     */
    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            String opName,
                            int...shape) {
        validateFunctionReference(i_v1);
        NDArrayInformation information =   inPlace ? i_v1.getResult() :  NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString())
                .id(opName + "(" + i_v1.getResult().getId() + " -> " +
                        i_v1.getResult().getId() + ")")
                .shape(shape).build();
        //result
        NDArrayVertex newVertex = new NDArrayVertex(
                sameDiff,
                sameDiff.graph().nextVertexId(),
                i_v1.getVertex().depth() + 1,
                information);
        this.vertexId = newVertex.vertexID();
        sameDiff.graph().addVertex(newVertex);
        Preconditions.checkArgument(sameDiff == i_v1.sameDiff,"Illegal samediff instance");
        OpState owner =  OpState.builder()
                .opType(resolveType()).differentialFunction(this)
                .opName(opName).inPlace(inPlace)
                .extraArgs(extraArgs).axes(dimensions)
                .id(opName + "(" + i_v1.getResult().getId() + " -> " + newVertex.getValue().getId() + ")")
                .vertexIds(sameDiff.generateVertexIds(i_v1.getVertex().vertexID(),newVertex.vertexID()))
                .n(ArrayUtil.prod(shape)).results(new NDArrayInformation[] { information })
                .build();


        sameDiff.getGraph().addEdge(
                new int[]{arg().resultVertexId()},
                new int[]{newVertex.vertexID()},
                owner,
                true);



        newVertex.setOpState(owner);
        information.setOwner(owner);
        owner.setResults(new NDArrayInformation[]{information});
        if(owner.isInPlace()) {
            information.setArrId(i_v1.getResult().getArrId());
        }

        this.opState = owner;

        if(!sameDiff.getVertexIdxToInfo().containsKey(newVertex.vertexID()))
            sameDiff.getVertexIdxToInfo().put(newVertex.vertexID(),information);

        else
            throw new IllegalStateException("Found duplicate vertex information");

    }



    /**
     * Set a forward function reference
     * and a gradient reference
     * for this function
     * @param gradient
     */
    public void setGradient(DifferentialFunction gradient) {
        DifferentialFunction functionRef = sameDiff.getFunctionInstances().get(vertexId);
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
        result = 31 * result + vertexId;
        return result;
    }


    protected void validateDifferentialFunctionsameDiff(
            List<DifferentialFunction> function) {
        for(DifferentialFunction differentialFunction : function)
            validateDifferentialFunctionsameDiff(differentialFunction);
    }



    protected void validateDifferentialFunctionsameDiff(
            DifferentialFunction function) {

        Preconditions.checkState(function != null,"Passed in function was null.");
        Preconditions.checkState(function.getSameDiff() == sameDiff);

        Preconditions.checkState(function.getSameDiff() ==
                        this.getSameDiff(),
                "Function applications must be contained " +
                        "in same sameDiff. The left " + function +"" +
                        " must match this function " + this);
        Preconditions.checkState(sameDiff ==
                this.getSameDiff(),"Function applications m" +
                "ust be " +
                "contained in same sameDiff. The left " + function +" " +
                "must " +
                "match this function " + this);

    }

    protected int fromBoolean(boolean bool) {
        return bool ? 1 : 0;
    }



}
