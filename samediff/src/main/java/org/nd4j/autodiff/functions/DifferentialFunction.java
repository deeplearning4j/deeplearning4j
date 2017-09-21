package org.nd4j.autodiff.functions;

import com.google.common.base.Preconditions;
import lombok.*;
import org.nd4j.autodiff.ArrayFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;

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


    protected Object[] extraArgs;

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


    /**
     * Get the result shape for this function
     * @return
     */
    public int[] getResultShape() {
        if(opState == null)
            throw new IllegalStateException("Unable to get result shape with null op state");
        return opState.getResult().getShape();
    }


    public DifferentialFunction getGradient() {
        return gradient;
    }

    public  boolean isVariable() {
        return false;
    }

    /**
     * Get the value of this function
     * @return
     */
    public abstract ArrayField doGetValue();


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

    /**
     * Shortcut for the {@link ArrayFactory}
     * @return
     */
    public ArrayFactory a() {
        return sameDiff.getArrayFactory();
    }


    /**
     * Get the value specifying
     * whether to freeze the graph or not
     * @param freeze whether to freeze the graph or not,
     *               this means whether to add nodes to the internal
     *               computation graph or not
     * @return the value of this function
     */
    public  ArrayField getValue(boolean freeze) {
        boolean graphAlreadyFrozen = this.sameDiff.getGraph().isFrozen();
        //if graph is already frozen leave it frozen
        if(freeze && !graphAlreadyFrozen) {
            this.sameDiff.getGraph().freeze();
        }

        ArrayField val = doGetValue();
        ArrayField arrayField = sameDiff.setupArrayField(val);
        val = arrayField;
        Preconditions.checkState(arrayField.getOps() == this.sameDiff,"Same diff instances for get value not the same.");



        if(!freeze) {
            Preconditions.checkState(arrayField.getOps() == this.sameDiff,"Same diff instances for get value not the same.");
            NDArrayVertex vertex = (NDArrayVertex) getSameDiff().getGraph().getVertex(getVertexId());
            arrayField.setVertex(vertex);
            arrayField.setOps(this.sameDiff);
            Preconditions.checkState(vertex != null,"Vertex " + getVertexId() + " was null.");
            Preconditions.checkState(vertex.getValue() != null,"Vertex did not have a value set.");
            arrayField.getInput().setScalarValue(vertex.getValue().getScalarValue());
            arrayField.setInput(vertex.getValue());
            Preconditions.checkState(sameDiff == arrayField.getOps(),"Passed in array factory != the passed in graph. Unable to instantiate.");

        }

        if(freeze && !graphAlreadyFrozen) {
            this.sameDiff.getGraph().unfreeze();
        }

        return sameDiff.setupArrayField(val);
    }


    @Override
    public  String getFormula(List<Variable> variables) {
        sameDiff.getGraph().freeze();
        String ret = doGetFormula(variables);
        sameDiff.getGraph().unfreeze();
        return ret;
    }

    public abstract String doGetFormula(List<Variable> variables);

    public abstract String functionName();

    @Override
    public abstract String toString();



    public boolean isConstant() {
        return false;
    }

    public abstract DifferentialFunction[] args();

    public abstract DifferentialFunction arg();


    @Override
    public  List<DifferentialFunction> diff(List<DifferentialFunction> i_v1) {
        List<DifferentialFunction> vals = doDiff(i_v1);
        for(int i = 0; i < vals.size(); i++) {
            DifferentialFunction differentialFunction = sameDiff.setupFunction(vals.get(i));
            DifferentialFunction arg = sameDiff.setupFunction(args()[i]);
            DifferentialFunction grad = arg.getGradient() != null ? sameDiff.setupFunction(arg.getGradient()) : null;
            if(grad != null)
                f().addi(differentialFunction,grad);
            else
                arg.setGradient(differentialFunction);
            differentialFunction.setGradFunction(true);
        }

        return vals;
    }

    private void validateDifferentialFunctionGraph(DifferentialFunction function) {
        Preconditions.checkState(function.getSameDiff() == this.getSameDiff(),"Function applications must be contained in same graph. The left " + function +" must match this function " + this);

    }




    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            DifferentialFunction i_v2,
                            String opName,
                            OpState.OpType opType,
                            int[] shape) {
        addEdges(sameDiff,
                i_v1,
                i_v2,
                opName,
                opType,
                shape,
                null);

    }

    /**
     * Get the result
     * @return
     */
    public NDArrayInformation getResult() {
        return opState.getResult();
    }

    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            DifferentialFunction i_v2,
                            String opName,
                            OpState.OpType opType,
                            int[] shape, Object[] extraArgs) {
        validateFunctionReference(i_v1);
        validateFunctionReference(i_v2);

        validateDifferentialFunctionGraph(i_v1);
        validateDifferentialFunctionGraph(i_v2);
        validateDifferentialFunctionsameDiff(i_v1.getValue(true));


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
        ArrayField v1 = i_v1.getValue(true);
        int v1VertexId = i_v1.resultVertexId();
        ArrayField v2 = i_v2.getValue(true);
        int v2VertexId = i_v2.resultVertexId();
        validateDifferentialFunctionsameDiff(v1);
        validateDifferentialFunctionsameDiff(v2);

        NDArrayInformation arrInfo = inPlace ?  i_v1.getResult() : NDArrayInformation.builder()
                .arrId(UUID.randomUUID().toString())
                .id(opName +"(" + v1.getInput().getId() + "," + v2.getInput().getId() + ")")
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
                    .vertexIds(new String[]{String.valueOf(v2VertexId),String.valueOf(newVertex.vertexID())})
                    .n(ArrayUtil.prod(shape))
                    .extraArgs(extraArgs)
                    .result(arrInfo)
                    .build();


        }
        else {
            opState =  OpState.builder()
                    .opType(opType)
                    .opName(opName).inPlace(inPlace)
                    .differentialFunction(this)
                    .id(opName + "(" + v1.getVertex().getValue().getId() + " -> " + newVertex.getValue().getId() + ")")
                    .vertexIds(new String[]{String.valueOf(v2VertexId),String.valueOf(newVertex.vertexID())})
                    .n(ArrayUtil.prod(shape))
                    .extraArgs(extraArgs)
                    .result(arrInfo)
                    .build();
        }

        opState2 = OpState.builder()
                .opType(opType).inPlace(inPlace)
                .opName(opName).result(arrInfo)
                .id(opName + "(" + v1.getVertex().getValue().getId() + " -> " + newVertex.getValue().getId() + ")")
                .vertexIds(new String[]{String.valueOf(v1VertexId),String.valueOf(newVertex.vertexID())})
                .n(ArrayUtil.prod(shape))
                .extraArgs(extraArgs)
                .differentialFunction(this)
                .result(arrInfo)
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



    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction i_v1,
                            DifferentialFunction i_v2,
                            String opName) {
        validateDifferentialFunctionGraph(i_v1);
        validateDifferentialFunctionGraph(i_v2);
        validateFunctionReference(i_v1);
        validateFunctionReference(i_v2);


        ArrayField arrayField = i_v1.getValue(true);
        addEdges(sameDiff,
                i_v1,
                i_v2,
                opName,
                OpState.OpType.TRANSFORM,
                arrayField.getInput().getShape());


    }




    /**
     * Duplicate this function
     * @return
     */
    public abstract DifferentialFunction dup();


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

    protected void validateDifferentialFunctionsameDiff(
            ArrayField function) {
        if(sameDiff.getGraph().isFrozen())
            return;
        Preconditions.checkState(function != null,"Passed in function was null.");
        Preconditions.checkState(function.getOps() ==
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
        ArrayField a = getValue(true);
        Preconditions.checkState(a.getOps() == sameDiff);
        Preconditions.checkState(a.getOps() == function.getSameDiff());

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


}
