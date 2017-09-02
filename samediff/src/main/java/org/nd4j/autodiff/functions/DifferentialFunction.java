package org.nd4j.autodiff.functions;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import com.google.common.base.Preconditions;
import lombok.*;
import org.nd4j.autodiff.ArrayFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.functions.impl.binary.transform.Add;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;


@AllArgsConstructor
@Data
@NoArgsConstructor
public abstract class DifferentialFunction<X extends Field<X>>
        implements Field<DifferentialFunction<ArrayField>>,
        Differential<ArrayField, DifferentialFunction<ArrayField>> {

    @Getter
    @Setter
    protected SameDiff sameDiff;
    @Getter
    protected OpState opState;
    @Getter
    @Setter
    protected int vertexId;
    @Getter
    @Setter
    protected DifferentialFunction<ArrayField> gradient;

    protected Object[] extraArgs;




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


    public  boolean isVariable() {
        return false;
    }

    /**
     * Get the value of this function
     * @return
     */
    public abstract X doGetValue();


    /**
     * Shortcut for the {@link DifferentialFunctionFactory}
     * @return
     */
    public DifferentialFunctionFactory<ArrayField> f() {
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
    public  X getValue(boolean freeze) {
        boolean graphAlreadyFrozen = this.sameDiff.getGraph().isFrozen();
        //if graph is already frozen leave it frozen
        if(freeze && !graphAlreadyFrozen) {
            this.sameDiff.getGraph().freeze();
        }

        X val = doGetValue();
        if(val instanceof ArrayField) {
            ArrayField arrayField = sameDiff.setupArrayField((ArrayField) val);
            val = (X) arrayField;
            Preconditions.checkState(arrayField.getOps() == this.sameDiff,"Same diff instances for get value not the same.");

        }

        if(val instanceof ArrayField && !freeze) {
            ArrayField arrayField = sameDiff.setupArrayField((ArrayField) val);
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

        return (X) sameDiff.setupArrayField((ArrayField) val);
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



    @Override
    public abstract List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1);

    private void validateDifferentialFunctionGraph(DifferentialFunction<ArrayField> function) {
        Preconditions.checkState(function.getSameDiff() == this.getSameDiff(),"Function applications must be contained in same graph. The left " + function +" must match this function " + this);

    }


    @Override
    public DifferentialFunction<ArrayField> rdivi(DifferentialFunction<ArrayField> i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> rsubi(DifferentialFunction<ArrayField> i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> addi(DifferentialFunction<ArrayField> i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> muli(DifferentialFunction<ArrayField> i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> subi(DifferentialFunction<ArrayField> i_v) {
        return null;

    }




    @Override
    public DifferentialFunction<ArrayField> divi(DifferentialFunction<ArrayField> i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> inversei() {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction<ArrayField> negatei() {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> muli(double i_n) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction<ArrayField> powi(int i_n) {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction<ArrayField> addi(double i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> subi(double i_v) {
        return null;

    }



    @Override
    public DifferentialFunction<ArrayField> divi(double v) {
        return null;

    }


    @Override
    public DifferentialFunction<ArrayField> rsubi(double v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> rdivi(double v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> set(DifferentialFunction<ArrayField> i_v) {
        return null;


    }


    @Override
    public DifferentialFunction<ArrayField> rdiv(DifferentialFunction<ArrayField> i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> rsub(DifferentialFunction<ArrayField> i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> add(DifferentialFunction<ArrayField> i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> mul(DifferentialFunction<ArrayField> i_v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> sub(DifferentialFunction<ArrayField> i_v) {
        return null;
    }




    @Override
    public DifferentialFunction<ArrayField> div(DifferentialFunction<ArrayField> i_v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> inverse() {
        throw new UnsupportedOperationException();

    }

    @Override
    public DifferentialFunction<ArrayField> negate() {
        DifferentialFunction<ArrayField> ret = new Negative(sameDiff,this.mul(1.0));
        return ret;
    }

    @Override
    public DifferentialFunction<ArrayField> mul(double i_n) {
       return null;
    }

    @Override
    public DifferentialFunction<ArrayField> pow(int i_n) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DifferentialFunction<ArrayField> add(double i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> sub(double i_v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> rsub(double v) {
        return null;

    }

    @Override
    public DifferentialFunction<ArrayField> rdiv(double v) {
        return null;

    }


    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction<ArrayField> i_v1,
                            DifferentialFunction<ArrayField> i_v2,
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

    @Override
    public ArrayField logSoftmax() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> tanhDerivative(DifferentialFunction<ArrayField> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> seluDerivative(DifferentialFunction<ArrayField> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> softmaxDerivative(ArrayField wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField>[] args() {
        return new DifferentialFunction[0];
    }

    @Override
    public DifferentialFunction<ArrayField> pow(DifferentialFunction<ArrayField> a) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> floor() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> ceil() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> round() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> abs() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> sqrt() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> minus(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> prod(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> div(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> pow(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> cos() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> acos() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> cosh() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> acosh() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> sin() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> asin() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> sinh() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> asinh() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> tan() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> atan() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> tanh() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> atanh() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> exp() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> log() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> log10() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> sgn() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> pwr(DifferentialFunction<ArrayField> y) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> pwrs(DifferentialFunction<ArrayField> y) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> square() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> relu() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> hardTanh() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> hardTanhDerivative(DifferentialFunction<ArrayField> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> leakyRelu() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> elu() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> eluDerivative(DifferentialFunction<ArrayField> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> leakyRelu(double cutoff) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> leakyReluDerivative() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> leakyReluDerivative(DifferentialFunction<ArrayField> wrt, double cutoff) {
        return null;
    }
    @Override
    public DifferentialFunction<ArrayField> selu() {
        return null;
    }
    @Override
    public DifferentialFunction<ArrayField> sigmoid() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> sigmoidDerivative(DifferentialFunction<ArrayField> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> step() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> softsign() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> softsignDerivative(DifferentialFunction<ArrayField> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> softmax() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> softplus() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> reshape(int[] shape) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> transpose() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> permute(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> expandDims(int dim) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> sum(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> prod(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> mean(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> std(int[] dimensions, boolean biasCorrected) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> variance(int[] dimensions, boolean biasCorrected) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> std(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> variance(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> max(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> min(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> norm1(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> norm2(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> normmax(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> valueArrayOf(int[] shape) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> tile(int[] repeat) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> repeat(int axis) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> broadcast(int[] shape) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> eq(DifferentialFunction<ArrayField> i_y) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> neq(DifferentialFunction<ArrayField> i_y) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> or(DifferentialFunction<ArrayField> i_y) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> rollAxis(int axis) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> cosineSimilarity(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> euclideanDistance(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> manhattanDistance(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossBinaryXENT(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossCosineSimilarity(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossHinge(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossKLD(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossL1(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossL2(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossMAE(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossMAPE(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossMSE(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossMCXENT(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossMSLE(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossNegativeLogLikelihood(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossPoisson(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction arg() {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> max(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> min(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> fmod(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> set(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<ArrayField> lossSquaredHinge(DifferentialFunction<ArrayField> i_y, int... dimensions) {
        return null;
    }

    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction<ArrayField> i_v1,
                            DifferentialFunction<ArrayField> i_v2,
                            String opName,
                            OpState.OpType opType,
                            int[] shape, Object[] extraArgs) {
        validateFunctionReference(i_v1);
        validateFunctionReference(i_v2);

        if(i_v1.getValue(true) instanceof ArrayField) {
            validateDifferentialFunctionGraph(i_v1);
            validateDifferentialFunctionGraph(i_v2);
            validateDifferentialFunctionsameDiff((ArrayField) i_v1.getValue(true));


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
            ArrayField v1 = (ArrayField) i_v1.getValue(true);
            int v1VertexId = i_v1.resultVertexId();
            ArrayField v2 = (ArrayField) i_v2.getValue(true);
            int v2VertexId = i_v2.resultVertexId();
            validateDifferentialFunctionsameDiff(v1);
            validateDifferentialFunctionsameDiff(v2);

            NDArrayInformation arrInfo = NDArrayInformation.builder()
                    .arrId(UUID.randomUUID().toString())
                    .id(opName +"(" + v1.getInput().getId() + "," + v2.getInput().getId() + ")")
                    .shape(shape).build();
            //result
            NDArrayVertex newVertex = new NDArrayVertex(sameDiff,sameDiff.getGraph().nextVertexId(), arrInfo);
            if(newVertex.vertexID() == v2VertexId || newVertex.vertexID() == v1VertexId)
                throw new ND4JIllegalStateException("Illegal vertex id specified in new vertex." +
                        " Perhaps a mismatched graph call? Another likely cause is applyGraph");
            this.vertexId = newVertex.vertexID();
            //add the result vertex
            sameDiff.getGraph().addVertex(newVertex);
            OpState opState,opState2;


            //ensure there's 2 vertices for when the 2 inputs are the same
            if(v1.equals(v2)) {
                NDArrayVertex dupVertex = new NDArrayVertex(sameDiff,sameDiff.getGraph().nextVertexId(),
                        NDArrayInformation.builder()
                                .shape(v1.getInput().getShape())
                                .id(v1.getInput().getId()).build());
                //update vertex id
                v2VertexId = dupVertex.vertexID();
                sameDiff.getGraph().addVertex(dupVertex);
                opState = OpState.builder()
                        .opType(opType)
                        .differentialFunction((DifferentialFunction<ArrayField>) this)
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
                        .opName(opName)
                        .differentialFunction((DifferentialFunction<ArrayField>) this)
                        .id(opName + "(" + v1.getVertex().getValue().getId() + " -> " + newVertex.getValue().getId() + ")")
                        .vertexIds(new String[]{String.valueOf(v2VertexId),String.valueOf(newVertex.vertexID())})
                        .n(ArrayUtil.prod(shape))
                        .extraArgs(extraArgs)
                        .result(arrInfo)
                        .build();
            }

            opState2 = OpState.builder()
                    .opType(opType)
                    .opName(opName).result(arrInfo)
                    .id(opName + "(" + v1.getVertex().getValue().getId() + " -> " + newVertex.getValue().getId() + ")")
                    .vertexIds(new String[]{String.valueOf(v1VertexId),String.valueOf(newVertex.vertexID())})
                    .n(ArrayUtil.prod(shape))
                    .extraArgs(extraArgs)
                    .differentialFunction((DifferentialFunction<ArrayField>) this)
                    .result(arrInfo)
                    .build();
            //add the first vertex no matter what as normal
            sameDiff.getGraph().addEdge(
                    v1VertexId,
                    newVertex.vertexID(),
                    opState2,true);

            sameDiff.getGraph().addEdge(v2VertexId,
                    newVertex.vertexID(),
                    opState
                    ,true);
            newVertex.setOpState(opState2);
            arrInfo.setOwner(opState2);

            this.opState = opState;

        }


    }



    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction<ArrayField> i_v1,
                            DifferentialFunction<ArrayField> i_v2,
                            String opName) {
        validateDifferentialFunctionGraph(i_v1);
        validateDifferentialFunctionGraph(i_v2);
        validateFunctionReference(i_v1);
        validateFunctionReference(i_v2);

        if(i_v1.getValue(true) instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) i_v1.getValue(true);
            addEdges(sameDiff,
                    i_v1,
                    i_v2,
                    opName,
                    OpState.OpType.TRANSFORM,
                    arrayField.getInput().getShape());

        }

        else
            throw new UnsupportedOperationException("Only supporting array fields");
    }

    protected boolean getInPlace(Object[] extraArgs) {
        if(extraArgs == null) {
            return false;
        }
        else {
            for(int i = 0; i < extraArgs.length; i++) {
                if(extraArgs[i] instanceof Boolean)
                    return (Boolean) extraArgs[i];
            }
        }

        return false;
    }


    /**
     * Duplicate this function
     * @return
     */
    public abstract DifferentialFunction<ArrayField> dup();


    protected void validateFunctionReference(List<DifferentialFunction<ArrayField>> reference) {
        for(int i = 0; i < reference.size(); i++) {
            validateFunctionReference(reference.get(i));
        }

    }
    protected void validateFunctionReference(DifferentialFunction<ArrayField> reference) {
        if(sameDiff.getFunctionInstances().containsKey(reference.getVertexId()))
            Preconditions.checkState(reference == sameDiff.getFunctionInstances().get(reference.getVertexId()),"Found invalid reference " + reference + " for vertex id " + reference.getVertexId());


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


    public DifferentialFunction<ArrayField> getDiffFunctionInput(DifferentialFunction<ArrayField> other) {
        return   other == this ?
                sameDiff.getFunctionFactory().var(UUID.randomUUID().toString(),
                        sameDiff.getArrayFactory().one(getResultShape())) :
                arg();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DifferentialFunction<?> that = (DifferentialFunction<?>) o;

        if (vertexId != that.vertexId) return false;
        if (opState != null ? !opState.equals(that.opState) : that.opState != null) return false;
        if (gradient != null ? !gradient.equals(that.gradient) : that.gradient != null) return false;
        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (opState != null ? opState.hashCode() : 0);
        result = 31 * result + vertexId;
        result = 31 * result + (gradient != null ? gradient.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(extraArgs);
        return result;
    }

    @Override
    public double getReal() {
        throw new UnsupportedOperationException("Get real not supported for array operations");
    }


    protected void validateDifferentialFunctionsameDiff(
            List<DifferentialFunction<ArrayField>> function) {
        for(DifferentialFunction<ArrayField> differentialFunction : function)
            validateDifferentialFunctionsameDiff(differentialFunction);
    }

    protected void validateDifferentialFunctionsameDiff(
            DifferentialFunction<ArrayField> function) {

        Preconditions.checkState(function != null,"Passed in function was null.");
        ArrayField a = (ArrayField)  getValue(true);
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
