package org.nd4j.autodiff.functions;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import com.google.common.base.Preconditions;
import lombok.*;
import org.nd4j.autodiff.ArrayFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
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
        implements Field<DifferentialFunction<X>>,
        Differential<X, DifferentialFunction<X>> {

    @Getter
    @Setter
    protected SameDiff sameDiff;
    @Getter
    protected OpState opState;
    @Getter
    @Setter
    protected int vertexId;
    protected Object[] extraArgs;


    @Override
    public double getReal() {
        throw new UnsupportedOperationException("Get real not supported for array operations");
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
            ArrayField arrayField = (ArrayField) val;
            Preconditions.checkState(arrayField.getOps() == this.sameDiff,"Same diff instances for get value not the same.");

        }

        if(val instanceof ArrayField && !freeze) {
            ArrayField arrayField = (ArrayField) val;
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

        return val;
    }


    @Override
    public  String getFormula(List<Variable<X>> variables) {
        sameDiff.getGraph().freeze();
        String ret = doGetFormula(variables);
        sameDiff.getGraph().unfreeze();
        return ret;
    }

    public abstract String doGetFormula(List<Variable<X>> variables);

    public abstract String functionName();

    @Override
    public abstract String toString();



    public boolean isConstant() {
        return false;
    }


    public boolean isVariable() {
        return false;
    }

    @Override
    public abstract List<DifferentialFunction<X>> diff(List<DifferentialFunction<X>> i_v1);

    private void validateDifferentialFunctionGraph(DifferentialFunction<X> function) {
        Preconditions.checkState(function.getSameDiff() == this.getSameDiff(),"Function applications must be contained in same graph. The left " + function +" must match this function " + this);

    }


    @Override
    public DifferentialFunction<X> rdivi(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).rdivi(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }

    @Override
    public DifferentialFunction<X> rsubi(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).rsubi(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }

    @Override
    public DifferentialFunction<X> addi(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).addi(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }

    @Override
    public DifferentialFunction<X> muli(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).muli(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }

    @Override
    public DifferentialFunction<X> subi(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).subi(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }




    @Override
    public DifferentialFunction<X> divi(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).divi(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }

    @Override
    public DifferentialFunction<X> inversei() {
        DifferentialFunction<X> ret = new Inverse<>(sameDiff,this,true);
        return ret;
    }

    @Override
    public DifferentialFunction<X> negatei() {
        DifferentialFunction<X> ret = new Negative<>(sameDiff,this,true);
        return ret;
    }

    @Override
    public DifferentialFunction<X> muli(double i_n) {
        PolynomialTerm<X> ret =  new PolynomialTerm<>(sameDiff,i_n, this, 1,true);
        return ret;
    }

    @Override
    public DifferentialFunction<X> powi(int i_n) {
        PolynomialTerm<X> ret = new PolynomialTerm<>(sameDiff,1L,
                this, i_n,true);
        return ret;
    }

    @Override
    public DifferentialFunction<X> addi(double i_v) {
        Scalar<X> constant = new Scalar<>(sameDiff, i_v,true);
        return constant.addi(this);
    }

    @Override
    public DifferentialFunction<X> subi(double i_v) {
        Scalar<X> constant = new Scalar<>(sameDiff, i_v,true);
        return constant.subi(this);
    }



    @Override
    public DifferentialFunction<X> divi(double v) {
        Scalar<X> constant = new Scalar<>(sameDiff, 
                v,true);
        return this.divi(constant);
    }


    @Override
    public DifferentialFunction<X> rsubi(double v) {
        Scalar<X> constant = new Scalar<>(sameDiff, v,true);
        return this.rsubi(constant);
    }

    @Override
    public DifferentialFunction<X> rdivi(double v) {
        Scalar<X> constant = new Scalar<>(sameDiff, v,true);
        return this.rdivi(constant);
    }

    @Override
    public DifferentialFunction<X> set(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = getValue(true).set(i_v.getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());

    }


    @Override
    public DifferentialFunction<X> rdiv(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).rdiv(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());

    }

    @Override
    public DifferentialFunction<X> rsub(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).rsub(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());

    }

    @Override
    public DifferentialFunction<X> add(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).add(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }

    @Override
    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).mul(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }

    @Override
    public DifferentialFunction<X> sub(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).sub(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }




    @Override
    public DifferentialFunction<X> div(DifferentialFunction<X> i_v) {
        validateDifferentialFunctionGraph(i_v);
        X ret = i_v.getValue(true).div(getValue(true));
        return new Constant<>(sameDiff, ret, i_v.getResultShape());
    }

    @Override
    public DifferentialFunction<X> inverse() {
        DifferentialFunction<X> ret = new Inverse<>(sameDiff,this.mul(1.0));
        return ret;
    }

    @Override
    public DifferentialFunction<X> negate() {
        DifferentialFunction<X> ret = new Negative<>(sameDiff,this.mul(1.0));
        return ret;
    }

    @Override
    public DifferentialFunction<X> mul(double i_n) {
        Scalar<X> constant = new Scalar<>(sameDiff, i_n);
        return this.mul(constant);
    }

    @Override
    public DifferentialFunction<X> pow(int i_n) {
        PolynomialTerm<X> ret = new PolynomialTerm<>(sameDiff,1L, this, i_n);
        return ret;
    }

    @Override
    public DifferentialFunction<X> add(double i_v) {
        Scalar<X> constant = new Scalar<>(sameDiff, i_v);
        return constant.add(this);
    }

    @Override
    public DifferentialFunction<X> sub(double i_v) {
        Scalar<X> constant = new Scalar<>(sameDiff, i_v);
        return constant.sub(this);
    }

    @Override
    public DifferentialFunction<X> rsub(double v) {
        Scalar<X> constant = new Scalar<>(sameDiff, v);
        return this.rsub(constant);
    }

    @Override
    public DifferentialFunction<X> rdiv(double v) {
        Scalar<X> constant = new Scalar<>(sameDiff, v);
        return this.rdiv(constant);
    }

    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
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
    public DifferentialFunction<X> tanhDerivative(DifferentialFunction<X> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<X> seluDerivative(DifferentialFunction<X> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<X> softmaxDerivative(ArrayField wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<X>[] args() {
        return new DifferentialFunction[0];
    }

    @Override
    public DifferentialFunction<X> pow(DifferentialFunction<X> a) {
        return null;
    }

    @Override
    public DifferentialFunction<X> floor() {
        return null;
    }

    @Override
    public DifferentialFunction<X> ceil() {
        return null;
    }

    @Override
    public DifferentialFunction<X> round() {
        return null;
    }

    @Override
    public DifferentialFunction<X> abs() {
        return null;
    }

    @Override
    public DifferentialFunction<X> sqrt() {
        return null;
    }

    @Override
    public DifferentialFunction<X> minus(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<X> prod(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<X> div(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<X> pow(double v) {
        return null;
    }

    @Override
    public DifferentialFunction<X> cos() {
        return null;
    }

    @Override
    public DifferentialFunction<X> acos() {
        return null;
    }

    @Override
    public DifferentialFunction<X> cosh() {
        return null;
    }

    @Override
    public DifferentialFunction<X> acosh() {
        return null;
    }

    @Override
    public DifferentialFunction<X> sin() {
        return null;
    }

    @Override
    public DifferentialFunction<X> asin() {
        return null;
    }

    @Override
    public DifferentialFunction<X> sinh() {
        return null;
    }

    @Override
    public DifferentialFunction<X> asinh() {
        return null;
    }

    @Override
    public DifferentialFunction<X> tan() {
        return null;
    }

    @Override
    public DifferentialFunction<X> atan() {
        return null;
    }

    @Override
    public DifferentialFunction<X> tanh() {
        return null;
    }

    @Override
    public DifferentialFunction<X> atanh() {
        return null;
    }

    @Override
    public DifferentialFunction<X> exp() {
        return null;
    }

    @Override
    public DifferentialFunction<X> log() {
        return null;
    }

    @Override
    public DifferentialFunction<X> log10() {
        return null;
    }

    @Override
    public DifferentialFunction<X> sgn() {
        return null;
    }

    @Override
    public DifferentialFunction<X> pwr(DifferentialFunction<X> y) {
        return null;
    }

    @Override
    public DifferentialFunction<X> pwrs(DifferentialFunction<X> y) {
        return null;
    }

    @Override
    public DifferentialFunction<X> square() {
        return null;
    }

    @Override
    public DifferentialFunction<X> relu() {
        return null;
    }

    @Override
    public DifferentialFunction<X> hardTanh() {
        return null;
    }

    @Override
    public DifferentialFunction<X> hardTanhDerivative(DifferentialFunction<X> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<X> leakyRelu() {
        return null;
    }

    @Override
    public DifferentialFunction<X> elu() {
        return null;
    }

    @Override
    public DifferentialFunction<X> eluDerivative(DifferentialFunction<X> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<X> leakyRelu(double cutoff) {
        return null;
    }

    @Override
    public DifferentialFunction<X> leakyReluDerivative() {
        return null;
    }

    @Override
    public DifferentialFunction<X> leakyReluDerivative(DifferentialFunction<X> wrt, double cutoff) {
        return null;
    }
    @Override
    public DifferentialFunction<X> selu() {
        return null;
    }
    @Override
    public DifferentialFunction<X> sigmoid() {
        return null;
    }

    @Override
    public DifferentialFunction<X> sigmoidDerivative(DifferentialFunction<X> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<X> step() {
        return null;
    }

    @Override
    public DifferentialFunction<X> softsign() {
        return null;
    }

    @Override
    public DifferentialFunction<X> softsignDerivative(DifferentialFunction<X> wrt) {
        return null;
    }

    @Override
    public DifferentialFunction<X> softmax() {
        return null;
    }

    @Override
    public DifferentialFunction<X> softplus() {
        return null;
    }

    @Override
    public DifferentialFunction<X> reshape(int[] shape) {
        return null;
    }

    @Override
    public DifferentialFunction<X> transpose() {
        return null;
    }

    @Override
    public DifferentialFunction<X> permute(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> expandDims(int dim) {
        return null;
    }

    @Override
    public DifferentialFunction<X> sum(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> prod(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> mean(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> std(int[] dimensions, boolean biasCorrected) {
        return null;
    }

    @Override
    public DifferentialFunction<X> variance(int[] dimensions, boolean biasCorrected) {
        return null;
    }

    @Override
    public DifferentialFunction<X> std(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> variance(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> max(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> min(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> norm1(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> norm2(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> normmax(int[] dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> valueArrayOf(int[] shape) {
        return null;
    }

    @Override
    public DifferentialFunction<X> tile(int[] repeat) {
        return null;
    }

    @Override
    public DifferentialFunction<X> repeat(int axis) {
        return null;
    }

    @Override
    public DifferentialFunction<X> broadcast(int[] shape) {
        return null;
    }

    @Override
    public DifferentialFunction<X> eq(DifferentialFunction<X> i_y) {
        return null;
    }

    @Override
    public DifferentialFunction<X> neq(DifferentialFunction<X> i_y) {
        return null;
    }

    @Override
    public DifferentialFunction<X> or(DifferentialFunction<X> i_y) {
        return null;
    }

    @Override
    public DifferentialFunction<X> rollAxis(int axis) {
        return null;
    }

    @Override
    public DifferentialFunction<X> cosineSimilarity(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> euclideanDistance(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> manhattanDistance(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossBinaryXENT(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossCosineSimilarity(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossHinge(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossKLD(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossL1(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossL2(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossMAE(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossMAPE(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossMSE(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossMCXENT(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossMSLE(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossNegativeLogLikelihood(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossPoisson(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    @Override
    public DifferentialFunction<X> lossSquaredHinge(DifferentialFunction<X> i_y, int... dimensions) {
        return null;
    }

    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName,
                            OpState.OpType opType,
                            int[] shape, Object[] extraArgs) {
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
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName) {
        validateDifferentialFunctionGraph(i_v1);
        validateDifferentialFunctionGraph(i_v2);
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


    public abstract DifferentialFunction<X> dup();

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


    public DifferentialFunction<ArrayField> getDiffFunctionInput(DifferentialFunction<X> other) {
      return   other == this ?
                sameDiff.getFunctionFactory().var(UUID.randomUUID().toString(),
                        sameDiff.getArrayFactory().one(getResultShape())) :
                arg();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        DifferentialFunction<?> that = (DifferentialFunction<?>) o;

        if (vertexId != that.vertexId) return false;
        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        return Arrays.equals(extraArgs, that.extraArgs);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + vertexId;
        result = 31 * result + Arrays.hashCode(extraArgs);
        return result;
    }

    protected void validateDifferentialFunctionsameDiff(
            List<DifferentialFunction<X>> function) {
        for(DifferentialFunction<X> differentialFunction : function)
            validateDifferentialFunctionsameDiff(differentialFunction);
    }

    protected void validateDifferentialFunctionsameDiff(
            DifferentialFunction<X> function) {

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
