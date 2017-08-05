package org.nd4j.autodiff.functions;

import java.util.List;

import lombok.*;
import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
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
    protected SDGraph graph;
    @Getter
    protected OpState opState;
    @Getter
    @Setter
    protected int vertexId;
    protected Object[] extraArgs;


    public DifferentialFunction(SDGraph graph, Object[] extraArgs) {
        this.graph = graph;
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
    protected abstract X doGetValue();




    /**
     * Get the value specifying
     * whether to freeze the graph or not
     * @param freeze whether to freeze the graph or not,
     *               this means whether to add nodes to the internal
     *               computation graph or not
     * @return the value of this function
     */
    public  X getValue(boolean freeze) {
        boolean graphAlreadyFrozen = this.graph.isFrozen();
        //if graph is already frozen leave it frozen
        if(freeze && !graphAlreadyFrozen) {
            this.graph.freeze();
        }

        X val = doGetValue();

        if(freeze && !graphAlreadyFrozen) {
            this.graph.unfreeze();
        }

        return val;
    }

    @Override
    public abstract double getReal();


    @Override
    public  String getFormula(List<Variable<X>> variables) {
        graph.freeze();
        String ret = doGetFormula(variables);
        graph.unfreeze();
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
    public abstract DifferentialFunction<X> diff(Variable<X> i_v1);



    @Override
    public DifferentialFunction<X> rdivi(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(false).rdivi(getValue(false));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }

    @Override
    public DifferentialFunction<X> rsubi(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(false).rsubi(getValue(false));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }

    @Override
    public DifferentialFunction<X> addi(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(false).addi(getValue(false));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }

    @Override
    public DifferentialFunction<X> muli(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(false).muli(getValue(false));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }

    @Override
    public DifferentialFunction<X> subi(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(false).subi(getValue(false));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }




    @Override
    public DifferentialFunction<X> divi(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(false).divi(getValue(false));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }

    @Override
    public DifferentialFunction<X> inversei() {
        DifferentialFunction<X> ret = new Inverse<>(graph,this,true);
        return ret;
    }

    @Override
    public DifferentialFunction<X> negatei() {
        DifferentialFunction<X> ret = new Negative<>(graph,this,true);
        return ret;
    }

    @Override
    public DifferentialFunction<X> muli(double i_n) {
        PolynomialTerm<X> ret =  new PolynomialTerm<>(graph,i_n, this, 1,true);
        return ret;
    }

    @Override
    public DifferentialFunction<X> powi(int i_n) {
        PolynomialTerm<X> ret = new PolynomialTerm<>(graph,1L, this, i_n,true);
        return ret;
    }

    @Override
    public DifferentialFunction<X> addi(double i_v) {
        Scalar<X> constant = new Scalar<>(graph, i_v, (AbstractIdentityFactory<X>) this.graph.getSameDiff().getArrayFactory(),true);
        return constant.add(this);
    }

    @Override
    public DifferentialFunction<X> subi(double i_v) {
        Scalar<X> constant = new Scalar<>(graph, i_v, (AbstractIdentityFactory<X> ) this.graph.getSameDiff().getArrayFactory(),true);
        return constant.sub(this);
    }



    @Override
    public DifferentialFunction<X> divi(double v) {
        Scalar<X> constant = new Scalar<>(graph, v, (AbstractIdentityFactory<X> ) this.graph.getSameDiff().getArrayFactory(),true);
        return this.divi(constant);
    }


    @Override
    public DifferentialFunction<X> rsubi(double v) {
        Scalar<X> constant = new Scalar<>(graph, v, (AbstractIdentityFactory<X> ) this.graph.getSameDiff().getArrayFactory(),true);
        return this.rsub(constant);
    }

    @Override
    public DifferentialFunction<X> rdivi(double v) {
        Scalar<X> constant = new Scalar<>(graph, v, (AbstractIdentityFactory<X> ) this.graph.getSameDiff().getArrayFactory(),true);
        return this.rdiv(constant);
    }




    @Override
    public DifferentialFunction<X> rdiv(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = this.div(i_v);
        return ret;
    }

    @Override
    public DifferentialFunction<X> rsub(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = i_v.sub(this);
        return ret;
    }

    @Override
    public DifferentialFunction<X> add(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(true).add(getValue(true));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }

    @Override
    public DifferentialFunction<X> mul(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(true).mul(getValue(true));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }

    @Override
    public DifferentialFunction<X> sub(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(true).sub(getValue(true));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }




    @Override
    public DifferentialFunction<X> div(DifferentialFunction<X> i_v) {
        X ret = i_v.getValue(true).div(getValue(true));
        return new Constant<>(graph, ret, i_v.getResultShape(), (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
    }

    @Override
    public DifferentialFunction<X> inverse() {
        DifferentialFunction<X> ret = new Inverse<>(graph,this);
        return ret;
    }

    @Override
    public DifferentialFunction<X> negate() {
        DifferentialFunction<X> ret = new Negative<>(graph,this);
        return ret;
    }

    @Override
    public DifferentialFunction<X> mul(double i_n) {
        PolynomialTerm<X> ret =  new PolynomialTerm<>(graph,i_n, this, 1);
        return ret;
    }

    @Override
    public DifferentialFunction<X> pow(int i_n) {
        PolynomialTerm<X> ret = new PolynomialTerm<>(graph,1L, this, i_n);
        return ret;
    }

    @Override
    public DifferentialFunction<X> add(double i_v) {
        Scalar<X> constant = new Scalar<>(graph, i_v, (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
        return constant.add(this);
    }

    @Override
    public DifferentialFunction<X> sub(double i_v) {
        Scalar<X> constant = new Scalar<>(graph, i_v, (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
        return constant.sub(this);
    }

    @Override
    public DifferentialFunction<X> rsub(double v) {
        Scalar<X> constant = new Scalar<>(graph, v, (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
        return this.rsub(constant);
    }

    @Override
    public DifferentialFunction<X> rdiv(double v) {
        Scalar<X> constant = new Scalar<>(graph, v, (AbstractIdentityFactory<X>) graph.getSameDiff().getArrayFactory());
        return this.rdiv(constant);
    }

    protected void addEdges(Graph<NDArrayInformation,OpState> graph,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName,
                            OpState.OpType opType,
                            int[] shape) {
        addEdges(graph,
                i_v1,
                i_v2,
                opName,
                opType,
                shape,
                null);

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
    public DifferentialFunction<X> hardTanhDerivative() {
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
    public DifferentialFunction<X> eluDerivative() {
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
    public DifferentialFunction<X> leakyReluDerivative(double cutoff) {
        return null;
    }

    @Override
    public DifferentialFunction<X> sigmoid() {
        return null;
    }

    @Override
    public DifferentialFunction<X> sigmoidDerivative() {
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
    public DifferentialFunction<X> softsignDerivative() {
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

    protected void addEdges(Graph<NDArrayInformation,OpState> graph,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName,
                            OpState.OpType opType,
                            int[] shape, Object[] extraArgs) {
        if(i_v1.getValue(true) instanceof ArrayField) {
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

            NDArrayInformation arrInfo = NDArrayInformation.builder()
                    .id(opName +"(" + v1.getInput().getId() + "," + v2.getInput().getId() + ")")
                    .shape(shape).build();
            //result
            NDArrayVertex newVertex = new NDArrayVertex(graph.nextVertexId(), arrInfo);
            if(newVertex.vertexID() == v2VertexId || newVertex.vertexID() == v1VertexId)
                throw new ND4JIllegalStateException("Illegal vertex id specified in new vertex." +
                        " Perhaps a mismatched graph call? Another likely cause is applyGraph");
            this.vertexId = newVertex.vertexID();
            //add the result vertex
            graph.addVertex(newVertex);
            OpState opState,opState2;


            //ensure there's 2 vertices for when the 2 inputs are the same
            if(v1.equals(v2)) {
                NDArrayVertex dupVertex = new NDArrayVertex(graph.nextVertexId(),
                        NDArrayInformation.builder()
                                .shape(v1.getInput().getShape())
                                .id(v1.getInput().getId()).build());
                //update vertex id
                v2VertexId = dupVertex.vertexID();
                graph.addVertex(dupVertex);
                opState = OpState.builder()
                        .opType(opType)
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
                    .result(arrInfo)
                    .build();
            //add the first vertex no matter what as normal
            graph.addEdge(v1VertexId,
                    newVertex.vertexID(),
                    opState2,true);

            graph.addEdge(v2VertexId,
                    newVertex.vertexID(),
                    opState
                    ,true);
            newVertex.setOpState(opState2);
            arrInfo.setOwner(opState2);

            this.opState = opState;

        }


    }



    protected void addEdges(Graph<NDArrayInformation,OpState> graph,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName) {
        if(i_v1.getValue(true) instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) i_v1.getValue(true);
            addEdges(graph,
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


}
