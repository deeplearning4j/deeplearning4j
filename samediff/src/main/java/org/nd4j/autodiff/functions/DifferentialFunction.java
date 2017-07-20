package org.nd4j.autodiff.functions;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
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

    protected SDGraph graph;
    @Getter
    protected OpState opState;
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
    public DifferentialFunction<X> rdiv(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = this.div(i_v);
        return ret;
    }

    @Override
    public DifferentialFunction<X> rsub(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = this.sub(i_v);
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
        DifferentialFunction<X> ret = add(i_v.negate());
        return ret;
    }




    @Override
    public DifferentialFunction<X> div(DifferentialFunction<X> i_v) {
        DifferentialFunction<X> ret = mul(i_v.inverse());
        return ret;
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
        Scalar<X> constant = new Scalar<>(graph, i_v, null);
        return constant.add(this);
    }

    @Override
    public DifferentialFunction<X> sub(double i_v) {
        Scalar<X> constant = new Scalar<>(graph, i_v, null);
        return constant.sub(this);
    }

    @Override
    public DifferentialFunction<X> rsub(double v) {
        Scalar<X> constant = new Scalar<>(graph, v, null);
        return this.rsub(constant);
    }

    @Override
    public DifferentialFunction<X> rdiv(double v) {
        Scalar<X> constant = new Scalar<>(graph, v, null);
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


    protected void addEdges(Graph<NDArrayInformation,OpState> graph,
                            DifferentialFunction<X> i_v1,
                            DifferentialFunction<X> i_v2,
                            String opName,
                            OpState.OpType opType,
                            int[] shape,Object[] extraArgs) {
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
                throw new ND4JIllegalStateException("Illegal vertex id specified in new vertex. Perhaps a mismatched graph call? Another likely cause is applyGraph");
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

    public abstract DifferentialFunction<X> dup();

    public  int resultVertexId() {
        return vertexId;
    }


}
