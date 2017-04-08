package org.nd4j.autodiff.autodiff;

import java.util.List;

import org.nd4j.autodiff.AbstractIdentityFactory;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.util.ArrayUtil;


public class Constant<X extends Field<X>> extends DifferentialFunction<X> {

    protected X m_x;
    private AbstractIdentityFactory<X> m_factory;

    protected Constant(Graph<NDArrayInformation,OpState> graph, X i_v, AbstractIdentityFactory<X> i_factory) {
        super(graph);
        if (i_v != null && i_factory != null) {
            m_x = i_v;
            m_factory = i_factory;
            addNode(graph);

        } else {
            throw new IllegalArgumentException("Input not null value.");
        }
    }

    protected AbstractIdentityFactory<X> factory() {
        return m_factory;
    }

    @Override
    public boolean isConstant() {
        return true;
    }

    @Override
    public X doGetValue() {
        return m_x;
    }

    @Override
    public double getReal() {
        return m_x.getReal();
    }

    @Override
    public DifferentialFunction<X> diff(Variable<X> i_v) {
        return new Zero<>(graph, m_factory);
    }

    @Override
    public String toString() {
        return getValue().toString();
    }

    @Override
    public String doGetFormula(List<Variable<X>> variables) {
        return getValue().toString();
    }

    @Override
    public String functionName() {
        return "constant";
    }

    @Override
    protected DifferentialFunction<X> plused(DifferentialFunction<X> i_v) {
        return i_v.isConstant() ? new Constant<>(graph, i_v.getValue(false).plus(this.m_x), m_factory)
                : super.plused(i_v);
    }

    @Override
    protected DifferentialFunction<X> muled(DifferentialFunction<X> i_v) {
        return i_v.isConstant() ? new Constant<>(graph, i_v.getValue(false).mul(this.m_x), m_factory)
                : super.muled(i_v);
    }

    protected void addNode(Graph<NDArrayInformation,OpState> graph) {
        if(m_x instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) m_x;
            NDArrayVertex newVertex = new NDArrayVertex(graph.getVertices().size() ,
                    NDArrayInformation.builder()
                            .id("constant(" + arrayField.getVertex().vertexID() + ")a")
                            .shape(arrayField.getInput().getShape()).build());
            graph.addVertex(newVertex);

        }
    }

    protected void addEdge(String opName) {
        if(m_x instanceof ArrayField) {
            ArrayField arrayField = (ArrayField) m_x;
            NDArrayVertex newVertex = new NDArrayVertex(graph.getVertices().size() ,
                    NDArrayInformation.builder()
                            .id(opName + "(" + arrayField.getVertex().vertexID() + ")")
                            .shape(arrayField.getInput().getShape()).build());
            graph.addVertex(newVertex);
            graph.addEdge(arrayField.getVertex().getIdx(),
                    newVertex.vertexID(),OpState.builder()
                            .n(ArrayUtil.prod(arrayField.getInput().getShape()))
                            .opName(opName)
                            .id(arrayField.getVertex().vertexID() +  "->  " + functionName() + " " +  newVertex.vertexID())
                            .vertexIds(new String[]{String.valueOf(arrayField.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                            .opType(OpState.OpType.TRANSFORM).build(),true);

        }
    }
    // public DifferentialFunction<X> inverse() {
    @Override
    public Constant<X> inverse() {
        return new Constant<>(graph, m_x.inverse(), m_factory);
    }

    // public DifferentialFunction<X> negate() {
    @Override
    public Constant<X> negate() {
        return new Constant<>(graph, m_x.negate(), m_factory);
    }

    // This class must be immutable.
    // set and assign must not be implemented.
    @SuppressWarnings("unused")
    private final void set(X i_x) {
    }

    @SuppressWarnings("unused")
    private final void assign(X i_x) {
    }

}
