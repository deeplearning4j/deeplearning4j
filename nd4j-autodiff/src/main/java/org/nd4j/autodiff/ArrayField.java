package org.nd4j.autodiff;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMultiplication;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarSubtraction;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.SubOp;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Created by agibsonccc on 4/4/17.
 */
@AllArgsConstructor
@Getter
public class ArrayField implements Field<ArrayField> {
    private Graph<NDArrayInformation,OpState> ops;
    private NDArrayInformation input;
    private NDArrayVertex vertex;

    public ArrayField(NDArrayVertex ndArrayVertex,
                      Graph<NDArrayInformation,OpState> ops) {
        this.input = ndArrayVertex.getValue();
        this.vertex = ndArrayVertex;
        this.ops = ops;
        ops.addVertex(vertex);
    }


    @Override
    public ArrayField negate() {
        return addTransformOp(new Negative().name());
    }

    @Override
    public ArrayField plus(ArrayField i_v) {
        return addPairTransformOp(new AddOp().name(),i_v);
    }



    @Override
    public ArrayField minus(ArrayField i_v) {
        return addPairTransformOp(new SubOp().name(),i_v);
    }

    @Override
    public ArrayField mul(long i_n) {
        return addScalarTransformOp(new ScalarMultiplication().name(),i_n);
    }

    @Override
    public ArrayField mul(ArrayField i_v) {
        return addPairTransformOp(new MulOp().name(),i_v);
    }

    @Override
    public ArrayField pow(int i_n) {
        return addScalarTransformOp(new Pow().name(),i_n);
    }

    @Override
    public ArrayField inverse() {
        //   return new ArrayField(InvertMatrix.invert(input,false)),ops);
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField div(ArrayField i_v) {
        return addPairTransformOp(new DivOp().name(),i_v);
    }

    @Override
    public double getReal() {
        throw new UnsupportedOperationException();
    }

    public ArrayField pow(ArrayField a) {
        return addPairTransformOp(new Pow().name(),a);
    }

    public ArrayField floor() {
        return addTransformOp(new Floor().name());
    }

    public ArrayField ceil() {
        return addTransformOp(new Ceil().name());
    }

    public ArrayField round() {
        return addTransformOp(new Round().name());
    }

    public ArrayField abs() {
        return addTransformOp(new Abs().name());
    }

    public ArrayField sqrt() {
        return addTransformOp(new Sqrt().name());
    }
    // Operators for double

    public ArrayField plus(double v) {
        return addScalarTransformOp(new ScalarAdd().name(),v);
    }

    public ArrayField minus(double v) {
        return addScalarTransformOp(new ScalarSubtraction().name(),v);
    }

    public ArrayField prod(double v) {
        return addScalarTransformOp(new ScalarMultiplication().name(),v);
    }

    public ArrayField div(double v) {
        return addScalarTransformOp(new ScalarDivision().name(),v);
    }

    public ArrayField pow(double v) {
        return addScalarTransformOp(new Pow().name(),v);
    }

    public ArrayField cos() {
        return addTransformOp(new Cos().name());
    }

    public ArrayField acos() {
        return addTransformOp(new ACos().name());
    }

    public ArrayField cosh() {
        return addTransformOp(new Cosh().name());
    }

    public ArrayField acosh() {
        //  return new ArrayField(OpState.fromOp(new INDArray(Math.log(Math.sqrt(Math.pow(x, 2) - 1) + x)),ops);
        throw new UnsupportedOperationException();

    }

    public ArrayField sin() {
        return addTransformOp(new Sin().name());
    }

    public ArrayField asin() {
        return addTransformOp(new ASin().name());
    }

    public ArrayField sinh() {
        return addTransformOp(new Sinh().name());
    }

    public ArrayField asinh() {
        //  return new ArrayField(OpState.fromOp(new INDArray(Math.log(Math.sqrt(Math.pow(x, 2) + 1) + x)),ops);
        throw new UnsupportedOperationException();

    }

    public ArrayField tan() {
        return addTransformOp(new Tan().name());
    }

    public ArrayField atan() {
        return addTransformOp(new ATan().name());
    }

    public ArrayField tanh() {
        return addTransformOp(new Tanh().name());
    }

    public ArrayField atanh() {
        return addTransformOp(new ATanh().name());
    }

    public ArrayField exp() {
        return addTransformOp(new Exp().name());
    }

    public ArrayField log() {
        return addTransformOp(new Log().name());
    }

    public ArrayField log10() {
        //return new ArrayField(OpState.fromOp(new INDArray(Math.log10(x)),ops);
        throw new UnsupportedOperationException();

    }

    public ArrayField sgn() {
        return addTransformOp(new Sign().name());
    }

    public ArrayField pwr(ArrayField y) {
        //return new ArrayField(OpState.fromOp(new INDArray(Math.pow(Math.abs(x)), y.doubleValue())),ops);
        throw new UnsupportedOperationException();
    }

    public ArrayField pwrs(ArrayField y) {
        // return new ArrayField(OpState.fromOp(new INDArray(Math.pow(Math.abs(x)), y.doubleValue()) * Math.signum(x)),ops);
        throw new UnsupportedOperationException();

    }

    public ArrayField square() {
        return mul(this);
    }

    public ArrayField relu() {
        return addTransformOp(new RectifedLinear().name());
    }

    public ArrayField hardTanh() {
        return addTransformOp(new HardTanh().name());
    }

    public ArrayField hardTanhDerivative() {
        return addTransformOp(new HardTanhDerivative().name());
    }

    public ArrayField leakyRelu() {
        return addTransformOp(new LeakyReLU().name());
    }

    public ArrayField elu() {
        return addTransformOp(new ELU().name());
    }

    public ArrayField eluDerivative() {
        return addTransformOp(new ELUDerivative().name());
    }



    public ArrayField leakyRelu(double cutoff)  {
        return addTransformOp(new LeakyReLU().name());
    }

    public ArrayField leakyReluDerivative() {
        return addTransformOp(new LeakyReLUDerivative().name());
    }

    public ArrayField leakyReluDerivative(double cutoff)  {
        return addTransformOp(new LeakyReLUDerivative().name());
    }


    public ArrayField sigmoid() {
        return addTransformOp(new Sigmoid().name());
    }

    public ArrayField sigmoidDerivative() {
        return addTransformOp(new SigmoidDerivative().name());
    }

    public ArrayField step() {
        return addTransformOp(new Step().name());
    }


    public ArrayField softsign() {
        return addTransformOp(new SoftSign().name());
    }

    public ArrayField softsignDerivative() {
        return addTransformOp(new LeakyReLUDerivative().name());
    }


    public ArrayField softmax() {
        return addTransformOp(new SoftMax().name());
    }


    public ArrayField softplus() {
        return addTransformOp(new SoftPlus().name());
    }


    private ArrayField addTransformOp(String name) {
        //result
        NDArrayVertex newVertex = new NDArrayVertex(this.ops.getVertices().size() ,
                NDArrayInformation.builder()
                        .id(name + "(" + input.getId() + ")")
                        .shape(input.getShape()).build());

        //add the result vertex to the graph
        this.getOps().addVertex(newVertex);

        //map x -> z
        this.ops.addEdge(vertex.getIdx(),
                newVertex.vertexID(),OpState.builder()
                        .n(ArrayUtil.prod(input.getShape()))
                        .opName(name)
                        .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                        .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                        .opType(OpState.OpType.TRANSFORM).build(),true);

        return new ArrayField(newVertex,ops);
    }


    private ArrayField addScalarTransformOp(String name,Number scalarValue) {
        //result
        NDArrayVertex newVertex = new NDArrayVertex(this.ops.getVertices().size(),
                NDArrayInformation.builder()
                        .id(name + "(" + input.getId() + ")")
                        .shape(input.getShape()).build());

        //add the result vertex to the graph
        this.getOps().addVertex(newVertex);

        //map x -> z
        this.ops.addEdge(vertex.getIdx(),
                newVertex.vertexID(),
                OpState.builder()
                        .n(ArrayUtil.prod(input.getShape()))
                        .opName(name)
                        .scalarValue(scalarValue)
                        .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                        .opType(OpState.OpType.SCALAR_TRANSFORM)
                        .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                        .build(),true);

        return new ArrayField(newVertex,ops);
    }

    private ArrayField addPairTransformOp(String name,ArrayField i_v) {
        //result
        NDArrayVertex newVertex = new NDArrayVertex(this.ops.getVertices().size() ,
                NDArrayInformation.builder()
                        .id(name + "(" + i_v.getVertex().getValue().getId() + ")")
                        .shape(input.getShape()).build());

        //add the result vertex to the graph
        this.getOps().addVertex(newVertex);

        //map x -> z
        this.ops.addEdge(vertex.getIdx(),
                newVertex.vertexID(),OpState.builder()
                        .n(ArrayUtil.prod(input.getShape()))
                        .opName(name)
                        .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                        .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                        .opType(OpState.OpType.TRANSFORM).build(),true);
        //map y -> z
        this.ops.addEdge(i_v.getVertex().getIdx(),
                newVertex.vertexID(),OpState.builder()
                        .n(ArrayUtil.prod(input.getShape()))
                        .opName(name)
                        .id(i_v.getVertex().getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                        .vertexIds(new String[]{String.valueOf(i_v.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                        .opType(OpState.OpType.TRANSFORM).build(),true);

        return new ArrayField(newVertex,ops);
    }


    @Override
    public String toString() {
        return "ArrayField{" +
                "input=" + input +
                '}';
    }
}
