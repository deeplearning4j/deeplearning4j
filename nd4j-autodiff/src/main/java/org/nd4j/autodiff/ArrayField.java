package org.nd4j.autodiff;

import com.google.common.base.Preconditions;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.SubOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * Created by agibsonccc on 4/4/17.
 */
@AllArgsConstructor
@Getter
@Builder
@EqualsAndHashCode
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

    public ArrayField(NDArrayInformation input,Graph<NDArrayInformation,OpState> ops) {
        this.input = input;
        this.ops = ops;
        NDArrayVertex vertex = new NDArrayVertex(ops.nextVertexId(), input);
        ops.addVertex(vertex);
        this.vertex = vertex;
    }


    @Override
    public ArrayField negate() {
        return addTransformOp(new Negative().name());
    }

    @Override
    public ArrayField add(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarAdd().name(),i_v.getInput().scalar());
        return addPairTransformOp(new AddOp().name(),i_v);
    }



    @Override
    public ArrayField sub(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarSubtraction().name(),i_v.getInput().scalar());
        return addPairTransformOp(new SubOp().name(),i_v);
    }

    @Override
    public ArrayField rsub(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarReverseSubtraction().name(),i_v.getInput().scalar());
        return addPairTransformOp("rsub",i_v);
    }

    @Override
    public ArrayField mul(double i_n) {
        return addScalarTransformOp(new ScalarMultiplication().name(),i_n);
    }

    @Override
    public ArrayField sub(double i_v) {
        return addScalarTransformOp("sub",i_v);
    }

    @Override
    public ArrayField mul(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarMultiplication().name(),i_v.getInput().scalar());
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
    public ArrayField rdiv(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarReverseDivision().name(),i_v.getInput().scalar());
        return addPairTransformOp("rdiv",i_v);
    }

    @Override
    public ArrayField div(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarDivision().name(),i_v.getInput().scalar());
        return addPairTransformOp(new DivOp().name(),i_v);
    }

    @Override
    public double getReal() {
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField[] args() {
        return new ArrayField[0];
    }

    @Override
    public ArrayField rsub(double v) {
        return addScalarTransformOp("rsub",v);
    }

    @Override
    public ArrayField rdiv(double v) {
        return addScalarTransformOp("rdiv",v);
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

    public ArrayField add(double v) {
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
        return addTransformOp(new LeakyReLU().name(),new Object[]{cutoff});
    }

    public ArrayField leakyReluDerivative() {
        return addTransformOp(new LeakyReLUDerivative().name());
    }

    public ArrayField leakyReluDerivative(double cutoff)  {
        return addTransformOp(new LeakyReLUDerivative().name(),new Object[]{cutoff});
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

    public ArrayField reshape(int[] shape) {
        return addTransformOp("reshape",new Object[]{shape});
    }

    public ArrayField transpose() {
        return addArrayOp(
                "transpose",
                null,
                ArrayUtil.reverseCopy(input.getShape()),
                null,
                OpState.OpType.SHAPE);
    }

    public ArrayField permute(int[] dimensions) {
        return addArrayOp(
                "permute",
                null,
                ArrayUtil.permute(input.getShape(),dimensions),
                null,
                OpState.OpType.SHAPE);

    }

    public ArrayField expandDims(int dim) {
        return addArrayOp(
                "expandDims",
                new int[]{dim},
                ArrayUtil.reverseCopy(input.getShape()),
                null,
                OpState.OpType.SHAPE);
    }

    public ArrayField sum(int[] dimensions) {
        return addArrayOp(
                new Sum().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    public ArrayField prod(int[] dimensions) {
        return addArrayOp(
                new Prod().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    public ArrayField mean(int[] dimensions) {
        return addArrayOp(
                new Mean().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }


    public ArrayField std(int[] dimensions,boolean biasCorrected) {
        return addArrayOp(
                new StandardDeviation().name()
                ,dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                new Object[]{biasCorrected},
                OpState.OpType.ACCUMULATION);
    }

    public ArrayField variance(int[] dimensions,boolean biasCorrected) {
        return addArrayOp(
                new Variance().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                new Object[]{biasCorrected},
                OpState.OpType.ACCUMULATION);
    }

    public ArrayField std(int[] dimensions) {
        return std(dimensions,false);
    }

    public ArrayField variance(int[] dimensions) {
        return variance(dimensions,false);
    }

    public ArrayField max(int[] dimensions) {
        return addArrayOp(
                new Max().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    public ArrayField min(int[] dimensions) {
        return addArrayOp(
                new Min().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    public ArrayField norm1(int[] dimensions) {
        return addArrayOp(
                new Norm1().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    public ArrayField norm2(int[] dimensions) {
        return addArrayOp(
                new Norm2().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    public ArrayField normmax(int[] dimensions) {
        return addArrayOp(
                new NormMax().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }


    public ArrayField valueArrayOf(int[] shape) {
        return addArrayOp(
                "full",
                null,
                shape,
                null,
                OpState.OpType.BROADCAST);
    }



    public ArrayField tile(int[] repeat) {
        return addArrayOp(
                "tile",
                null,
                null,
                new Object[]{repeat},
                OpState.OpType.BROADCAST);
    }



    public ArrayField repeat(int axis) {
        return addArrayOp("repeat",
                new int[]{axis},
                input.getShape(),
                null,
                OpState.OpType.BROADCAST);
    }

    public ArrayField broadcast(int[] shape) {
        return addArrayOp("broadcast",null,shape,null, OpState.OpType.BROADCAST);
    }


    public ArrayField eq(ArrayField i_y) {
        return addPairTransformOp(new EqualsWithEps().name(),i_y);
    }

    public ArrayField neq(ArrayField i_y) {
        return addPairTransformOp(new Not().name(),i_y);
    }
    public ArrayField or(ArrayField i_y) {
        return addPairTransformOp(new Or().name(),i_y);
    }

    public ArrayField rollAxis(int axis) {
        return addTransformOp("rollAxis",new Object[]{axis});
    }

    public ArrayField cosineSimilarity(ArrayField i_y, int...dimensions) {
        return addPairReduceOp(new CosineSimilarity().name(),i_y,dimensions,null);
    }

    public ArrayField euclideanDistance(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new EuclideanDistance().name(),i_y,dimensions,null);

    }

    public ArrayField manhattanDistance(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new ManhattanDistance().name(),i_y,dimensions,null);

    }

    public ArrayField lossBinaryXENT(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossBinaryXENT().name(),i_y,dimensions,null);
    }


    public ArrayField lossCosineSimilarity(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossCosineProximity().name(),i_y,dimensions,null);

    }

    public ArrayField lossHinge(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossHinge().name(),i_y,dimensions,null);

    }

    public ArrayField lossKLD(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossKLD().name(),i_y,dimensions,null);
    }


    public ArrayField lossL1(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossL1().name(),i_y,dimensions,null);
    }

    public ArrayField lossL2(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new CosineSimilarity().name(),i_y,dimensions,null);
    }

    public ArrayField lossMAE(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMAE().name(),i_y,dimensions,null);
    }

    public ArrayField lossMAPE(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMAPE().name(),i_y,dimensions,null);
    }

    public ArrayField lossMSE(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMSE().name(),i_y,dimensions,null);
    }

    public ArrayField lossMCXENT(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMCXENT().name(),i_y,dimensions,null);
    }

    public ArrayField lossMSLE(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMSLE().name(),i_y,dimensions,null);
    }

    public ArrayField lossNegativeLogLikelihood(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossNegativeLogLikelihood().name(),i_y,dimensions,null);
    }

    public ArrayField lossPoisson(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossPoisson().name(),i_y,dimensions,null);
    }

    public ArrayField lossSquaredHinge(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossSquaredHinge().name(),i_y,dimensions,null);
    }


    private ArrayField addTransformOp(String name) {
        return addTransformOp(name,null,null);
    }


    private ArrayField addScalarTransformOp(String name,Number scalarValue) {
        //result
        NDArrayVertex newVertex = new NDArrayVertex(this.ops.nextVertexId(),
                NDArrayInformation.builder()
                        .id(name + "(" + input.getId() + ")")
                        .shape(input.getShape()).build());

        //add the result vertex to the graph
        this.getOps().addVertex(newVertex);

        //map x -> z
        this.ops.addEdge(vertex.vertexID(),
                newVertex.vertexID(),
                OpState.builder()
                        .n(ArrayUtil.prod(input.getShape()))
                        .opName(name).extraArgs(new Object[]{scalarValue})
                        .scalarValue(scalarValue)
                        .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                        .opType(OpState.OpType.SCALAR_TRANSFORM).result(newVertex.getValue())
                        .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                        .build(),true);

        return new ArrayField(newVertex,ops);
    }

    private ArrayField addPairReduceOp(String name,ArrayField i_v,
                                       int[] dimensions,
                                       Object[] extraArgs) {
        return addPairReduceOp(name,i_v,dimensions,Shape.getReducedShape(input.getShape(),dimensions),extraArgs);
    }

    private ArrayField addPairReduceOp(String name,ArrayField i_v,
                                       int[] dimensions,
                                       int[] resultShape,Object[] extraArgs) {

        NDArrayInformation information =   NDArrayInformation.builder()
                .id(name + "("+ getVertex().getValue().getId() + "," + i_v.getVertex().getValue().getId() + ")")
                .shape(resultShape).build();

        NDArrayVertex newVertex = new NDArrayVertex(this.ops.nextVertexId(), information);

        //add the result vertex to the graph
        this.getOps().addVertex(newVertex);

        //map x -> z
        OpState xToz = OpState.builder()
                .n(ArrayUtil.prod(resultShape))
                .opName(name).extraArgs(extraArgs)
                .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(OpState.OpType.ACCUMULATION).build();
        xToz.setResult(information);
        this.ops.addEdge(vertex.vertexID(),
                newVertex.vertexID(),xToz,true);
        //map y -> z
        OpState yToZ = OpState.builder()
                .n(ArrayUtil.prod(resultShape))
                .opName(name).extraArgs(extraArgs)
                .id(i_v.getVertex().getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(i_v.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(OpState.OpType.ACCUMULATION).build();
        yToZ.setResult(information);
        this.ops.addEdge(i_v.getVertex().vertexID(),
                newVertex.vertexID(),yToZ,true);

        return new ArrayField(newVertex,ops);
    }



    private ArrayField addPairReduceOp(String name,ArrayField i_v,Object[] extraArgs) {
        //result
        NDArrayInformation resultInfo =  NDArrayInformation.builder()
                .id(name + "("+ getVertex().getValue().getId() + "," + i_v.getVertex().getValue().getId() + ")")
                .shape(input.getShape()).build();
        NDArrayVertex newVertex = new NDArrayVertex(this.ops.nextVertexId(), resultInfo);

        //add the result vertex to the graph
        this.getOps().addVertex(newVertex);

        //map x -> z
        OpState xToZ = OpState.builder()
                .n(ArrayUtil.prod(input.getShape()))
                .opName(name).extraArgs(extraArgs)
                .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(OpState.OpType.ACCUMULATION).build();
        xToZ.setResult(resultInfo);
        this.ops.addEdge(vertex.getIdx(),
                newVertex.vertexID(),xToZ,true);
        //map y -> z
        OpState yToZ = OpState.builder()
                .n(ArrayUtil.prod(input.getShape()))
                .opName(name).extraArgs(extraArgs)
                .id(i_v.getVertex().getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(i_v.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(OpState.OpType.ACCUMULATION).build();
        yToZ.setResult(resultInfo);
        this.ops.addEdge(i_v.getVertex().getIdx(),
                newVertex.vertexID(),yToZ,true);
        resultInfo.setOwner(yToZ);
        return new ArrayField(newVertex,ops);
    }


    private ArrayField addPairTransformOp(String name,ArrayField i_v,Object[] extraArgs) {
        //result
        NDArrayInformation resultInfo =  NDArrayInformation.builder()
                .id(name + "("+ getVertex().getValue().getId() + "," + i_v.getVertex().getValue().getId() + ")")
                .shape(input.getShape()).build();
        NDArrayVertex newVertex = new NDArrayVertex(this.ops.nextVertexId(), resultInfo);

        Preconditions.checkArgument(Arrays.equals(input.getShape(),i_v.getInput().getShape()),"X and y not equal shapes.");

        //add the result vertex to the graph
        this.getOps().addVertex(newVertex);

        //map x -> z
        OpState xToZ = OpState.builder()
                .n(ArrayUtil.prod(input.getShape()))
                .opName(name).extraArgs(extraArgs)
                .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(OpState.OpType.TRANSFORM).build();
        xToZ.setResult(resultInfo);
        this.ops.addEdge(vertex.getIdx(),
                newVertex.vertexID(),xToZ,true);
        //map y -> z
        OpState yToZ = OpState.builder()
                .n(ArrayUtil.prod(input.getShape()))
                .opName(name).extraArgs(extraArgs)
                .id(i_v.getVertex().getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(i_v.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(OpState.OpType.TRANSFORM).build();
        yToZ.setResult(resultInfo);
        this.ops.addEdge(i_v.getVertex().getIdx(),
                newVertex.vertexID(),yToZ,true);
        resultInfo.setOwner(yToZ);
        return new ArrayField(newVertex,ops);
    }

    private ArrayField addPairTransformOp(String name,ArrayField i_v) {
        return addPairTransformOp(name,i_v,null);
    }

    private ArrayField addTransformOp(String name,Object[] extraArgs) {
        return addTransformOp(name,null,extraArgs);
    }

    private ArrayField addTransformOp(String name,int[] axes,Object[] extraArgs) {
        return addArrayOp(name,
                axes,extraArgs,
                OpState.OpType.TRANSFORM);
    }




    private NDArrayVertex getVertex(String name,int[] shape) {
        //result
        NDArrayVertex newVertex = new NDArrayVertex(this.ops.nextVertexId() ,
                NDArrayInformation.builder()
                        .id(name + "(" + input.getId() + ")")
                        .shape(shape).build());
        return newVertex;

    }

    private ArrayField addArrayOp(String name,
                                  int[] axes,
                                  Object[] extraArgs,
                                  OpState.OpType opType) {
        return addArrayOp(name,
                axes,
                input.getShape(),
                extraArgs,
                opType);
    }

    private ArrayField addArrayOp(String name,
                                  int[] axes,
                                  int[] shape,
                                  Object[] extraArgs,
                                  OpState.OpType opType) {
        //result
        NDArrayVertex newVertex = getVertex(name,shape);
        //add the result vertex to the graph
        this.getOps().addVertex(newVertex);

        //map x -> z
        this.ops.addEdge(vertex.getIdx(),
                newVertex.vertexID(),OpState.builder()
                        .n(ArrayUtil.prod(input.getShape()))
                        .opName(name).extraArgs(extraArgs).axes(axes).result(newVertex.getValue())
                        .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                        .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                        .opType(opType).build(),true);

        return new ArrayField(newVertex,ops);
    }



    @Override
    public String toString() {
        return "ArrayField{" +
                "input=" + input +
                '}';
    }


    public ArrayField mmul(ArrayField value) {
        return addPairReduceOp("mmul",value,
                null,
                Shape.getMatrixMultiplyShape(getInput().getShape(),
                        value.getInput().getShape()),null);
    }

    public ArrayField tensorMmul(DifferentialFunction<ArrayField> y, int[][] dimensions) {
        return addPairReduceOp("tensorMmul",y.getValue(),
                null,
                ArrayUtil.getTensorMmulShape(getInput().getShape(),
                        y.getValue().getInput().getShape(),
                        dimensions),new Object[]{dimensions});

    }


}
