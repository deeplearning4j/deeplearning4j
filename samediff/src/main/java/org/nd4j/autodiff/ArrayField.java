package org.nd4j.autodiff;

import com.google.common.base.Preconditions;
import lombok.*;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.impl.binary.transform.gradient.GradientBackwardsMarker;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.scalar.*;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.*;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.UUID;

/**
 * Created by agibsonccc on 4/4/17.
 */
@AllArgsConstructor
@Getter
@Builder
@EqualsAndHashCode
public class ArrayField implements Field {
    @Getter
    @Setter
    private SameDiff ops;
    @Getter
    @Setter
    private NDArrayInformation input;
    @Getter
    @Setter
    private NDArrayVertex vertex;

    public ArrayField(NDArrayVertex ndArrayVertex,
                      SameDiff ops) {
        this.input = ndArrayVertex.getValue();
        this.vertex = ndArrayVertex;
        if(ops.getGraph().getVertex(vertex.getIdx()) == null)
            ops.getGraph().addVertex(ndArrayVertex);
        this.ops = ops;
    }

    public NDArrayInformation getInput() {
        return input;
    }

    public void setInput(NDArrayInformation input) {
        this.input = input;
    }

    @Override
    public ArrayField negate() {
        return addTransformOp(new Negative().name());
    }

    @Override
    public ArrayField add(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarAdd().name(),getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),false);

        return addPairTransformOp(new AddOp().name(),i_v);
    }



    @Override
    public ArrayField sub(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 || ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarSubtraction().name(),getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),false);
        return addPairTransformOp(new SubOp().name(),i_v);
    }

    @Override
    public ArrayField rsub(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarReverseSubtraction().name(),
                    getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),false);
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
    public ArrayField negatei() {
        return addTransformOp(new Negative().name(),new Object[]{true});
    }

    @Override
    public ArrayField addi(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 || ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarAdd().name(),getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),true);
        return addPairTransformOp(new AddOp().name(),i_v,new Object[]{true});
    }

    @Override
    public ArrayField addi(double i_v) {
        return addScalarTransformOp(new ScalarAdd().name(),input,getInput().getShape(),i_v,true);
    }

    @Override
    public ArrayField muli(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarMultiplication().name(),getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),true);
        return addPairTransformOp(new MulOp().name(),i_v,new Object[]{true});
    }

    @Override
    public ArrayField muli(double v) {
        return addScalarTransformOp(new ScalarMultiplication().name(),input,getInput().getShape(),v,true);
    }

    @Override
    public ArrayField powi(int i_n) {
        return null;
    }

    @Override
    public ArrayField mul(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarMultiplication().name(),getNonScalar(i_v),getNonScalarShape(i_v),getScalar(i_v),false);
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
    public ArrayField rsubi(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new RSubOp().name(),
                    getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),
                    true);
        return addPairTransformOp(new RSubOp().name(),i_v,new Object[]{true});
    }

    @Override
    public ArrayField rdivi(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new RDivOp().name(),
                    getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),
                    true);
        return addPairTransformOp(new RDivOp().name(),i_v,new Object[]{true});
    }

    @Override
    public ArrayField subi(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarSubtraction().name(),
                    getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),
                    true);
        return addPairTransformOp(new SubOp().name(),i_v,new Object[]{true});
    }

    @Override
    public ArrayField divi(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarDivision().name(),getNonScalar(i_v),
                    getNonScalarShape(i_v),getScalar(i_v),true);
        return addPairTransformOp(new DivOp().name(),i_v,new Object[]{true});
    }

    @Override
    public ArrayField inversei() {
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField subi(double i_v) {
        return addScalarTransformOp(new ScalarSubtraction().name(),input,getInput().getShape(),i_v,true);
    }

    @Override
    public ArrayField rsubi(double v) {
        return addScalarTransformOp(new ScalarReverseSubtraction().name(),input,getInput().getShape(),v,true);
    }

    @Override
    public ArrayField rdivi(double v) {
        return addScalarTransformOp(new ScalarReverseDivision().name(),input,getInput().getShape(),v,true);
    }

    @Override
    public ArrayField divi(double v) {
        return addScalarTransformOp(new ScalarDivision().name(),input,getInput().getShape(),v,true);
    }

    @Override
    public ArrayField rdiv(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 ||
                ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarReverseDivision().name(),getNonScalar(i_v),getNonScalarShape(i_v),getScalar(i_v),false);
        return addPairTransformOp("rdiv",i_v);
    }

    @Override
    public ArrayField div(ArrayField i_v) {
        if(ArrayUtil.prod(i_v.getInput().getShape()) == 1 || ArrayUtil.prod(getInput().getShape()) == 1)
            return addScalarTransformOp(new ScalarDivision().name(),getNonScalar(i_v),getNonScalarShape(i_v),getScalar(i_v),false);
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

    @Override
    public ArrayField pow(ArrayField a) {
        return addPairTransformOp(new Pow().name(),a);
    }

    @Override
    public ArrayField floor() {
        return addTransformOp(new Floor().name());
    }

    @Override
    public ArrayField ceil() {
        return addTransformOp(new Ceil().name());
    }

    @Override
    public ArrayField round() {
        return addTransformOp(new Round().name());
    }

    @Override
    public ArrayField abs() {
        return addTransformOp(new Abs().name());
    }


    @Override
    public ArrayField sqrt() {
        return addTransformOp(new Sqrt().name());
    }
    // Operators for double
    @Override
    public ArrayField add(double v) {
        return addScalarTransformOp(new ScalarAdd().name(),v);
    }

    @Override
    public ArrayField minus(double v) {
        return addScalarTransformOp(new ScalarSubtraction().name(),v);
    }

    @Override
    public ArrayField prod(double v) {
        return addScalarTransformOp(new ScalarMultiplication().name(),v);
    }

    @Override
    public ArrayField div(double v) {
        return addScalarTransformOp(new ScalarDivision().name(),v);
    }

    @Override
    public ArrayField pow(double v) {
        return addScalarTransformOp(new Pow().name(),v);
    }

    @Override
    public ArrayField cos() {
        return addTransformOp(new Cos().name());
    }

    @Override
    public ArrayField acos() {
        return addTransformOp(new ACos().name());
    }

    @Override
    public ArrayField cosh() {
        return addTransformOp(new Cosh().name());
    }

    @Override
    public ArrayField acosh() {
        //  return new ArrayField(OpState.fromOp(new INDArray(Math.log(Math.sqrt(Math.pow(x, 2) - 1) + x)),ops);
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField sin() {
        return addTransformOp(new Sin().name());
    }

    @Override
    public ArrayField asin() {
        return addTransformOp(new ASin().name());
    }

    @Override
    public ArrayField sinh() {
        return addTransformOp(new Sinh().name());
    }

    @Override
    public ArrayField asinh() {
        return addTransformOp(new ASinh().name());


    }

    @Override
    public ArrayField tan() {
        return addTransformOp(new Tan().name());
    }

    @Override
    public ArrayField atan() {
        return addTransformOp(new ATan().name());
    }

    @Override
    public ArrayField tanh() {
        return addTransformOp(new Tanh().name());
    }

    @Override
    public ArrayField tanhDerivative(ArrayField wrt) {
        return addTransformOp(new TanhDerivative().name());
    }


    @Override
    public ArrayField atanh() {
        return addTransformOp(new ATanh().name());
    }

    @Override
    public ArrayField exp() {
        return addTransformOp(new Exp().name());
    }

    @Override
    public ArrayField log() {
        return addTransformOp(new Log().name());
    }

    @Override
    public ArrayField log10() {
        //return new ArrayField(OpState.fromOp(new INDArray(Math.log10(x)),ops);
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField sgn() {
        return addTransformOp(new Sign().name());
    }

    @Override
    public ArrayField pwr(ArrayField y) {
        //return new ArrayField(OpState.fromOp(new INDArray(Math.pow(Math.abs(x)), y.doubleValue())),ops);
        throw new UnsupportedOperationException();
    }

    @Override
    public ArrayField pwrs(ArrayField y) {
        // return new ArrayField(OpState.fromOp(new INDArray(Math.pow(Math.abs(x)), y.doubleValue()) * Math.signum(x)),ops);
        throw new UnsupportedOperationException();

    }

    @Override
    public ArrayField square() {
        return mul(this);
    }

    @Override
    public ArrayField relu() {
        return addTransformOp(new RectifedLinear().name());
    }

    @Override
    public ArrayField hardTanh() {
        return addTransformOp(new HardTanh().name());
    }

    @Override
    public ArrayField hardTanhDerivative(ArrayField wrt) {
        return addTransformOp(new HardTanhDerivative().name());
    }

    @Override
    public ArrayField leakyRelu() {
        return addTransformOp(new LeakyReLU().name());
    }

    @Override
    public ArrayField elu() {
        return addTransformOp(new ELU().name());
    }

    @Override
    public ArrayField eluDerivative(ArrayField wrt) {
        return addTransformOp(new ELUDerivative().name());
    }


    @Override
    public ArrayField leakyRelu(double cutoff)  {
        return addTransformOp(new LeakyReLU().name(),new Object[]{cutoff});
    }

    @Override
    public ArrayField leakyReluDerivative() {
        return addTransformOp(new LeakyReLUDerivative().name());
    }

    @Override
    public ArrayField leakyReluDerivative(ArrayField wrt, double cutoff)  {
        return addTransformOp(new LeakyReLUDerivative().name(),new Object[]{cutoff});
    }

    @Override
    public ArrayField selu() {
        return addTransformOp(new SELU().name());
    }


    @Override
    public ArrayField seluDerivative(ArrayField wrt) {
        return addTransformOp(new SELUDerivative().name());
    }

    @Override
    public ArrayField max(double v) {
        return addScalarTransformOp(new ScalarMax().name(),v);
    }

    @Override
    public ArrayField min(double v) {
        return addScalarTransformOp(new ScalarMin().name(),v);
    }

    @Override
    public ArrayField fmod(double v) {
        return addScalarTransformOp(new ScalarFMod().name(),v);
    }

    @Override
    public ArrayField set(double v) {
        return addScalarTransformOp(new ScalarSet().name(),v);
    }


    @Override
    public ArrayField sigmoid() {
        return addTransformOp(new Sigmoid().name());
    }

    @Override
    public ArrayField sigmoidDerivative(ArrayField wrt) {
        return addTransformOp(new  org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative().name());
    }

    @Override
    public ArrayField step() {
        return addTransformOp(new Step().name());
    }

    @Override
    public ArrayField softsign() {
        return addTransformOp(new SoftSign().name());
    }

    @Override
    public ArrayField softsignDerivative(ArrayField wrt) {
        return addTransformOp(new SoftSignDerivative().name());
    }

    @Override
    public ArrayField softmax() {
        return addTransformOp(new SoftMax().name());
    }

    @Override
    public ArrayField logSoftmax() {
        return addTransformOp(new LogSoftMax().name());

    }

    @Override
    public ArrayField softmaxDerivative(ArrayField wrt) {
        return addGradientOp(new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftMaxDerivative().name(),wrt,null);
    }

    @Override
    public ArrayField softplus() {
        return addTransformOp(new SoftPlus().name());
    }

    @Override
    public ArrayField reshape(int[] shape) {
        return addTransformOp("reshape",new Object[]{shape});
    }

    @Override
    public ArrayField transpose() {
        return addArrayOp(
                "transpose",
                null,
                ArrayUtil.reverseCopy(input.getShape()),
                null,
                OpState.OpType.SHAPE);
    }

    @Override
    public ArrayField permute(int[] dimensions) {
        return addArrayOp(
                "permute",
                null,
                ArrayUtil.permute(input.getShape(),dimensions),
                null,
                OpState.OpType.SHAPE);

    }

    @Override
    public ArrayField expandDims(int dim) {
        return addArrayOp(
                "expandDims",
                new int[]{dim},
                ArrayUtil.reverseCopy(input.getShape()),
                null,
                OpState.OpType.SHAPE);
    }

    @Override
    public ArrayField sum(int[] dimensions) {
        return addArrayOp(
                new Sum().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    @Override
    public ArrayField prod(int[] dimensions) {
        return addArrayOp(
                new Prod().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    @Override
    public ArrayField mean(int[] dimensions) {
        return addArrayOp(
                new Mean().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }


    @Override
    public ArrayField std(int[] dimensions,boolean biasCorrected) {
        return addArrayOp(
                new StandardDeviation().name()
                ,dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                new Object[]{biasCorrected},
                OpState.OpType.ACCUMULATION);
    }

    @Override
    public ArrayField variance(int[] dimensions,boolean biasCorrected) {
        return addArrayOp(
                new Variance().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                new Object[]{biasCorrected},
                OpState.OpType.ACCUMULATION);
    }

    @Override
    public ArrayField std(int[] dimensions) {
        return std(dimensions,false);
    }

    @Override
    public ArrayField variance(int[] dimensions) {
        return variance(dimensions,false);
    }

    @Override
    public ArrayField max(int[] dimensions) {
        return addArrayOp(
                new Max().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    @Override
    public ArrayField min(int[] dimensions) {
        return addArrayOp(
                new Min().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    @Override
    public ArrayField norm1(int[] dimensions) {
        return addArrayOp(
                new Norm1().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    @Override
    public ArrayField norm2(int[] dimensions) {
        return addArrayOp(
                new Norm2().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }

    @Override
    public ArrayField normmax(int[] dimensions) {
        return addArrayOp(
                new NormMax().name(),
                dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                null,
                OpState.OpType.ACCUMULATION);
    }


    @Override
    public ArrayField valueArrayOf(int[] shape) {
         Preconditions.checkArgument(shape != null,"Passed in shape must not be null.");
        return addArrayOp(
                "full",
                null,
                shape,
                null,
                OpState.OpType.BROADCAST);
    }



    @Override
    public ArrayField tile(int[] repeat) {
        return addArrayOp(
                "tile",
                null,
                null,
                new Object[]{repeat},
                OpState.OpType.BROADCAST);
    }


    @Override
    public ArrayField repeat(int axis) {
        return addArrayOp("repeat",
                new int[]{axis},
                input.getShape(),
                null,
                OpState.OpType.BROADCAST);
    }


    @Override
    public ArrayField set(ArrayField value1) {
        return addPairTransformOp("set",value1);
    }

    @Override
    public ArrayField broadcast(int[] shape) {
        return addArrayOp("broadcast",null,shape,null, OpState.OpType.SHAPE);
    }


    @Override
    public ArrayField eq(ArrayField i_y) {
        return addPairTransformOp(new EqualsWithEps().name(),i_y);
    }

    @Override
    public ArrayField neq(ArrayField i_y) {
        return addPairTransformOp(new Not().name(),i_y);
    }

    @Override
    public ArrayField or(ArrayField i_y) {
        return addPairTransformOp(new Or().name(),i_y);
    }

    @Override
    public ArrayField rollAxis(int axis) {
        return addTransformOp("rollAxis",new Object[]{axis});
    }

    @Override
    public ArrayField cosineSimilarity(ArrayField i_y, int...dimensions) {
        return addPairReduceOp(new CosineSimilarity().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField euclideanDistance(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new EuclideanDistance().name(),i_y,dimensions,null);

    }

    @Override
    public ArrayField manhattanDistance(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new ManhattanDistance().name(),i_y,dimensions,null);

    }

    @Override
    public ArrayField lossBinaryXENT(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossBinaryXENT().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField lossCosineSimilarity(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossCosineProximity().name(),i_y,dimensions,null);

    }

    @Override
    public ArrayField lossHinge(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossHinge().name(),i_y,dimensions,null);

    }

    @Override
    public ArrayField lossKLD(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossKLD().name(),i_y,dimensions,null);
    }


    @Override
    public ArrayField lossL1(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossL1().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField lossL2(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new CosineSimilarity().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField lossMAE(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMAE().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField lossMAPE(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMAPE().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField lossMSE(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMSE().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField lossMCXENT(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMCXENT().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField lossMSLE(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossMSLE().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField lossNegativeLogLikelihood(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossNegativeLogLikelihood().name(),i_y,dimensions,null);
    }


    @Override
    public ArrayField lossPoisson(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossPoisson().name(),i_y,dimensions,null);
    }


    @Override
    public ArrayField lossSquaredHinge(ArrayField i_y,int...dimensions) {
        return addPairReduceOp(new LossSquaredHinge().name(),i_y,dimensions,null);
    }

    @Override
    public ArrayField arg() {
        throw new UnsupportedOperationException();
    }



    private ArrayField addTransformOp(String name) {
        return addTransformOp(name,null,null);
    }


    private ArrayField addFirstScalarTransformOp(String name,
                                                 ArrayField i_v,
                                                 Object[] extraArgs) {
        Preconditions.checkState(this.ops == i_v.ops, "If adding a field. Must be apart of the same graph.");

        NDArrayInformation ndArrayInformation =  NDArrayInformation.builder()
                .id(name + "(" + input.getId() + ")")
                .scalarValue(this.getInput().getScalarValue())
                .arrId(UUID.randomUUID().toString())
                .shape(getInput().getShape()).build();
        //result
        NDArrayVertex newVertex = new NDArrayVertex(
                this.ops,
                this.ops.getGraph().nextVertexId(),
                this.vertex.depth() + 1,
                ndArrayInformation);

        //add the result vertex to the graph
        this.ops.getGraph().addVertex(newVertex);

        OpState owner =    OpState.builder()
                .n(ArrayUtil.prod(getInput().getShape()))
                .opName(name).extraArgs(extraArgs)
                .scalarValue(ArrayUtil.prod(getInput().getShape()) == 1 ? getInput().scalar() : null)
                .arrayField(this)
                .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .opType(OpState.OpType.SCALAR_TRANSFORM).result(newVertex.getValue())
                .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                .build();

        //map x -> z
        this.ops.getGraph().addEdge(
                new int[]{vertex.vertexID()},
                new int[]{newVertex.vertexID()},owner
                ,true);

        ndArrayInformation.setOwner(owner);
        if(owner.isInPlace()) {
            ndArrayInformation.setArrId(input.getArrId());
        }
        return new ArrayField(newVertex,ops);
    }

    private ArrayField addScalarTransformOp(String name,
                                            NDArrayInformation input,
                                            int[] nonScalarShape,
                                            Number scalarValue,
                                            boolean inPlace) {
        //for the purpose of the graph, we only need the scalar
        //value, therefore the input should be the
        //non scalar


        NDArrayInformation result =  NDArrayInformation.builder()
                .scalarValue(scalarValue)
                .id(name + "(" + input.getId() + ")")
                .arrId(UUID.randomUUID().toString())
                .shape(nonScalarShape).build();
        //result
        NDArrayVertex newVertex = new NDArrayVertex(
                this.ops,
                this.ops.getGraph().nextVertexId(),
                this.vertex.depth() + 1,
                result);

        //add the result vertex to the graph
        this.ops.getGraph().addVertex(newVertex);

        OpState owner =  OpState.builder()
                .n(ArrayUtil.prod(input.getShape()))
                .opName(name).extraArgs(new Object[]{scalarValue,inPlace})
                .scalarValue(scalarValue).arrayField(this)
                .id(vertex.getValue().getId() + "-> " +
                        name + " " + newVertex.getValue().getId())
                .opType(OpState.OpType.SCALAR_TRANSFORM)
                .result(newVertex.getValue())
                .vertexIds(new String[]{String.valueOf(vertex.vertexID()),
                        String.valueOf(newVertex.vertexID())})
                .build();
        //map x -> z
        this.ops.getGraph().addEdge(
                new int[]{vertex.vertexID()},
                new int[]{newVertex.vertexID()}, owner,true);
        result.setOwner(owner);
        if(owner.isInPlace()) {
            result.setArrId(input.getArrId());
        }
        return new ArrayField(newVertex,ops);
    }

    private ArrayField addScalarTransformOp(String name,Number scalarValue) {
        return addScalarTransformOp(name,input,input.getShape(),scalarValue,false);
    }

    private ArrayField addPairReduceOp(String name,
                                       ArrayField i_v,
                                       int[] dimensions,
                                       Object[] extraArgs) {
        return addPairReduceOp(name,
                i_v,dimensions,
                Shape.getReducedShape(input.getShape(),dimensions),
                extraArgs);
    }

    private ArrayField addPairReduceOp(String name,
                                       ArrayField i_v,
                                       int[] dimensions,
                                       int[] resultShape,
                                       Object[] extraArgs) {
        Preconditions.checkState(this.ops == i_v.ops, "If adding a field. Must be apart of the same graph.");

        NDArrayInformation information =   NDArrayInformation.builder()
                .id(name + "("+ getVertex().getValue().getId() + "," + i_v.getVertex().getValue().getId() + ")")
                .arrId(UUID.randomUUID().toString())
                .shape(resultShape).build();

        NDArrayVertex newVertex = new NDArrayVertex(
                this.ops,
                this.ops.getGraph().nextVertexId(),
                this.vertex.depth() + 1,
                information);

        //add the result vertex to the graph
        this.ops.getGraph().addVertex(newVertex);

        //map x -> z
        OpState xToz = OpState.builder()
                .n(ArrayUtil.prod(resultShape)).axes(dimensions)
                .opName(name).extraArgs(extraArgs).arrayField( this)

                .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(OpState.OpType.ACCUMULATION).build();
        xToz.setResult(information);
        this.ops.getGraph().addEdge(
                new int[] {vertex.vertexID()},
                new int[]{newVertex.vertexID()},xToz,true);
        //map y -> z
        OpState yToZ = OpState.builder()
                .n(ArrayUtil.prod(resultShape))
                .opName(name).extraArgs(extraArgs).arrayField( this)
                .id(i_v.getVertex().getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(i_v.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(OpState.OpType.ACCUMULATION).build();
        yToZ.setResult(information);

        if(xToz.isInPlace()) {
            information.setArrId(input.getArrId());
        }

        this.ops.getGraph().addEdge(
                new int[]{i_v.getVertex().vertexID()},
                new int[]{newVertex.vertexID()},yToZ,true);

        return new ArrayField(newVertex,ops);
    }
    /**
     *
     * @param name
     * @param i_v
     * @param extraArgs
     * @return
     */

    private ArrayField addPairTransformOp(String name, ArrayField i_v, OpState.OpType opType,Object[] extraArgs) {
        if(ArrayUtil.prod(getInput().getShape()) == 1 || ArrayUtil.prod(i_v.getInput().getShape()) == 1) {
            return addFirstScalarTransformOp(name + "_scalar",
                    i_v,extraArgs);
        }

        Preconditions.checkState(this.ops == i_v.ops, "If adding a field. Must be apart of the same graph.");
        //result
        NDArrayInformation resultInfo =  NDArrayInformation.builder().arrId(UUID.randomUUID().toString())
                .id(name + "("+ getVertex().getValue().getId() + "," + i_v.getVertex().getValue().getId() + ")")
                .shape(input.getShape()).build();
        NDArrayVertex newVertex = new NDArrayVertex(
                this.ops,
                this.ops.getGraph().nextVertexId(),
                this.vertex.depth() + 1,
                resultInfo);

        Preconditions.checkArgument(Arrays.equals(input.getShape(),i_v.getInput().getShape()),"X and y not equal shapes.");

        //add the result vertex to the graph
        this.ops.getGraph().addVertex(newVertex);

        //map x -> z
        OpState xToZ = OpState.builder()
                .n(ArrayUtil.prod(input.getShape()))
                .opName(name).extraArgs(extraArgs)
                .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(opType).build();
        xToZ.setResult(resultInfo);
        if(!ops.graph().isFrozen() && vertex.vertexID() == newVertex.vertexID())
            throw new IllegalStateException("Attempted to add edge with vertex id of " + newVertex.vertexID() +
                    " when next vertex id was " + this.ops.getGraph().getNextVertexId() + " . This usually means that the vertex id generation was behind the nodes being added.");

        this.ops.getGraph().addEdge(
                new int[]{vertex.vertexID()},
                new int[]{newVertex.vertexID()},xToZ,true);
        //map y -> z
        OpState yToZ = OpState.builder()
                .n(ArrayUtil.prod(input.getShape()))
                .opName(name).extraArgs(extraArgs)
                .id(i_v.getVertex().getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(i_v.getVertex().vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(opType).build();
        yToZ.setResult(resultInfo);
        if(!ops.graph().isFrozen() && i_v.getVertex().vertexID() == newVertex.vertexID())
            throw new IllegalStateException("Attempted to add edge with vertex id of " + newVertex.vertexID() +
                    " when next vertex id was " + this.ops.getGraph().getNextVertexId() + " . This usually means that the vertex id generation was behind the nodes being added.");
        this.ops.getGraph().addEdge(
                new int[]{i_v.getVertex().vertexID()},
                new int[]{newVertex.vertexID()},yToZ,true);
        resultInfo.setOwner(yToZ);

        if(xToZ.isInPlace()) {
            resultInfo.setArrId(input.getArrId());
        }

        return new ArrayField(newVertex,ops);
    }


    /**
     *
     * @param name
     * @param wrt
     * @param extraArgs
     * @return
     */

    private ArrayField addGradientOp(String name,ArrayField wrt,Object[] extraArgs) {
        return addPairTransformOp(name,wrt, OpState.OpType.GRADIENT,extraArgs);
    }
    /**
     *
     * @param name
     * @param i_v
     * @param extraArgs
     * @return
     */

    private ArrayField addPairTransformOp(String name,ArrayField i_v,Object[] extraArgs) {
      return addPairTransformOp(name,i_v, OpState.OpType.TRANSFORM,extraArgs);
    }

    private ArrayField  addPairTransformOp(String name,ArrayField i_v) {
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
        NDArrayVertex newVertex = new NDArrayVertex(this.ops,
                this.ops.getGraph().nextVertexId() ,
                this.vertex.depth() + 1,
                NDArrayInformation.builder().arrId(UUID.randomUUID().toString())
                        .scalarValue(getInput().getScalarValue())
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
        this.ops.getGraph().addVertex(newVertex);

        OpState opState = OpState.builder()
                .n(ArrayUtil.prod(input.getShape()))
                .opName(name).extraArgs(extraArgs).axes(axes)
                .result(newVertex.getValue())
                .id(vertex.getValue().getId() + "-> " + name + " " + newVertex.getValue().getId())
                .vertexIds(new String[]{String.valueOf(vertex.vertexID()),String.valueOf(newVertex.vertexID())})
                .opType(opType).build();

        if(opState.isInPlace()) {
            newVertex.getValue().setArrId(input.getArrId());
        }
        //map x -> z
        this.ops.getGraph().addEdge(
                new int[]{vertex.vertexID()},
                new int[]{newVertex.vertexID()},opState,true);

        return new ArrayField(newVertex,ops);
    }


    private double getScalar(ArrayField other) {
        if(ArrayUtil.prod(getInput().getShape()) == 1) {
            if(this.getInput().getScalarValue() != null)
                return this.getInput().getScalarValue().doubleValue();
            else if(this.getInput().getOwner() != null && this.getInput().getOwner().getScalarValue() != null)
                return this.getInput().getOwner().getScalarValue().doubleValue();

            else if(ops.getVertexToArray().get(input.getArrId()) != null) {
                return ops.getVertexToArray().get(input.getArrId()).getDouble(0);
            }
        }
        else if(ArrayUtil.prod(other.getInput().getShape()) == 1) {
            if(other.getInput().getScalarValue() != null)
                return other.getInput().getScalarValue().doubleValue();
            else if(other.getInput().getScalarValue() != null)
                return other.getInput().getScalarValue().doubleValue();

            else if(ops.getVertexToArray().get(other.getInput().getArrId())
                    != null) {
                return ops.getVertexToArray().get(other.getInput().getArrId())
                        .getDouble(0);
            }
        }

        return Double.MIN_VALUE;
    }

    private NDArrayInformation getNonScalar(ArrayField other) {
        if(ArrayUtil.prod(getInput().getShape()) != 1 &&
                ArrayUtil.prod(other.getInput().getShape()) == 1)
            return this.getInput();
        else if(ArrayUtil.prod(other.getInput().getShape()) != 1 &&
                ArrayUtil.prod(getInput().getShape()) == 1)
            return other.getInput();
            //both scalar
        else {
            return other.getInput();
        }

    }

    private int[] getNonScalarShape(ArrayField other) {
        if(ArrayUtil.prod(getInput().getShape()) != 1 && ArrayUtil.prod(
                other.getInput().getShape()) == 1)
            return this.getInput().getShape();
        else if(ArrayUtil.prod(other.getInput().getShape()) != 1 &&
                ArrayUtil.prod(getInput().getShape()) == 1)
            return other.getInput().getShape();
        else
            return new int[] {1,1};


    }

    @Override
    public String toString() {
        return "ArrayField{" +
                "input=" + input +
                '}';
    }

    /**
     * Matrix multiply with a
     * transpose specifier.
     * @param value
     * @param mMulTranspose
     * @return
     */
    public ArrayField mmul(ArrayField value,MMulTranspose mMulTranspose) {
        int[] inputShape = mMulTranspose.isTransposeA() ? ArrayUtil.reverseCopy(getInput().getShape()) : getInput().getShape();
        int[] otherShape =  mMulTranspose.isTransposeB() ? ArrayUtil.reverseCopy(value.getInput().getShape()) : value.getInput().getShape();
        return addPairReduceOp("mmul",value,
                null,
                Shape.getMatrixMultiplyShape(inputShape,
                        otherShape),new Object[]{mMulTranspose});
    }


    /**
     * Normal matrix multiply
     * @param value
     * @return
     */
    public ArrayField mmul(ArrayField value) {
        return addPairReduceOp("mmul",
                value,
                null,
                Shape.getMatrixMultiplyShape(getInput().getShape(),
                        value.getInput().getShape()),new Object[]{MMulTranspose.allFalse()});
    }

    /**
     * Transpsoe matrix multiply
     * @param y
     * @param dimensions
     * @return
     */
    public ArrayField tensorMmul(DifferentialFunction y,
                                 int[][] dimensions) {
        return addPairReduceOp("tensorMmul",y.getValue(true),
                null,
                ArrayUtil.getTensorMmulShape(getInput().getShape(),
                        y.getValue(true).getInput().getShape(),
                        dimensions),new Object[]{dimensions,MMulTranspose.allFalse()});

    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ArrayField that = (ArrayField) o;

        if (input != null ? !input.equals(that.input) : that.input != null) return false;
        return vertex != null ? vertex.equals(that.vertex) : that.vertex == null;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (input != null ? input.hashCode() : 0);
        result = 31 * result + (vertex != null ? vertex.hashCode() : 0);
        return result;
    }

    public ArrayField gradientBackwardsMarker(ArrayField value, ArrayField value1) {
        return addGradientOp(new GradientBackwardsMarker().functionName(),value,new Object[]{value1});
    }
}
