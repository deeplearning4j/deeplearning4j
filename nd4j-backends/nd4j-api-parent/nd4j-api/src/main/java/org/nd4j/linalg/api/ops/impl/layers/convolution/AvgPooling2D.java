package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Pooling2D operation
 */
@Slf4j
@Getter
public class AvgPooling2D extends DynamicCustomOp {

    protected Pooling2DConfig config;

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }

    public AvgPooling2D() {}

    @Builder(builderMethodName = "builder")
    @SuppressWarnings("Used in lombok")
    public AvgPooling2D(SameDiff sameDiff, DifferentialFunction[] inputs, INDArray[] arrayInputs, INDArray[] arrayOutputs, Pooling2DConfig config) {
        super(null,sameDiff, inputs, false);
        if(arrayInputs != null) {
            getInputArguments().addAll(Arrays.asList(arrayInputs));
        }

        if(arrayOutputs != null) {
            getOutputArguments().addAll(Arrays.asList(arrayOutputs));
        }

        this.config = config;


        addArgs();
    }


    private void addArgs() {
        getIArguments().add(config.getKh());
        getIArguments().add(config.getKw());
        getIArguments().add(config.getSy());
        getIArguments().add(config.getSx());
        getIArguments().add(config.getPh());
        getIArguments().add(config.getPw());
        getIArguments().add(config.getDh());
        getIArguments().add(config.getDw());
        getIArguments().add(fromBoolean(config.isSameMode()));
        getIArguments().add((int) config.getExtra());

    }

    @Override
    public String opName() {
        return getPoolingPrefix() + "pool2d";
    }


    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        List<DifferentialFunction> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Pooling2DDerivative pooling2DDerivative = Pooling2DDerivative.derivativeBuilder()
                .inputs(inputs.toArray(new DifferentialFunction[inputs.size()]))
                .sameDiff(sameDiff)
                .config(config)
                .build();
        ret.addAll(Arrays.asList(pooling2DDerivative.getOutputFunctions()));
        return ret;
    }


    @Override
    public TOp asIntermediateRepresentation(NodeDef tfNode, TGraph graph) {
        val tNode = buildBasicNode(tfNode, graph);

        val aStrides = tfNode.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aKernels = tfNode.getAttrOrThrow("ksize");
        val tfKernels = aKernels.getList().getIList();

        val kY = tfKernels.get(1);
        val kX = tfKernels.get(2);

        val aPadding = tfNode.getAttrOrThrow("padding");

        val paddingMode = aPadding.getS().toStringUtf8().replaceAll("\"","");

        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

        if (!isSameMode)
            log.debug("Mode: {}", paddingMode);

        log.debug("Pooling: k: [{},{}]; s: [{}, {}], padding: {}", kY, kX, sY, sX, aPadding);

        tNode.getOpState().setExtraBits(new int[] {kY.intValue(), kX.intValue(), sY.intValue(), sX.intValue(), 0, 0, 1, 1, isSameMode ? 1 : 0 });

        return tNode;
    }


    @Override
    public String onnxName() {
        return "AveragePool";
    }

    @Override
    public String tensorflowName() {
        return "avg_pool";
    }


    public String getPoolingPrefix() {
        return "avg";
    }

}
