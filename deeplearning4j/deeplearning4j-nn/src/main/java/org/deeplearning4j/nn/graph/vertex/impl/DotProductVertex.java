package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.meta.Prediction;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce3.Dot;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

public class DotProductVertex extends BaseGraphVertex {

    private INDArray dimensions;

    public DotProductVertex(ComputationGraph graph, String name, int vertexIndex, INDArray dimensions) {
        this(graph, name, vertexIndex, null, null, dimensions);
    }

    public DotProductVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                             VertexIndices[] outputVertices, INDArray dimensions) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
        this.dimensions = dimensions;
    }

    public void setDimensions(INDArray dimensions) {
        this.dimensions = dimensions;
    }

    @Override
    public String toString() {

        StringBuilder sb = new StringBuilder();
        sb.append("DotProductVertex(id=").append(vertexIndex).append(",name=\"").append(vertexName).append("\",inputs=")
                .append(Arrays.toString(inputVertices)).append(",outputs=")
                .append(Arrays.toString(outputVertices)).append(")");
        return sb.toString();
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray a = inputs[0];
        INDArray b = inputs[1];
        Preconditions.checkState(a.rows() == b.columns() || a.columns() == b.rows());
        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)) {
            INDArray result = Nd4j.getExecutioner().exec(new Dot(a,b,
                            dimensions.toIntVector()));
            return result;
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {

        return new Pair<>(null, new INDArray[] {inputs[1], inputs[0]});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new RuntimeException("Vertex does not have gradients; gradients view array cannot be set here");
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        //No op
        if (maskArrays == null || maskArrays.length == 0) {
            return null;
        }

        return new Pair<>(maskArrays[0], currentMaskState);
    }
}
