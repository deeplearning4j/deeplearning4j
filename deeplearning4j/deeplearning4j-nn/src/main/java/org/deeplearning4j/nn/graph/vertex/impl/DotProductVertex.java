package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce3.Dot;
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

public class DotProductVertex extends BaseGraphVertex {

    public DotProductVertex(ComputationGraph graph, String name, int vertexIndex) {
        this(graph, name, vertexIndex, null, null);
    }

    public DotProductVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                             VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
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
        try(MemoryWorkspace ws = workspaceMgr.notifyScopeBorrowed(ArrayType.ACTIVATIONS)) {
            INDArray result = Nd4j.getExecutioner().exec(new Dot(a,b));
            return result;
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        INDArray a = inputs[0];
        INDArray b = inputs[1];
        return null;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {

    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return null;
    }
}
