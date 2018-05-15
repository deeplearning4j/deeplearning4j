package org.deeplearning4j.nn.graph.vertex;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

public class BaseWrapperVertex implements GraphVertex {

    protected GraphVertex underlying;

    protected BaseWrapperVertex(GraphVertex underlying){
        this.underlying = underlying;
    }

    @Override
    public String getVertexName() {
        return underlying.getVertexName();
    }

    @Override
    public int getVertexIndex() {
        return underlying.getVertexIndex();
    }

    @Override
    public int getNumInputArrays() {
        return underlying.getNumInputArrays();
    }

    @Override
    public int getNumOutputConnections() {
        return underlying.getNumOutputConnections();
    }

    @Override
    public VertexIndices[] getInputVertices() {
        return underlying.getInputVertices();
    }

    @Override
    public void setInputVertices(VertexIndices[] inputVertices) {
        underlying.setInputVertices(inputVertices);
    }

    @Override
    public VertexIndices[] getOutputVertices() {
        underlying.getOutputVertices();
    }

    @Override
    public void setOutputVertices(VertexIndices[] outputVertices) {
        underlying.setOutputVertices(outputVertices);
    }

    @Override
    public boolean hasLayer() {
        return underlying.hasLayer();
    }

    @Override
    public boolean isInputVertex() {
        return underlying.isInputVertex();
    }

    @Override
    public boolean isOutputVertex() {
        return underlying.isOutputVertex();
    }

    @Override
    public void setOutputVertex(boolean outputVertex) {
        underlying.setOutputVertex(outputVertex);
    }

    @Override
    public Layer getLayer() {
        return underlying.getLayer();
    }

    @Override
    public void setInput(int inputNumber, INDArray input, LayerWorkspaceMgr workspaceMgr) {
        underlying.setInput(inputNumber, input, workspaceMgr);
    }

    @Override
    public void setEpsilon(INDArray epsilon) {
        underlying.setEpsilon(epsilon);
    }

    @Override
    public void clear() {
        underlying.clear();
    }

    @Override
    public boolean canDoForward() {
        return underlying.canDoForward();
    }

    @Override
    public boolean canDoBackward() {
        return underlying.canDoBackward();
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return underlying.doForward(training, workspaceMgr);
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        return underlying.doBackward(tbptt, workspaceMgr);
    }

    @Override
    public INDArray[] getInputs() {
        return underlying.getInputs();
    }

    @Override
    public INDArray getEpsilon() {
        return underlying.getEpsilon();
    }

    @Override
    public void setInputs(INDArray... inputs) {
        underlying.setInputs(inputs);
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        underlying.setBackpropGradientsViewArray(backpropGradientsViewArray);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        underlying.feedForwardMaskArrays(maskArrays, currentMaskState, minibatchSize);
    }

    @Override
    public void setLayerAsFrozen() {
        underlying.setLayerAsFrozen();
    }

    @Override
    public void clearVertex() {
        underlying.clearVertex();
    }
}
