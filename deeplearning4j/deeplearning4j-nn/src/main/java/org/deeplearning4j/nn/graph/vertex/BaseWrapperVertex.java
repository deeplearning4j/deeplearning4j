package org.deeplearning4j.nn.graph.vertex;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

/**
 * A base class for wrapper vertices: i.e., those vertices that have another vertex inside.
 * Use this as the basis of such wrapper vertices, which can selectively override only
 * the vertices that are required.
 *
 * @author Alex Black
 */
public abstract class BaseWrapperVertex implements GraphVertex {

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
        return underlying.getOutputVertices();
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
    public INDArray getGradientsViewArray() {
        return underlying.getGradientsViewArray();
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        underlying.setBackpropGradientsViewArray(backpropGradientsViewArray);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return underlying.feedForwardMaskArrays(maskArrays, currentMaskState, minibatchSize);
    }

    @Override
    public void setLayerAsFrozen() {
        underlying.setLayerAsFrozen();
    }

    @Override
    public void clearVertex() {
        underlying.clearVertex();
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropOnly) {
        return underlying.paramTable(backpropOnly);
    }

    @Override
    public TrainingConfig getConfig() {
        return underlying.getConfig();
    }

    @Override
    public INDArray params() {
        return underlying.params();
    }

    @Override
    public int numParams() {
        return underlying.numParams();
    }
}
