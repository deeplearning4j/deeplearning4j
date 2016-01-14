package org.deeplearning4j.nn.graph;

import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**ComputationGraph network (neural network with arbitrary connection structure)
 */
public class ComputationGraph implements Serializable, Model {

    protected ComputationGraphConfiguration configuration;
    protected transient Solver solver;	//Used to call optimizers during backprop

    protected GraphVertex[] vertices;
    protected int[] topologicalOrder;

    private int layerCount;
    private int numInputArrays;
    private int numOutputArrays;

    public ComputationGraph(ComputationGraphConfiguration configuration){
        this.configuration = configuration;
    }

    /** The number of inputs to this network */
    public int getNumInputArrays(){
        return numInputArrays;
    }

    /** The number of output (arrays) for this network */
    public int getNumOutputArrays(){
        return numOutputArrays;
    }

    /** Initialize the ComputationGraph network */
    public void init(){
        //Initialization: create the GraphVertex objects, based on configuration structure

        throw new UnsupportedOperationException("Not implemented");
    }

    /** Pretrain network with a single input and single output */
    public void pretrain(DataSetIterator iter){
        if(numInputArrays != 1 || numOutputArrays != 1) throw new UnsupportedOperationException("Cannot train ComputationGraph network with "
            + " multiple inputs or outputs using a DataSetIterator");

        throw new UnsupportedOperationException("Not implemnted");
    }

    /** Pretrain network with multiple inputs and/or outputs */
    public void pretrain(Object multipleInputOutputIterator){
        throw new UnsupportedOperationException("Not implemnted");
    }

    public void fit(DataSet dataSet){
        if(numInputArrays != 1 || numOutputArrays != 1) throw new UnsupportedOperationException("Cannot train ComputationGraph network with "
                + " multiple inputs or outputs using a DataSet");

        throw new UnsupportedOperationException("Not implemnted");
    }

    public void fit(DataSetIterator dataSetIterator){
        if(numInputArrays != 1 || numOutputArrays != 1) throw new UnsupportedOperationException("Cannot train ComputationGraph network with "
                + " multiple inputs or outputs using a DataSetIterator");

        throw new UnsupportedOperationException("Not implemented");
    }

    public void fit(Object multipleInputOutputIterator){

        throw new UnsupportedOperationException("Not implemented");
    }

    public void fit(INDArray[] inputs, INDArray[] labels ){
//        setInputs(inputs);
//        setLabels(labels);


        if(configuration.isPretrain()){

            throw new UnsupportedOperationException("Not implemented");
        }

        if(configuration.isBackprop()){
            if(configuration.getBackpropType() == BackpropType.TruncatedBPTT){

                throw new UnsupportedOperationException("Not implemented");
            } else {
                if( solver == null) {
                    solver = new Solver.Builder()
                            .configure(conf())
//                            .listeners(getListeners())
                            .model(this).build();
                }

                solver.optimize();
            }
        }
    }

    /** Calculate a topological sort order for the vertices in the graph.
     * Note that this is used for
     * (a) working out what order to do forward pass,
     * (b) what order to do backprop (i.e., reverse of this)
     * (c) order to flatten parameters
     *  */
    private int[] topologicalSortOrder(){
        //https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
        int[] out = new int[vertices.length];
        int outCounter = 0;

        throw new UnsupportedOperationException("Not implemented");
    }



    @Override
    public ComputationGraph clone(){

        throw new UnsupportedOperationException("Not implemented");
    }

    //------------------------------------------------------
    //Model methods:

    @Override
    public void fit() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public double score() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void computeGradientAndScore() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void accumulateScore(double accum) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public INDArray params() {
        List<INDArray> list = new ArrayList<>(layerCount);
        for( int i=0; i<topologicalOrder.length; i++ ){
            if(!vertices[i].hasLayer()) continue;

            Layer l = vertices[i].getLayer();
            list.add(l.params());
        }

        return Nd4j.toFlattened('f',list);
    }

    @Override
    public int numParams() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public int numParams(boolean backwards) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void setParams(INDArray params) {
        int idx = 0;
        for( int i=0; i<topologicalOrder.length; i++ ){
            if(!vertices[i].hasLayer()) continue;

            Layer layer = vertices[i].getLayer();
            int range = (layer instanceof BasePretrainNetwork ?
                    ((BasePretrainNetwork<?>)layer).numParamsBackprop() : layer.numParams());
            INDArray get = params.get(NDArrayIndex.point(0),NDArrayIndex.interval(idx, range + idx));
            layer.setParams(get);
            idx += range;
        }
    }

    @Override
    public void applyLearningRateScoreDecay() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void fit(INDArray data) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
    }

    @Override
    public int batchSize() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public NeuralNetConfiguration conf() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public INDArray input() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void validateInput() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public INDArray getParam(String param) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void initParams() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public Map<String, INDArray> paramTable() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void setParam(String key, INDArray val) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("Not implemnted");
    }


}
