package org.nd4j.linalg.api.ops;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class BaseIndexAccumulation extends BaseOp implements IndexAccumulation {
    protected int finalResult;

    public BaseIndexAccumulation() {
    }

    /**
     * Initialize with the given
     * input, pairwise transform, result, and number
     * of elements
     *
     * @param x the input
     * @param y the pairwise transform
     * @param z the result
     * @param n the number of elements
     */
    public BaseIndexAccumulation(INDArray x, INDArray y, INDArray z, int n) {
        super(x, y, z, n);
        init();
    }

    public BaseIndexAccumulation(INDArray x, INDArray y, int n) {
        this(x, y, x, n);
    }

    public BaseIndexAccumulation(INDArray x) {
        this(x, null, x, x.length());
    }

    public BaseIndexAccumulation(INDArray x, INDArray y) {
        this(x, y, x, x.length());
    }

    @Override
    public double zeroDouble() {
        return 0.0;
    }

    @Override
    public float zeroFloat(){
        return 0.0f;
    }

    @Override
    public Pair<Double,Integer> zeroPair(){
        return new Pair<>(zeroDouble(),-1);
    }

    @Override
    public IComplexNumber zeroComplex() {
        return Nd4j.createComplexNumber(0.0, 0.0);
    }

    private void init() {
        init(x, y, x, x.length());
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, int n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[]{zeroDouble()};
    }

    @Override
    public int combineSubResults(double first, int idxFirst, double second, int idxSecond){
        return update(first,idxFirst,second,idxSecond);
    }

    @Override
    public int combineSubResults(float first, int idxFirst, float second, int idxSecond){
        return update(first,idxFirst,second,idxSecond);
    }

    @Override
    public Pair<Double,Integer> combineSubResults(Pair<Double,Integer> first, Pair<Double,Integer> second){
        int idxFirst = first.getSecond();
        int idxSecond = second.getSecond();
        int idxOut = update(first.getFirst(),idxFirst,second.getFirst(),idxSecond);
        return (idxOut == idxFirst ? first : second);
    }

    @Override
    public void setFinalResult( int idx ){
        this.finalResult = idx;
    }

    @Override
    public int getFinalResult(){
        return finalResult;
    }
}
