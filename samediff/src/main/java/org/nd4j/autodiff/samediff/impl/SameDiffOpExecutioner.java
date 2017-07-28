package org.nd4j.autodiff.samediff.impl;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.cache.TADManager;

import java.util.*;

public class SameDiffOpExecutioner implements OpExecutioner {
    private Map<String,INDArray> ops;
    private Set<INDArray> arrays;
    private SameDiff sameDiff;
    private Map<INDArray,String> arrayToId;

    public SameDiffOpExecutioner() {
        ops = new HashMap<>();
        arrayToId = new IdentityHashMap<>();
        arrays = new HashSet<>();
        sameDiff = SameDiff.create();
    }

    /**
     * This method returns name of the last invoked op
     *
     * @return
     */
    @Override
    public String getLastOp() {
        return null;
    }

    /**
     * Execute the operation
     *
     * @param op the operation to execute
     */
    @Override
    public Op exec(Op op) {
        return null;
    }

    /**
     * Iterate over every row of every slice
     *
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllRows(Op op) {

    }

    /**
     * Iterate over every column of every slice
     *
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllColumns(Op op) {

    }

    /**
     * Execute a TransformOp and return the result
     *
     * @param op the operation to execute
     */
    @Override
    public INDArray execAndReturn(TransformOp op) {
        return null;
    }

    /**
     * Execute and return the result from an accumulation
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    @Override
    public Accumulation execAndReturn(Accumulation op) {
        return null;
    }

    /**
     * Execute and return the result from an accumulation
     *
     * @param op            the operation to execute
     * @param biasCorrected
     * @return the accumulated result
     */
    @Override
    public Accumulation execAndReturn(Variance op, boolean biasCorrected) {
        return null;
    }

    /**
     * Execute and return the result from an index accumulation
     *
     * @param op the index accumulation operation to execute
     * @return the accumulated index
     */
    @Override
    public IndexAccumulation execAndReturn(IndexAccumulation op) {
        return null;
    }

    /**
     * Execute and return the result from a scalar op
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    @Override
    public INDArray execAndReturn(ScalarOp op) {
        return null;
    }

    /**
     * Execute and return the result from a vector op
     *
     * @param op
     */
    @Override
    public INDArray execAndReturn(BroadcastOp op) {
        return null;
    }

    /**
     * Execute the operation along 1 or more dimensions
     *
     * @param op        the operation to execute
     * @param dimension
     */
    @Override
    public Op exec(Op op, int... dimension) {
        return null;
    }

    /**
     * Execute an accumulation along one or more dimensions
     *
     * @param accumulation the accumulation
     * @param dimension    the dimension
     * @return the accumulation op
     */
    @Override
    public INDArray exec(Accumulation accumulation, int... dimension) {
        return null;
    }

    /**
     * Execute an broadcast along one or more dimensions
     *
     * @param broadcast the accumulation
     * @param dimension the dimension
     * @return the broadcast op
     */
    @Override
    public INDArray exec(BroadcastOp broadcast, int... dimension) {
        return null;
    }

    /**
     * Execute an accumulation along one or more dimensions
     *
     * @param accumulation  the accumulation
     * @param biasCorrected
     * @param dimension     the dimension  @return the accmulation op
     */
    @Override
    public INDArray exec(Variance accumulation, boolean biasCorrected, int... dimension) {
        return null;
    }

    /**
     * Execute an index accumulation along one or more dimensions
     *
     * @param indexAccum the index accumulation operation
     * @param dimension  the dimension/s to execute along
     * @return result
     */
    @Override
    public INDArray exec(IndexAccumulation indexAccum, int... dimension) {
        return null;
    }

    /**
     * Execute and return  a result
     * ndarray from the given op
     *
     * @param op the operation to execute
     * @return the result from the operation
     */
    @Override
    public INDArray execAndReturn(Op op) {
        return null;
    }

    /**
     * Get the execution mode for this
     * execuioner
     *
     * @return the execution mode for this executioner
     */
    @Override
    public ExecutionMode executionMode() {
        return null;
    }

    /**
     * Set the execution mode
     *
     * @param executionMode the execution mode
     */
    @Override
    public void setExecutionMode(ExecutionMode executionMode) {

    }

    /**
     * Execute MetaOp
     *
     * @param op
     */
    @Override
    public void exec(MetaOp op) {

    }

    /**
     * Execute GridOp
     *
     * @param op
     */
    @Override
    public void exec(GridOp op) {

    }

    @Override
    public void exec(Aggregate op) {

    }

    /**
     * This method executes previously built batch
     *
     * @param batch
     */
    @Override
    public <T extends Aggregate> void exec(Batch<T> batch) {

    }

    /**
     * This method takes abritrary sized list of aggregates, and packs them into batches
     *
     * @param batch
     */
    @Override
    public void exec(List<Aggregate> batch) {

    }

    /**
     * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
     *
     * @param op
     */
    @Override
    public INDArray exec(RandomOp op) {
        return null;
    }

    /**
     * This method executes specific RandomOp against specified RNG
     *
     * @param op
     * @param rng
     */
    @Override
    public INDArray exec(RandomOp op, Random rng) {
        return null;
    }

    /**
     * This method return set of key/value and
     * key/key/value objects,
     * describing current environment
     *
     * @return
     */
    @Override
    public Properties getEnvironmentInformation() {
        return null;
    }

    /**
     * This method specifies desired profiling mode
     *
     * @param mode
     */
    @Override
    public void setProfilingMode(ProfilingMode mode) {

    }

    /**
     * Ths method returns current profiling
     *
     * @return
     */
    @Override
    public ProfilingMode getProfilingMode() {
        return null;
    }

    /**
     * This method returns TADManager instance used for this OpExecutioner
     *
     * @return
     */
    @Override
    public TADManager getTADManager() {
        return null;
    }

    /**
     * This method prints out environmental information returned by getEnvironmentInformation() method
     */
    @Override
    public void printEnvironmentInformation() {

    }

    /**
     * This method ensures all operations that supposed to be executed at this moment, are executed.
     */
    @Override
    public void push() {

    }

    /**
     * This method ensures all operations that supposed to be executed at this moment, are executed and finished.
     */
    @Override
    public void commit() {

    }

    /**
     * This method encodes array as thresholds, updating input array at the same time
     *
     * @param input
     * @param threshold
     * @return encoded array is returned
     */
    @Override
    public INDArray thresholdEncode(INDArray input, double threshold) {
        return null;
    }

    /**
     * This method encodes array as thresholds, updating input array at the same time
     *
     * @param input
     * @param threshold
     * @param boundary  @return encoded array is returned
     */
    @Override
    public INDArray thresholdEncode(INDArray input, double threshold, Integer boundary) {
        return null;
    }

    /**
     * This method decodes thresholds array, and puts it into target array
     *
     * @param encoded
     * @param target
     * @return target is returned
     */
    @Override
    public INDArray thresholdDecode(INDArray encoded, INDArray target) {
        return null;
    }

    /**
     * This method returns number of elements affected by encoder
     *
     * @param indArray
     * @param target
     * @param threshold
     * @return
     */
    @Override
    public long bitmapEncode(INDArray indArray, INDArray target, double threshold) {
        return 0;
    }

    @Override
    public INDArray bitmapEncode(INDArray indArray, double threshold) {
        return null;
    }

    @Override
    public INDArray bitmapDecode(INDArray encoded, INDArray target) {
        return null;
    }
}
