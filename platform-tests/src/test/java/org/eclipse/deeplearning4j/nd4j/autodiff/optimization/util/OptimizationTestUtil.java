package org.eclipse.deeplearning4j.nd4j.autodiff.optimization.util;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * TODO:
 * - Add ability to track which optimization functions exactly were applied!
 */
public class OptimizationTestUtil {

    private OptimizationTestUtil(){ }

    public static SameDiff testOptimization(OptTestConfig config) {
        Preconditions.checkNotNull(config.getTempFolder(), "Temp folder should be specified before running test");

        List<OptimizerSet> optimizerSets = config.getOptimizerSets();
        if(optimizerSets == null)
            optimizerSets = GraphOptimizer.defaultOptimizations();
        OptimizationRecordingDebugger debugger = new OptimizationRecordingDebugger();

        //
        Map<String,INDArray> ph = config.getPlaceholders();
        List<String> outputs = config.getOutputs();
        SameDiff original = config.getOriginal();
        SameDiff copy = original.dup();
        SameDiff optimized = GraphOptimizer.optimize(original, outputs, optimizerSets, debugger);

        //Check that SOMETHING changed in the optimized - number of constants, variables, or ops; or the settings for ops; or the values of some arrays
        //TODO
        boolean sameNumConst = original.getConstantArrays().size() == optimized.getConstantArrays().size();
        boolean sameNumVars = original.getVariablesArrays().size() == optimized.getVariablesArrays().size();
        boolean sameNumSDVars = original.getVariables().size() == optimized.getVariables().size();
        boolean sameNumOps = original.getOps().size() == optimized.getOps().size();

        if(sameNumConst && sameNumVars && sameNumSDVars && sameNumOps){


            throw new IllegalStateException("Did not detect any changes to the graph structure after optimization (but check is AS YET WIP)");
        }

        //Check that optimizations we expected to be applied were in fact applied:
        Map<String,Class<? extends Optimizer>> mustApply = config.getMustApply();
        Map<String,Optimizer> applied = debugger.getApplied();
        for(String s : mustApply.keySet()){
            assertTrue("Expected optimizer of type " + mustApply.get(s).getSimpleName() + " to be applied to op " + s,
                    applied.containsKey(s));
        }


        //Second: check that they all produce the same
        //TODO this won't work for random ops!
        Map<String,INDArray> origOut = original.output(ph, outputs);
        Map<String,INDArray> copyOut = copy.output(ph, outputs);
        Map<String,INDArray> optimizedOut = optimized.output(ph, outputs);

        assertEquals(copyOut, origOut);
        assertEquals(copyOut, optimizedOut);

        File f = new File(config.getTempFolder(), "optimized.sd");
        optimized.save(f, true);

        SameDiff loaded = SameDiff.load(f, true);
        Map<String,INDArray> loadedOut = loaded.output(ph, outputs);
        assertEquals(copyOut, loadedOut);

        //TODO add support for training checks!
        //This is especially important for updaters... if we permute the weights, we should permute the updater state also

        //Check that nothing has changed (from the user API perspective) for the original graph
        //i.e.,
        for(SDVariable v : copy.variables()){
            SDVariable ov = original.getVariable(v.name());

            assertEquals(v.dataType(), ov.dataType());
            assertEquals(v.getVariableType(), ov.getVariableType());
            if(v.getVariableType() == VariableType.CONSTANT || v.getVariableType() == VariableType.VARIABLE){
                INDArray arrCopy = v.getArr();
                INDArray arrOrig = ov.getArr();
                assertEquals(arrCopy, arrOrig);
            }

        }

        return optimized;
    }

}