package org.nd4j.autodiff.samediff.optimize;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.optimizations.ConstantFunctionOptimizations;

import java.util.Arrays;
import java.util.List;

/**
 *
 * @author Alex Black
 */
@Slf4j
public class GraphOptimizer {

    public static List<OptimizerSet> defaultOptimizations(){
        return Arrays.<OptimizerSet>asList(
                new ConstantFunctionOptimizations()
        );
    }

    public static SameDiff optimize(SameDiff graph){
        return optimize(graph, defaultOptimizations());
    }

    public static SameDiff optimize(SameDiff graph, List<OptimizerSet> optimizations){
        SameDiff sd = graph.dup();

        ArrayHolder cArr = sd.getConstantArrays();
        ArrayHolder vArr = sd.getVariablesArrays();

        OptimizationConfig config = new OptimizationConfig();   //TODO

        for( int i=0; i<3; i++ ) {  //Run multiple times - one run isn't enough, as some more optimizations may need to be applied to the output of earlier optimizations
            for (OptimizerSet s : optimizations) {
                List<Optimizer> l = s.getOptimizers();
                for(Optimizer o : l ){
                    for(SameDiffOp op : sd.getOps().values()) {
                        boolean applied = o.checkAndApply(sd, config, op, cArr, vArr);
                        if(applied) {
                            log.info("Operation was applied: ");
                        }
                    }
                }
            }
        }

        int constBefore = 0;
        int constAfter = 0;
        int varBefore = 0;
        int varAfter = 0;
        int arrBefore = 0;
        int arrAfter = 0;

        for(SDVariable v : graph.variables()){
            switch(v.getVariableType()){
                case VARIABLE:
                    varBefore++;
                    break;
                case CONSTANT:
                    constBefore++;
                    break;
                case ARRAY:
                    arrBefore++;
                    break;
                case PLACEHOLDER:
                    break;
            }
        }

        for(SDVariable v : sd.variables()){
            switch(v.getVariableType()){
                case VARIABLE:
                    varAfter++;
                    break;
                case CONSTANT:
                    constAfter++;
                    break;
                case ARRAY:
                    arrAfter++;
                    break;
                case PLACEHOLDER:
                    break;
            }
        }


        log.info("Total variables: {} before, {} after", graph.getVariables().size(), sd.getVariables().size());
        log.info("Constant variables: {} before, {} after", constBefore, constAfter);
        log.info("Array type variables: {} before, {} after", arrBefore, arrAfter);
        log.info("Variable type variables: {} before, {} after", varBefore, varAfter);
        log.info("Ops: {} before, {} after", graph.getOps().size(), sd.getOps().size());

        return sd;
    }

}
