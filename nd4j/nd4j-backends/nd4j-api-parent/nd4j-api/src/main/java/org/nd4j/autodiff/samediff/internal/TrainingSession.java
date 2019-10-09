package org.nd4j.autodiff.samediff.internal;

import com.sun.prism.paint.Gradient;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.*;

/**
 * TrainingSession extends InferenceSession, to add training-specific functionality:
 * - Application of regularization (L1, L2, weight decay etc)
 * - Inline updating of variables
 */
@Slf4j
public class TrainingSession extends InferenceSession2 {

    protected TrainingConfig config;
    protected Map<String,String> gradVarToVarMap;
    protected Map<String, GradientUpdater> updaters;

    public TrainingSession(SameDiff sameDiff) {
        super(sameDiff);
    }

    public void trainingIteration(TrainingConfig config, Map<String, INDArray> placeholders, Set<String> paramsToTrain, Map<String, GradientUpdater> updaters,
                                  MultiDataSet batch, List<Listener> listeners, At at){
        this.config = config;
        this.updaters = updaters;

        List<String> requiredActivations = new ArrayList<>();
        gradVarToVarMap = new HashMap<>();       //Key: gradient variable. Value: variable that the key is gradient for
        for(String s : paramsToTrain){
            Preconditions.checkState(sameDiff.hasVariable(s), "SameDiff instance does not have a variable with name \"%s\"", s);
            SDVariable v = sameDiff.getVariable(s);
            Preconditions.checkState(v.getVariableType() == VariableType.VARIABLE, "Can only train VARIABLE type variable - \"%s\" has type %s",
                    s, v.getVariableType());
            SDVariable grad = sameDiff.getVariable(s).getGradient();
            Preconditions.checkState(grad != null, "No gradient is defined for variable \"%s\"", s);

            requiredActivations.add(grad.getVarName());

            gradVarToVarMap.put(grad.getVarName(), s);
        }

        List<String> outputVars = new ArrayList<>(gradVarToVarMap.keySet());    //TODO this should be empty, and grads calculated in requiredActivations

//        Map<String,INDArray> m = output(Collections.<String>emptyList(), placeholders, batch, requiredActivations, listeners, at );
        Map<String,INDArray> m = output(outputVars, placeholders, batch, requiredActivations, listeners, at );
    }

    @Override
    public INDArray[] getOutputs(SameDiffOp op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                 Set<String> constAndPhInputs, List<Listener> listeners, At at, MultiDataSet batch, Set<String> allReqVariables) {

        INDArray[] out = super.getOutputs(op, outputFrameIter, opInputs, allIterInputs, constAndPhInputs, listeners, at, batch, allReqVariables);

        List<String> outputs = op.getOutputsOfOp();
        for(String s : outputs){
            if(gradVarToVarMap.containsKey(s)){
                String varName = gradVarToVarMap.get(s);
                log.info("Calculated gradient for variable \"{}\": (grad var name: \"{}\")", varName, s);

                Variable gradVar = sameDiff.getVariables().get(s);
                if(gradVar.getInputsForOp() != null && gradVar.getInputsForOp().isEmpty()){
                    //Should be rare, and we should handle this by tracking dependencies, and only update when safe
                    // (i.e., dependency tracking)
                    throw new IllegalStateException("Op depends on gradient variable: " + s + " for variable " + varName);
                }

                GradientUpdater u = updaters.get(varName);
                Preconditions.checkState(u != null, "No updater found for variable \"%s\"", varName);

                Variable var = sameDiff.getVariables().get(varName);
                int outIdx = op.getOutputsOfOp().indexOf(s);
                INDArray gradArr = out[outIdx];
                INDArray paramArr = var.getVariable().getArr();

                //TODO Parameter sharing case - this is *probably* actually safe, as backprop will add gradients
                // in this case. i.e., gradient for x, dL/dx, is only available after all components have been added
                // as x.gradient is updated to account for all uses, during backprop
                // Put another way: gradient variable representing dL/dx must account for all uses of x - and hence accounts
                // for gradient accumulation - already
                List<String> inputsToOps = var.getInputsForOp();
                //Preconditions.checkState(inputsToOps.size() == 2, "FIXME: Possible parameter sharing detected (not handled yet)");

                //Pre-updater regularization (L1, L2)
                List<Regularization> r = config.getRegularization();
                if(r != null && r.size() > 0){
                    double lr = config.getUpdater().hasLearningRate() ? config.getUpdater().getLearningRate(at.iteration(), at.epoch()) : 1.0;
                    for (Regularization reg : r) {
                        if (reg.applyStep() == Regularization.ApplyStep.BEFORE_UPDATER) {
                            reg.apply(paramArr, gradArr, lr, at.iteration(), at.epoch());
                        }
                    }
                }

                u.applyUpdater(gradArr, at.iteration(), at.epoch());

                //Post-apply regularization (weight decay)
                if (r != null && r.size() > 0) {
                    double lr = config.getUpdater().hasLearningRate() ? config.getUpdater().getLearningRate(at.iteration(), at.epoch()) : 1.0;
                    for (Regularization reg : r) {
                        if (reg.applyStep() == Regularization.ApplyStep.POST_UPDATER) {
                            reg.apply(paramArr, gradArr, lr, at.iteration(), at.epoch());
                            /*
                            if (hasListeners) {
                                double score = reg.score(param, iterCount, epochCount);
                                if (!regScore.containsKey(reg.getClass())) {
                                    regScore.put(reg.getClass(), new AtomicDouble());
                                }
                                regScore.get(reg.getClass()).addAndGet(score);
                            }
                             */
                        }
                    }
                }

                //Update:
                if (config.isMinimize()) {
                    paramArr.subi(gradArr);
                } else {
                    paramArr.addi(gradArr);
                }
                log.info("Applied updater to gradient and updated variable: {}", varName);
            }
        }

        return out;
    }
}
