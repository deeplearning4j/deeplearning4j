package org.nd4j.autodiff.samediff.internal;

import com.sun.prism.paint.Gradient;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.primitives.AtomicDouble;

import java.util.*;

/**
 * TrainingSession extends InferenceSession, to add training-specific functionality:
 * - Application of regularization (L1, L2, weight decay etc)
 * - Inline updating of variables
 * - Calculation of regularization scores (Score for L1, L2, etc)
 */
@Slf4j
public class TrainingSession extends InferenceSession2 {

    protected TrainingConfig config;
    protected Map<String,String> gradVarToVarMap;
    protected Map<String, GradientUpdater> updaters;
    protected Map<String,Integer> lossVarsToLossIdx;
    protected double[] currIterLoss;
    protected Map<Class<?>, AtomicDouble> currIterRegLoss;


    public TrainingSession(SameDiff sameDiff) {
        super(sameDiff);
    }

    public Loss trainingIteration(TrainingConfig config, Map<String, INDArray> placeholders, Set<String> paramsToTrain, Map<String, GradientUpdater> updaters,
                                  MultiDataSet batch, List<String> lossVariables, List<Listener> listeners, At at){
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

        //Set up losses
        lossVarsToLossIdx = new LinkedHashMap<>();
        List<String> lossVars;
        currIterLoss = new double[lossVariables.size()];
        currIterRegLoss = new HashMap<>();
        for( int i=0; i<lossVariables.size(); i++ ){
            lossVarsToLossIdx.put(lossVariables.get(i), i);
        }

        //Do training iteration
        List<String> outputVars = new ArrayList<>(gradVarToVarMap.keySet());    //TODO this should be empty, and grads calculated in requiredActivations
        Map<String,INDArray> m = output(outputVars, placeholders, batch, requiredActivations, listeners, at );


        double[] finalLoss = new double[currIterLoss.length + currIterRegLoss.size()];
        System.arraycopy(currIterLoss, 0, finalLoss, 0, currIterLoss.length);
        if (currIterRegLoss.size() > 0) {
            lossVars = new ArrayList<>(lossVariables.size() + currIterRegLoss.size());
            lossVars.addAll(lossVariables);
            int s = currIterRegLoss.size();
            //Collect regularization losses
            for (Map.Entry<Class<?>, AtomicDouble> entry : currIterRegLoss.entrySet()) {
                lossVars.add(entry.getKey().getSimpleName());
                finalLoss[s] = entry.getValue().get();
            }
        } else {
            lossVars = lossVariables;
        }

        Loss loss = new Loss(lossVariables, finalLoss);
        if (listeners != null) {
            for (Listener l : listeners) {
                l.iterationDone(sameDiff, at, batch, loss);
            }
        }

        return loss;
    }

    @Override
    public INDArray[] getOutputs(SameDiffOp op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                 Set<String> constAndPhInputs, List<Listener> listeners, At at, MultiDataSet batch, Set<String> allReqVariables) {

        INDArray[] out = super.getOutputs(op, outputFrameIter, opInputs, allIterInputs, constAndPhInputs, listeners, at, batch, allReqVariables);

        List<String> outputs = op.getOutputsOfOp();
        int outIdx = 0;
        for(String s : outputs){
            //If this is a loss variable - record it
            if(lossVarsToLossIdx.containsKey(s)){
                int lossIdx = lossVarsToLossIdx.get(s);
                INDArray arr = out[outIdx];
                double l = arr.isScalar() ? arr.getDouble(0) : arr.sumNumber().doubleValue();
                currIterLoss[lossIdx] += l;
            }

            //If this is a gradient variable - apply the updater and update the parameter array in-line
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

                if (listeners != null) {
                    for (Listener l : listeners) {
                        if (l.isActive(at.operation()))
                            l.preUpdate(sameDiff, at, var, gradArr);
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

            outIdx++;
        }

        return out;
    }
}
