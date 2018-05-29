package org.nd4j.autodiff.gradcheck;

import com.google.common.collect.ImmutableSet;
import com.google.common.reflect.ClassPath;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.util.*;

/**
 * Gradient check utility
 *
 * @author Adam Gibson
 */
@Slf4j
public class GradCheckUtil {

    private static final boolean DEFAULT_PRINT = true;
    private static final boolean DEFAULT_EXIT_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-5;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-5;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-6;

    private static Map<Class,Integer> countPerClass = new HashMap<>();

    //Collect coverage information
    static {
        initializeCoverage();
    }

    private static void initializeCoverage(){
        //Scan classpath to find all DifferentialFunction instances, so tensorflow/onnx mappings can be made
        //We're assuming here that all instances with such mappings are defined in ND4J
        //As of 04/2018 all DifferentialFunction classes are defined in org.nd4j.linalg.api.ops - with the exception
        // of ILossFunction instances, which don't have TF/Onnx import working anyway
        ImmutableSet<ClassPath.ClassInfo> info;
        try {
            //Dependency note: this ClassPath class was added in Guava 14
            info = com.google.common.reflect.ClassPath.from(DifferentialFunctionClassHolder.class.getClassLoader())
                    .getTopLevelClassesRecursive("org.nd4j.linalg.api.ops");
        } catch (IOException e){
            //Should never happen
            throw new RuntimeException(e);
        }


        int count = 0;
        for(ClassPath.ClassInfo c : info){
            //Load method: Loads (but doesn't link or initialize) the class.
            Class<?> clazz;
            try{
                clazz = Class.forName(c.getName());
            } catch (ClassNotFoundException e){
                //Should never happen as  this was found on the classpath
                throw new RuntimeException(e);
            }


            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface() || !DifferentialFunction.class.isAssignableFrom(clazz))
                continue;

            if(DifferentialFunction.class.isAssignableFrom(clazz)){
                countPerClass.put(clazz, 0);
            }
        }
    }

    public static void logCoverageInformation( boolean logSeen, boolean logUnseen, boolean excludeBackpropOps ){

        int countSeen = 0;
        if(logSeen){
            log.info(" --- Gradient Checks: Classes Seen in Tests ---");
            for(Map.Entry<Class,Integer> e : countPerClass.entrySet()){
                if(e.getValue() > 0 && (!excludeBackpropOps || isBackpropOp(e.getKey()))){
                    log.info("GradientCheck: Seen {} instances of op {}", e.getValue(), e.getKey().getName());
                    countSeen++;
                }
            }
        }

        if(logUnseen){
            log.info(" --- Gradient Checks: Classes NOT Seen in Tests ---");
            for(Map.Entry<Class,Integer> e : countPerClass.entrySet()){
                if(e.getValue() == 0 && (!excludeBackpropOps || isBackpropOp(e.getKey()))){
                    log.info("GradientCheck: NO instances of op {}", e.getKey().getName());
                }
            }
        }


        int total;
        if(excludeBackpropOps){
            total = 0;
            for(Class c : countPerClass.keySet()){
                if(isBackpropOp(c)){
                    total++;
                }
            }
        } else {
            total = countPerClass.size();
        }

        double frac = countSeen / (double)total;
        String fracPc = String.format("%.2f",frac*100.0);
        log.info("*****************************************************");
        log.info("Gradient Checks: {} of {} classes checked ({}% coverage - {} backprop ops)", countSeen, total, fracPc,
                (excludeBackpropOps ? "excluding" : "including"));
        log.info("*****************************************************");
    }

    private static boolean isBackpropOp(Class<?> c){
        String name = c.getSimpleName();
        return name.contains("Bp");
    }


    /**
     *
     * @param function
     * @param epsilon
     * @param maxRelError
     * @param print
     * @param inputParameters
     * @return
     */
    public static boolean checkGradients(
            SDVariable function,
            SDVariable wrt,
            double epsilon,
            double maxRelError,
            boolean print,
            Map<String,INDArray> inputParameters) {
        //Basic sanity checks on input:
        if (epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError);

        DataBuffer.Type dataType = DataTypeUtil.getDtypeFromContext();
        if (dataType != DataBuffer.Type.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Datatype is not set to double precision ("
                    + "is: " + dataType + "). Double precision must be used for gradient checks. Set "
                    + "DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE); before using GradientCheckUtil");
        }

        /**
         * Need to pass in the exact gradient.
         * This is obtained from executing a subgraph
         * with just the gradient part to get the exact values.
         * You then run the comparison vs the approximation from that.
         *
         * To obtain the comparison/computing the values,  use the below routine
         */


        SameDiff sameDiff = function.getSameDiff();
        //get just the subgraph for the graph
        SameDiff opExec = SameDiff.create(sameDiff);

        INDArray[] eval = opExec.eval(inputParameters);
        int totalNFailures = 0;
        double maxError = 0.0;

        for(Map.Entry<String,INDArray> entry : inputParameters.entrySet()) {
            long nParams = entry.getValue().length();
            INDArray params = entry.getValue().dup();
            for (int i = 0; i < nParams; i++) {
                INDArray zeros = Nd4j.create(nParams);
                zeros.putScalar(i,epsilon / 2.0);

                //(w+epsilon): Do forward pass and score
                double origValue = params.getDouble(i);
                params.putScalar(i, origValue + epsilon);
                Map<String, INDArray> evalParams = new HashMap<>();
                for (Map.Entry<String, INDArray> entry2 : inputParameters.entrySet()) {
                    if (!entry2.getKey().equals(entry.getKey())) {
                        evalParams.put(entry2.getKey(), entry2.getValue());
                    } else {
                        evalParams.put(entry.getKey(), params);
                    }
                }

                /**
                 * Need to figure out how I want to extract
                 * parameters for computing the delta..
                 *
                 */
                INDArray[] plusParams = sameDiff.eval(evalParams);


                INDArray[] minusParams = sameDiff.eval(evalParams);


                /**
                 * Difference between new params and old
                 */
                INDArray[] newDifferences = new INDArray[minusParams.length];
                for (int j = 0; j < newDifferences.length; j++) {
                    newDifferences[j] = plusParams[j].subi(minusParams[j]).divi(epsilon);
                }

                double diff = plusParams[plusParams.length - 1].sumNumber().doubleValue() - minusParams[minusParams.length - 1].sumNumber().doubleValue();
                double eps = diff / epsilon;
                double correctVal = eval[eval.length - 1].sumNumber().doubleValue();
                double gradDiff = Math.abs(correctVal - eps);
                if(gradDiff > maxRelError)
                    totalNFailures++;
                if (print) {
                    long nPass = nParams - totalNFailures;
                    log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, "
                            + totalNFailures + " failed. Largest relative error = " + maxError);
                }
            }
        }

        return totalNFailures == 0;
    }

    public static boolean checkGradients(SameDiff sd){
        return checkGradients(sd, DEFAULT_PRINT, DEFAULT_EXIT_FIRST_FAILURE);
    }

    public static boolean checkGradients(SameDiff sd, boolean print, boolean exitOnFirstFailure){
        return checkGradients(sd, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, print, exitOnFirstFailure);
    }


    public static boolean checkGradients(SameDiff sd, double eps, double maxRelError, double minAbsError, boolean print,
                                         boolean exitOnFirstFailure){

        //Collect coverage information:
        collectCoverageInformation(sd);


        //Check data type:
        if(Nd4j.dataType() != DataBuffer.Type.DOUBLE){
            throw new IllegalStateException("Data type must be set to double");
        }

        Set<String> fnOutputs = new HashSet<>();
        for(DifferentialFunction f : sd.functions()){
            for(SDVariable s : f.outputVariables()){
                fnOutputs.add(s.getVarName());
            }
        }

        //Check that all *input* SDVariables have arrays associated with them
        for(SDVariable s : sd.variables()){
            if (fnOutputs.contains(s.getVarName())) {
                //This is not an input to the graph
                continue;
            }
            if(s.getArr() == null){
                throw new IllegalStateException("Variable \"" + s.getVarName() + "\" does not have array associated with it");
            }
        }

        //Do forward pass, check that output is a scalar:
        INDArray out = sd.execAndEndResult();
        if(out.length() != 1){
            throw new IllegalStateException("Output variable is not a scalar - has shape " + Arrays.toString(out.shape()));
        }

        //TODO also check that all inputs are non-zero (otherwise: consider out = sum(x * y) with all x and y being 0
        // in this case, gradients of x and y are all 0 too

        sd.execBackwards();
        Map<String,INDArray> grad = new HashMap<>();
        for(SDVariable v : sd.variables()){
            if (fnOutputs.contains(v.getVarName())) {
                //This is not an input to the graph
                continue;
            }
            SDVariable g = sd.grad(v.getVarName());
            if(g == null){
                throw new IllegalStateException("Null gradient variable for \"" + v.getVarName() + "\"");
            }
            INDArray ga = g.getArr();
            if(ga == null){
                throw new IllegalStateException("Null gradient array encountered for variable: " + v.getVarName());
            }
            if(!Arrays.equals(v.getArr().shape(), g.getArr().shape())){
                throw new IllegalStateException("Gradient shape does not match variable shape for variable \"" +
                    v.getVarName() + "\": shape " + Arrays.toString(v.getArr().shape()) + " vs. gradient shape " +
                    Arrays.toString(ga.shape()));
            }
            grad.put(v.getVarName(), ga.dup());
        }

        //Validate gradients for each variable:
        int totalNFailures = 0;
        int totalCount = 0;
        double maxError = 0.0;
        for(SDVariable s : sd.variables()){
            if (fnOutputs.contains(s.getVarName())) {
                //This is not an input to the graph
                continue;
            }

            String name = s.getVarName();
            INDArray a = s.getArr();
            long n = a.length();
            if(print){
                log.info("Starting test for variable \"{}\" with {} values", s.getVarName(), n);
            }

            NdIndexIterator iter = new NdIndexIterator('c',a.shape());

            int i=0;
            while(iter.hasNext()){
                val idx = iter.next();
                String strIdx = null;
                if(print){
                    strIdx = Arrays.toString(idx).replaceAll(" ","");
                }

                totalCount++;
                double orig = a.getDouble(idx);
                a.putScalar(idx, orig+eps);
                double scorePlus = sd.execAndEndResult().getDouble(0);
                a.putScalar(idx, orig-eps);
                double scoreMinus = sd.execAndEndResult().getDouble(0);
                a.putScalar(idx, orig);

                double numericalGrad = (scorePlus - scoreMinus) / (2 * eps);
                double analyticGrad = grad.get(s.getVarName()).getDouble(idx);

                if (Double.isInfinite(numericalGrad) || Double.isNaN(numericalGrad)) {
                    throw new IllegalStateException("Numerical gradient was " + numericalGrad + " for variable \"" + name
                            + "\", parameter " + i + " of " + n + " (position: " + strIdx + ")");
                }
                if (Double.isInfinite(analyticGrad) || Double.isNaN(analyticGrad)) {
                    throw new IllegalStateException("Analytic (SameDiff) gradient was " + analyticGrad + " for variable \"" + name
                            + "\", parameter " + i + " of " + n + " (position: " + strIdx + ")");
                }


                double relError;
                if(numericalGrad == 0.0 && analyticGrad == 0.0){
                    relError = 0.0;
                } else {
                    relError = Math.abs(analyticGrad - numericalGrad) / (Math.abs(Math.abs(analyticGrad) + Math.abs(numericalGrad)));
                }

                if (relError > maxError)
                    maxError = relError;

                if (relError > maxRelError || Double.isNaN(relError)) {
                    double absError = Math.abs(analyticGrad - numericalGrad);
                    if (absError < minAbsError) {
                        if(print) {
                            log.info("Param " + i + " (" + name + strIdx + ") passed: grad= " + analyticGrad
                                    + ", numericalGrad= " + numericalGrad + ", relError= " + relError
                                    + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsError);
                        }
                    } else {
                        if (print)
                            log.info("Param " + i + " (" + name + strIdx + ") FAILED: grad= " + analyticGrad
                                    + ", numericalGrad= " + numericalGrad + ", relError= " + relError
                                    + ", absError=" + absError
                                    + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                        if (exitOnFirstFailure)
                            return false;
                        totalNFailures++;
                    }
                } else if (print) {
                    log.info("Param " + i + " (" + name + strIdx + ") passed: grad= " + analyticGrad + ", numericalGrad= "
                            + numericalGrad + ", relError= " + relError);
                }
                i++;
            }
        }

        if (print) {
            int nPass = totalCount - totalNFailures;
            log.info("GradCheckUtil.checkGradients(): " + totalCount + " params checked, " + nPass + " passed, "
                    + totalNFailures + " failed. Largest relative error = " + maxError);
        }

        return totalNFailures == 0;
    }

    private static void collectCoverageInformation(SameDiff sd){
        DifferentialFunction[] functions = sd.functions();
        for(DifferentialFunction df : functions){
            countPerClass.put(df.getClass(), countPerClass.get(df.getClass()) + 1);
        }
    }
}
