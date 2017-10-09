package org.nd4j.linalg.api.ops;

import lombok.Getter;
import lombok.val;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.StringUtils;
import org.nd4j.linalg.primitives.ImmutablePair;
import org.nd4j.linalg.util.HashUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Basic implementation for CustomOp
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DynamicCustomOp implements CustomOp {
    private String opName;
    @Getter private List<INDArray> inputArguments;
    @Getter private List<INDArray> outputArguments;
    @Getter private List<Double> tArguments = new ArrayList<>();
    @Getter private List<Integer> iArguments = new ArrayList<>();
    @Getter private boolean inplaceCall;
    @Getter private long hash;

    protected DynamicCustomOp(String opName) {
        this.opName = opName;
    }

    /**
     * This method returns op name as string
     *
     * @return
     */
    @Override
    public String opName() {
        return opName;
    }

    /**
     * This method returns LongHash of the opName()
     *
     * @return
     */
    @Override
    public long opHash() {
        return hash;
    }

    /**
     * This method takes custom opname, and return Op Builder instance
     * @param opName
     * @return
     */
    public static Builder builder(String opName) {
        val map = Nd4j.getExecutioner().getCustomOperations();
        val lcName = opName.toLowerCase();
        val desc = map.get(lcName);

        if (desc == null)
            throw new ND4JIllegalStateException("Unknown operations requested: [" + opName + "]");

        return new Builder(opName, desc.getHash(), desc.getNumInputs(), desc.getNumOutputs(), desc.isAllowsInplace(), desc.getNumTArgs(), desc.getNumIArgs());
    }

    @Override
    public List<int[]> calculateOutputShape() {
        return Nd4j.getExecutioner().calculateOutputShape(this);
    }

    public static class Builder {
        protected String opName;
        protected int numInputs;
        protected int numOutputs;
        protected int numTArguments;
        protected int numIArguments;
        protected boolean inplaceCall;
        protected boolean inplaceAllowed;
        protected long opHash;

        private List<INDArray> inputArguments = new ArrayList<>();
        private List<INDArray> outputArguments = new ArrayList<>();
        private List<Double> tArguments = new ArrayList<>();
        private List<Integer> iArguments = new ArrayList<>();

        protected Builder(String opName, long hash, int numInputs, int numOutputs, boolean inplaceAllowed, int numTArguments, int numIArguments) {
            this.opHash = hash;
            this.opName = opName;
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            this.numIArguments = numIArguments;
            this.numTArguments = numTArguments;
            this.inplaceAllowed = inplaceAllowed;
        }

        /**
         * This methos takes arbitrary number of input INDArrays in, as Op input
         *
         * PLEASE NOTE: this method does NOT validate lengths/shapes.
         *
         * @param inputs
         * @return
         */
        public Builder setInputs(INDArray... inputs) {
            // if we have positive value as numInputs - we should ensure equal amount of arguments
            if (numInputs >= 0) {
                if (inputs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numInputs + " arguments. Null was passed instead.");

                if (numInputs != inputs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numInputs + " arguments, but " + inputs.length + " was passed to constructor");
            }

            for (val in: inputs)
                inputArguments.add(in);

            return this;
        }

        /**
         * This methos takes arbitrary number of output INDArrays in, to store operation result
         *
         * PLEASE NOTE: this method does NOT validate lengths/shapes.
         *
         * @param outputs
         * @return
         */
        public Builder setOutputs(INDArray... outputs) {
            if (numOutputs >= 0) {
                if (outputs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numOutputs + " arguments. Null was passed instead.");

                if (numOutputs != outputs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numOutputs + " arguments, but " + outputs.length + " was passed to constructor");
            }

            for (val in: outputs)
                outputArguments.add(in);

            return this;
        }

        public Builder callInplace(boolean reallyCall) {
            if (reallyCall && !inplaceAllowed)
                throw new ND4JIllegalStateException("Resuested op can't be called inplace");

            this.inplaceCall = reallyCall;
            return this;
        }

        /**
         * This methos takes arbitrary number of Integer arguments for op,
         *
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param iargs
         * @return
         */
        public Builder setIntegerArguments(Integer... iargs) {
            if (numIArguments >= 0) {
                if (iargs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. Null was passed instead.");

                if (numIArguments != iargs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments, but " + iargs.length + " was passed to constructor");
            }

            for (val in: iargs)
                iArguments.add(in);

            return this;
        }

        /**
         * This methos takes arbitrary number of Integer arguments for op,
         *
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param arg
         * @return
         */
        public Builder setIntegerArguments(int arg) {
            if (numIArguments != 1 && numIArguments > 0)
                throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. One arg was passed instead.");

            iArguments.add(arg);

            return this;
        }

        /**
         * This methos takes arbitrary number of Integer arguments for op,
         *
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @param iargs
         * @return
         */
        public Builder setIntegerArguments(int... iargs) {
            if (numIArguments >= 0) {
                if (iargs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments. Null was passed instead.");

                if (numIArguments != iargs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numIArguments + " integer arguments, but " + iargs.length + " was passed to constructor");
            }

            for (val in: iargs)
                iArguments.add(in);

            return this;
        }

        /**
         * This methos takes arbitrary number of Double arguments for op,
         *
         * PLEASE NOTE: this method does NOT validate values.
         *
         * @return
         */
        public Builder setFloatingPointArguments(Double... targs) {
            if (numTArguments >= 0) {
                if (targs == null)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numTArguments + " integer arguments. Null was passed instead.");

                if (numTArguments != targs.length)
                    throw new ND4JIllegalStateException("CustomOp [" + opName + "] expects " + numTArguments + " integer arguments, but " + targs.length + " was passed to constructor");
            }

            for (val in: targs)
                tArguments.add(in);

            return this;
        }





        public DynamicCustomOp build() {
            // Eventually we probably will lift this restriction
            //if (!inplaceCall && outputArguments.size() == 0)
            //    throw new ND4JIllegalStateException("If operation is not-inplace, it must have outputs defined");

            val result = new DynamicCustomOp(opName);
            result.inputArguments = inputArguments;
            result.outputArguments = outputArguments;
            result.iArguments = iArguments;
            result.tArguments = tArguments;
            result.inplaceCall = inplaceCall;
            result.hash = opHash;

            return result;
        }
    }
}
