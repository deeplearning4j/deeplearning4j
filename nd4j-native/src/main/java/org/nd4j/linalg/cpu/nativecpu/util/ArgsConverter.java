package org.nd4j.linalg.cpu.nativecpu.util;

import org.nd4j.linalg.api.ops.Op;

/**
 * @author Adam Gibson
 */
public class ArgsConverter {
    public static double[] convertExtraArgsDouble(Op op) {
        if(op.extraArgs() == null)
            return null;
        else {
            double[] ret = new double[op.extraArgs().length];
            for(int i = 0; i < ret.length; i++)
                ret[i] = Double.valueOf(op.extraArgs()[i].toString());
            return ret;
        }
    }

    public static float[] convertExtraArgsFloat(Op op) {
        if(op.extraArgs() == null)
            return null;
        else {
            float[] ret = new float[op.extraArgs().length];
            for(int i = 0; i < ret.length; i++)
                ret[i] = Float.valueOf(op.extraArgs()[i].toString());
            return ret;
        }
    }
}
