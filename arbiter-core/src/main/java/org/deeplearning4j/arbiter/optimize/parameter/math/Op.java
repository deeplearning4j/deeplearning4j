package org.deeplearning4j.arbiter.optimize.parameter.math;

public enum Op {
    ADD, SUB, MUL, DIV;


    //Package private
    <T extends Number> T doOp(T first, T second){
        if(first instanceof Integer || first instanceof Long){
            long result;
            switch (this){
                case ADD:
                    result = Long.valueOf(first.longValue() + second.longValue());
                    break;
                case SUB:
                    result = Long.valueOf(first.longValue() - second.longValue());
                    break;
                case MUL:
                    result = Long.valueOf(first.longValue() * second.longValue());
                    break;
                case DIV:
                    result = Long.valueOf(first.longValue() / second.longValue());
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown op: " + this);
            }
            if(first instanceof Long){
                return (T)Long.valueOf(result);
            } else {
                return (T)Integer.valueOf((int)result);
            }
        } else if(first instanceof Double || first instanceof Float){
            double result;
            switch (this){
                case ADD:
                    result = Double.valueOf(first.doubleValue() + second.doubleValue());
                    break;
                case SUB:
                    result = Double.valueOf(first.doubleValue() - second.doubleValue());
                    break;
                case MUL:
                    result = Double.valueOf(first.doubleValue() * second.doubleValue());
                    break;
                case DIV:
                    result = Double.valueOf(first.doubleValue() / second.doubleValue());
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown op: " + this);
            }
            if(first instanceof Double){
                return (T)Double.valueOf(result);
            } else {
                return (T)Float.valueOf((float)result);
            }
        } else {
            throw new UnsupportedOperationException("Not supported type: only Integer, Long, Double, Float supported" +
                    " here. Got type: " + first.getClass());
        }
    }
}
