/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.util;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;

/**
 * Реализация полной интервальной арифметики Каухера.
 * Пока без правильных округлений и проверки правильности создания интервалов.
 *
 * @author dmtrl
 */
public class IntervalNumber implements AbstractNumber {
    //    private static BigDecimal eps=new BigDecimal(1e-24);
    private static MathContext mcLow = new MathContext(9, RoundingMode.FLOOR);
    private static MathContext mcHigh = new MathContext(9, RoundingMode.CEILING);
    protected BigDecimal low;
    protected BigDecimal high;

    public IntervalNumber() {
        low = BigDecimal.ZERO;
        high = BigDecimal.ZERO;
    }

    public IntervalNumber(double l, double r) {
        this(new BigDecimal(l), new BigDecimal(r));
    }

    public IntervalNumber(String l, String r) {
        this(new BigDecimal(l), new BigDecimal(r));
    }


    public IntervalNumber(BigDecimal l, BigDecimal r) {
        low = l;
        high = r;
    }

    public IntervalNumber(IntervalNumber a) {
        this(a.low, a.high);
    }

    public IntervalNumber(AbstractNumber a) {
        if (a instanceof IntervalNumber) {
            IntervalNumber _a = (IntervalNumber) a;
            low = _a.low;
            high = _a.high;
        }
    }

    public static IntervalNumber neg(IntervalNumber x, IntervalNumber y) {
        return new IntervalNumber(x.add(new IntervalNumber(y.high.negate(mcLow), y.low.negate(mcLow))));
    }

    public BigDecimal inf() {
        return low;
    }

    public BigDecimal sup() {
        return high;
    }

    @Override
    public AbstractNumber add(AbstractNumber _b) {
        if (this == null || _b == null) throw new ArithmeticException("Null pointer argument in add");
        if (this instanceof IntervalNumber && _b instanceof IntervalNumber) {
            IntervalNumber a = this;
            IntervalNumber b = (IntervalNumber) _b;
            return new IntervalNumber(a.low.add(b.low, mcLow), a.high.add(b.high, mcHigh));
        }
        throw new ArithmeticException("Unexpected arithmetical error!!!");
    }

    @Override
    public AbstractNumber sub(AbstractNumber _b) {
        if (this == null || _b == null) throw new ArithmeticException("Null pointer argument in sub");
        if (this instanceof IntervalNumber && _b instanceof IntervalNumber) {
            IntervalNumber a = this;
            IntervalNumber b = (IntervalNumber) _b;
            return new IntervalNumber(a.low.subtract(b.high, mcLow), a.high.subtract(b.low, mcLow));
        }
        throw new ArithmeticException("Unexpected arithmetical error!!!");
    }

    @Override
    public AbstractNumber mult(AbstractNumber _b) {
        if (this == null || _b == null) throw new ArithmeticException("Null pointer argument in mult");
        if (this instanceof IntervalNumber && _b instanceof IntervalNumber) {
            IntervalNumber a = this;
            IntervalNumber b = (IntervalNumber) _b;
            switch (a.getTypeOfInterval()) {
                case P:
                    switch (b.getTypeOfInterval()) {
                        case P: {
                            return new IntervalNumber(a.low.multiply(b.low, mcLow), a.high.multiply(b.high, mcHigh));
                        }
                        case Z: {
                            return new IntervalNumber(a.high.multiply(b.low, mcLow), a.high.multiply(b.high, mcHigh));
                        }
                        case negP: {
                            return new IntervalNumber(a.high.multiply(b.low, mcLow), a.low.multiply(b.high, mcHigh));
                        }
                        case dualZ: {
                            return new IntervalNumber(a.low.multiply(b.low, mcLow), a.low.multiply(b.high, mcHigh));
                        } // Куда округлять на границах?
                    }
                case Z:
                    switch (b.getTypeOfInterval()) {
                        case P: {
                            return new IntervalNumber(a.low.multiply(b.high, mcLow), a.high.multiply(b.high, mcHigh));
                        }
                        case Z: {
                            return new IntervalNumber(a.low.multiply(b.high, mcLow).min(a.high.multiply(b.low, mcLow)), a.low.multiply(b.low, mcHigh).max(a.high.multiply(b.high, mcHigh)));
                        }
                        case negP: {
                            return new IntervalNumber(a.high.multiply(b.low, mcLow), a.low.multiply(b.low, mcHigh));
                        }
                        case dualZ: {
                            return new IntervalNumber(BigDecimal.ZERO, BigDecimal.ZERO);
                        }
                    }
                case negP:
                    switch (b.getTypeOfInterval()) {
                        case P: {
                            return new IntervalNumber(a.low.multiply(b.high, mcLow), a.high.multiply(b.low, mcHigh));
                        }
                        case Z: {
                            return new IntervalNumber(a.low.multiply(b.high, mcLow), a.low.multiply(b.low, mcHigh));
                        }
                        case negP: {
                            return new IntervalNumber(a.high.multiply(b.high, mcLow), a.low.multiply(b.low, mcHigh));
                        }
                        case dualZ: {
                            return new IntervalNumber(a.high.multiply(b.high, mcLow), a.high.multiply(b.low, mcHigh));
                        }
                    }
                case dualZ:
                    switch (b.getTypeOfInterval()) {
                        case P: {
                            return new IntervalNumber(a.low.multiply(b.low, mcLow), a.high.multiply(b.low, mcHigh));
                        }
                        case Z: {
                            return new IntervalNumber(BigDecimal.ZERO, BigDecimal.ZERO);
                        }
                        case negP: {
                            return new IntervalNumber(a.high.multiply(b.high, mcLow), a.low.multiply(b.high, mcHigh));
                        }
                        case dualZ: {
                            return new IntervalNumber(a.low.multiply(b.low, mcLow).max(a.high.multiply(b.high, mcLow)), a.low.multiply(b.high, mcHigh).min(a.high.multiply(b.low, mcHigh)));
                        }
                    }
            }
        }
        throw new ArithmeticException("Unexpected arithmetical error!!!");
    }

    @Override
    public AbstractNumber div(AbstractNumber _b) {
        if (this == null || _b == null) throw new ArithmeticException("Null pointer argument in div");
        if (this instanceof IntervalNumber && _b instanceof IntervalNumber) {
            IntervalNumber a = this;
            IntervalNumber b = (IntervalNumber) _b;
            if (!b.zeroInPro())
                return new IntervalNumber(mult(new IntervalNumber(BigDecimal.ONE.divide(b.high, mcLow), BigDecimal.ONE.divide(b.low, mcHigh))));
            else throw new ArithmeticException("Division by interval in Z");
        }
        throw new ArithmeticException("Unexpected arithmetical error!!!");
    }

 /*   @Override
    public AbstractNumber sqrt() {
         switch (getTypeOfInterval()){
            case P: {return new IntervalNumber(low, low);}
            case Z: {}
            case negP: {}
            case dualZ: {throw new ArithmeticException("Operation sqrt not defined at Z, negP & dualZ");}
        }
        throw new ArithmeticException("Unexpected arithmetical exception");
        // throw new UnsupportedOperationException("Not supported yet.");
    }
*/

    @Override
    public String toString() {
        return "[" + low.toPlainString() + ", " + high.toPlainString() + "]";
    }


    @Override
    public boolean equals(Object obj) {
        if (obj == null) return false;
        if (obj instanceof IntervalNumber && this instanceof IntervalNumber) {
            IntervalNumber interval = (IntervalNumber) obj;
            IntervalNumber thisInterval = this;
            return Math.abs(thisInterval.low.doubleValue() - interval.low.doubleValue()) <= 1e-7
                    && Math.abs(thisInterval.high.doubleValue() - interval.high.doubleValue()) <= 1e-7;
        }
        return false;
    }


    //
    public typeOfInterval getTypeOfInterval() {
        if (low.compareTo(BigDecimal.ZERO) <= 0 && 0 <= high.compareTo(BigDecimal.ZERO)) return typeOfInterval.Z;
        else if (low.compareTo(BigDecimal.ZERO) >= 0 && high.compareTo(BigDecimal.ZERO) >= 0) return typeOfInterval.P;
        else if (low.compareTo(BigDecimal.ZERO) <= 0 && high.compareTo(BigDecimal.ZERO) <= 0)
            return typeOfInterval.negP;
        else if (high.compareTo(BigDecimal.ZERO) <= 0 && 0 <= low.compareTo(BigDecimal.ZERO))
            return typeOfInterval.dualZ;
        return null;
    }

    public BigDecimal abs() {
        BigDecimal al = low.abs(mcLow);
        BigDecimal ar = high.abs(mcHigh);
        return al.compareTo(ar) > 0 ? al : ar;
    }

    public IntervalNumber dual() {
        return new IntervalNumber(high, low);
    }

    public IntervalNumber pro() {
        return low.compareTo(high) < 0 ? new IntervalNumber(this) : dual();
    }

    public IntervalNumber opp() {
        return new IntervalNumber(low.negate(mcLow), high.negate(mcHigh));
    }

    public IntervalNumber inv() {
        return new IntervalNumber(BigDecimal.ONE.divide(low, mcLow), BigDecimal.ONE.divide(high, mcHigh));
    }

    public IntervalNumber innerSub(IntervalNumber y) {
        return new IntervalNumber(low.subtract(y.low, mcLow), high.subtract(y.high, mcHigh));
    }

    public boolean zeroInPro() {  // "Новый" сособ сравнения
        IntervalNumber y = pro();
        return y.low.compareTo(BigDecimal.ZERO) <= 0 && 0 <= y.high.compareTo(BigDecimal.ZERO);
    }

    public IntervalNumber innerDiv(IntervalNumber y) {
        if (!y.zeroInPro())
            return new IntervalNumber(mult(y.inv()));
        else
            throw new ArithmeticException("Division by interval in Z");
    }

    public static enum typeOfInterval {P, Z, negP, dualZ}
}