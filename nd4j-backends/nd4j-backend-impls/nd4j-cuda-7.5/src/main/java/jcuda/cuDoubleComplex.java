/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

package jcuda;

/**
 * Java port of the CUDA complex number structure for double
 * precision numbers
 */
public class cuDoubleComplex
{
    /** The real part of the complex number */
    public double x;

    /** The imaginary part of the complex number */
    public double y;

    /* Private constructor */
    private cuDoubleComplex()
    {
    }

    /**
     * Returns the real part of the given complex number.
     *
     * @param x The complex number whose real part should be returned
     * @return The real part of the given complex number
     */
    public static double cuCreal (cuDoubleComplex x)
    {
        return x.x;
    }

    /**
     * Returns the imaginary part of the given complex number.
     *
     * @param x The complex number whose imaginary part should be returned
     * @return The imaginary part of the given complex number
     */
    public static double cuCimag (cuDoubleComplex x)
    {
        return x.y;
    }

    /**
     * Creates a new complex number consisting of the given real and
     * imaginary part.
     *
     * @param r The real part of the complex number
     * @param i The imaginary part of the complex number
     * @return A complex number with the given real and imaginary part
     */
    public static cuDoubleComplex cuCmplx (double r, double i)
    {
        cuDoubleComplex res = new cuDoubleComplex();
        res.x = r;
        res.y = i;
        return res;
    }

    /**
     * Returns the complex conjugate of the given complex number.
     *
     * @param x The complex number whose complex conjugate should be returned
     * @return The complex conjugate of the given complex number
     */
    public static  cuDoubleComplex cuConj (cuDoubleComplex x)
    {
        return cuCmplx (cuCreal(x), -cuCimag(x));
    }

    /**
     * Returns a new complex number that is the sum of the given
     * complex numbers.
     *
     * @param x The first addend
     * @param y The second addend
     * @return The sum of the given addends
     */
    public static  cuDoubleComplex cuCadd (cuDoubleComplex x, cuDoubleComplex y)
    {
        return cuCmplx (cuCreal(x) + cuCreal(y), cuCimag(x) + cuCimag(y));
    }

    /**
     * Returns the product of the given complex numbers.<br />
     * <br />
     * Original comment:<br />
     * <br />
     * This implementation could suffer from intermediate overflow even though
     * the final result would be in range. However, various implementations do
     * not guard against this (presumably to avoid losing performance), so we
     * don't do it either to stay competitive.
     *
     * @param x The first factor
     * @param y The second factor
     * @return The product of the given factors
     */
    public static cuDoubleComplex cuCmul (cuDoubleComplex x, cuDoubleComplex y)
    {
        cuDoubleComplex prod;
        prod = cuCmplx ((cuCreal(x) * cuCreal(y)) - (cuCimag(x) * cuCimag(y)),
                        (cuCreal(x) * cuCimag(y)) + (cuCimag(x) * cuCreal(y)));
        return prod;
    }

    /**
     * Returns the quotient of the given complex numbers.<br />
     * <br />
     * Original comment:<br />
     * <br />
     * This implementation guards against intermediate underflow and overflow
     * by scaling. Such guarded implementations are usually the default for
     * complex library implementations, with some also offering an unguarded,
     * faster version.
     *
     * @param x The dividend
     * @param y The divisor
     * @return The quotient of the given complex numbers
     */
    public static  cuDoubleComplex cuCdiv (cuDoubleComplex x, cuDoubleComplex y)
    {
        cuDoubleComplex quot;
        double s = Math.abs(cuCreal(y)) + Math.abs(cuCimag(y));
        double oos = 1.0 / s;
        double ars = cuCreal(x) * oos;
        double ais = cuCimag(x) * oos;
        double brs = cuCreal(y) * oos;
        double bis = cuCimag(y) * oos;
        s = (brs * brs) + (bis * bis);
        oos = 1.0 / s;
        quot = cuCmplx (((ars * brs) + (ais * bis)) * oos,
                        ((ais * brs) - (ars * bis)) * oos);
        return quot;
    }

    /**
     * Returns the absolute value of the given complex number.<br />
     * <br />
     * Original comment:<br />
     * <br />
     * This implementation guards against intermediate underflow and overflow
     * by scaling. Otherwise the we'd lose half the exponent range. There are
     * various ways of doing guarded computation. For now chose the simplest
     * and fastest solution, however this may suffer from inaccuracies if sqrt
     * and division are not IEEE compliant.
     *
     * @param x The complex number whose absolute value should be returned
     * @return The absolute value of the given complex number
     */
    public static double cuCabs (cuDoubleComplex x)
    {
        double p = cuCreal(x);
        double q = cuCimag(x);
        double r;
        if (p == 0) return q;
        if (q == 0) return p;
        p = Math.sqrt(p);
        q = Math.sqrt(q);
        if (p < q) {r = p; p = q; q = r;}
        r = q / p;
        return p * Math.sqrt (1.0f + r * r);
    }

    /**
     * Returns a String representation of this complex number.
     *
     * @return A String representation of this complex number
     */
    public String toString()
    {
        return "("+x+","+y+")";
    }
}

