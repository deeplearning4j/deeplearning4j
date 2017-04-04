package org.nd4j.autodiff.bigdecimal;

import java.util.*;
import java.math.*;


public class Factorial {

    static Vector<Ifactor> a = new Vector<Ifactor>();


    public Factorial() {
        if (a.size() == 0) {
            a.add(Ifactor.ONE);
            a.add(Ifactor.ONE);
        }
    } /* ctor */


    public BigInteger at(int n) {
        /*
         * extend the internal list if needed.
         */
        growto(n);
        return a.elementAt(n).n;
    } /* at */


    public Ifactor toIfactor(int n) {
        /*
         * extend the internal list if needed.
         */
        growto(n);
        return a.elementAt(n);
    } /* at */


    private void growto(int n) {
        /*
         * extend the internal list if needed. Size to be 2 for n<=1, 3 for n<=2
         * etc.
         */
        while (a.size() <= n) {
            final int lastn = a.size() - 1;
            final Ifactor nextn = new Ifactor(lastn + 1);
            a.add(a.elementAt(lastn).multiply(nextn));
        }
    } /* growto */

} /* Factorial */
