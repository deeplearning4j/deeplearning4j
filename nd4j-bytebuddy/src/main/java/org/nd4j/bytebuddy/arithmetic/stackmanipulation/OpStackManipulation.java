package org.nd4j.bytebuddy.arithmetic.stackmanipulation;

import net.bytebuddy.implementation.bytecode.StackManipulation;
import org.nd4j.bytebuddy.arithmetic.ByteBuddyIntArithmetic;

/**
 * Stack manipulations for integer
 * arithmetic:
 * add
 * sub
 * mul
 * div
 * mod
 *
 * @author Adam Gibson
 */
public class OpStackManipulation {
    /**
     * Stack manipulation for addition
     * @return
     */
    public static StackManipulation add() {return ByteBuddyIntArithmetic.IntegerAddition.INSTANCE;}

    /**
     * Stack manipulation for subtraction
     * @return
     */
    public static StackManipulation sub() {return ByteBuddyIntArithmetic.IntegerSubtraction.INSTANCE;}

    /**
     * Stack manipulation for multiplication
     * @return
     */
    public static StackManipulation mul() {return ByteBuddyIntArithmetic.IntegerMultiplication.INSTANCE;}

    /**
     * Stack manipulation for division
     * @return
     */
    public static StackManipulation div() {return ByteBuddyIntArithmetic.IntegerDivision.INSTANCE;}

    /**
     * Stack manipulation for mod
     * @return
     */
    public static StackManipulation mod() {return ByteBuddyIntArithmetic.IntegerMod.INSTANCE;}

}
