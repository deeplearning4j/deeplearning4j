package org.nd4j.bytebuddy.util;

/**
 * Utilities for op codes
 *
 * @author Adam Gibson
 */
public class OpCodeUtil {

    /**
     * Get the a load reference code starting at 42.
     * See:
     * https://github.com/raphw/byte-buddy/blob/master/byte-buddy-dep/src/test/java/net/bytebuddy/test/utility/MoreOpcodes.java#L40-40
     * and:
     * http://stackoverflow.com/questions/4641416/jvm-instruction-aload-0-in-the-main-method-points-to-args-instead-of-this.
     *
     * For byte code instructions see:
     * https://en.wikipedia.org/wiki/Java_bytecode_instruction_listings
     *
     * Convert the binary to base 10 to get the op code using something like:
     * http://www.unitconversion.org/numbers/binary-to-base-10-conversion.html
     *
     * @param ref the id to load
     * @return the op code instruction for loading a particular
     * reference on the stack
     */
    public static int getAloadInstructionForReference(int ref) {
          return 42 + ref;
    }

}
