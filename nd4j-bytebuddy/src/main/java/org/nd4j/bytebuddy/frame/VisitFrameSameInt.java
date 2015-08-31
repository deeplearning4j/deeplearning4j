package org.nd4j.bytebuddy.frame;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * @author Adam Gibson
 */
public class VisitFrameSameInt implements StackManipulation {
    private int nLocal = -1;
    private  int nStack = -1;

    public VisitFrameSameInt(int nLocal, int nStack) {
        this.nLocal = nLocal;
        this.nStack = nStack;
    }

    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
        methodVisitor.visitFrame(Opcodes.F_SAME1, nLocal, null, nStack, new Object[] {Opcodes.INTEGER});
        return new Size(0,0);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        VisitFrameSameInt that = (VisitFrameSameInt) o;

        if (nLocal != that.nLocal) return false;
        return nStack == that.nStack;

    }

    @Override
    public int hashCode() {
        int result = nLocal;
        result = 31 * result + nStack;
        return result;
    }
}
