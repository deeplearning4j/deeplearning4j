package org.nd4j.bytebuddy.frame;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;
import org.nd4j.bytebuddy.shape.OffsetMapper;

/**
 * @author Adam Gibson
 */
public class VisitFrameFullInt implements StackManipulation {
    private int nLocal = -1;
    private  int nStack = -1;


    public VisitFrameFullInt(int nLocal, int nStack) {
        this.nLocal = nLocal;
        this.nStack = nStack;
    }

    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
        methodVisitor.visitFrame(Opcodes.F_FULL, nLocal, new Object[] {OffsetMapper.class.getName().replace(".","/"), Opcodes.INTEGER, "[I", "[I", "[I"}
                , nStack, new Object[] {Opcodes.INTEGER, Opcodes.INTEGER});
        return new Size(0,0);
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        VisitFrameFullInt that = (VisitFrameFullInt) o;

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
