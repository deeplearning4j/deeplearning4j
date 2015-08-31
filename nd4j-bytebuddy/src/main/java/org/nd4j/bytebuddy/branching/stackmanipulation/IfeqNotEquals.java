package org.nd4j.bytebuddy.branching.stackmanipulation;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.Label;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 *
 * Calls the branching op code
 * with the operand 14 (not equals)
 *
 * @author Adam Gibson
 */
public class IfeqNotEquals implements StackManipulation,Opcodes {
    private Label label;

    public IfeqNotEquals(Label label) {
        this.label = label;
    }

    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
        methodVisitor.visitJumpInsn(Opcodes.IFEQ, label);
        return new StackManipulation.Size(-1,-1);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        IfeqNotEquals that = (IfeqNotEquals) o;

        return !(label != null ? !label.equals(that.label) : that.label != null);

    }

    @Override
    public int hashCode() {
        return label != null ? label.hashCode() : 0;
    }
}
