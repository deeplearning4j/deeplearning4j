package org.nd4j.bytebuddy.gotoop;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.Label;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;

/**
 * Go to instruction
 *
 * @author Adam Gibson
 */
public class GoToOp implements StackManipulation {
    private Label label;

    public GoToOp(Label label) {
        this.label = label;
    }

    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
        methodVisitor.visitJumpInsn(Opcodes.GOTO,label);
        return new Size(0,0);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        GoToOp goToOp = (GoToOp) o;

        return !(label != null ? !label.equals(goToOp.label) : goToOp.label != null);

    }

    @Override
    public int hashCode() {
        return label != null ? label.hashCode() : 0;
    }
}
