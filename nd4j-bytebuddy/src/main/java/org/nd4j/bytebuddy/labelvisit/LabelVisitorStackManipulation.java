package org.nd4j.bytebuddy.labelvisit;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.jar.asm.Label;
import net.bytebuddy.jar.asm.MethodVisitor;

/**
 * @author Adam Gibson
 */
public class LabelVisitorStackManipulation implements StackManipulation {
    private Label label;

    public LabelVisitorStackManipulation(Label label) {
        this.label = label;
    }

    @Override
    public boolean isValid() {
        return true;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext) {
        methodVisitor.visitLabel(label);
        return new Size(0,0);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        LabelVisitorStackManipulation that = (LabelVisitorStackManipulation) o;

        return !(label != null ? !label.equals(that.label) : that.label != null);

    }

    @Override
    public int hashCode() {
        return label != null ? label.hashCode() : 0;
    }
}
