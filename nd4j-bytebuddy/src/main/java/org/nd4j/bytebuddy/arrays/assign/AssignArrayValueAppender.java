package org.nd4j.bytebuddy.arrays.assign;

import com.sun.org.apache.xpath.internal.compiler.OpCodes;
import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.implementation.FieldAccessor;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.StackSize;
import net.bytebuddy.implementation.bytecode.collection.ArrayAccess;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.jar.asm.Opcodes;
import net.bytebuddy.pool.TypePool;

/**
 * Handles loading the proper index
 * To assign an element to an array the byte code looks like the following:
 *   public static void main(java.lang.String[]);
 Code:
 0: invokestatic  #2                  // Method returnArr:()[I <- this is our reference
 3: astore_1 //store the variable in 1
 4: aload_1 //load the actual varaible 1
 5: iconst_0 push a 0 on the stack (now we're here asking for the index of the array based on the given variable
 6: iconst_5 //push a 5 on to the stack (this is the value we want to assign in the array)
 7: iastore //do the actual store operation, we don't do this here
 8: return
 }

 * This is intended to be used with the following method signature:
 * void(int[] arr,int index,int value)
 *
 * @author Adam Gibson
 */
public class AssignArrayValueAppender implements ByteCodeAppender {
    private int index;
    private int newVal;
    private static TypePool typePool = TypePool.Default.ofClassPath();

    /**
     *
     * @param index the index to enqueue
     * @param newVal the new value to assign to
     *               the index in the array
     */
    public AssignArrayValueAppender(int index, int newVal) {
        this.index = index;
        this.newVal = newVal;
    }

    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        StackManipulation compound =  assignOperation();
        StackManipulation.Size size = compound.apply(methodVisitor, implementationContext);
        StackManipulation store = ArrayAccess.of(typePool.describe("int").resolve()).store();
        size = size.aggregate(store.apply(methodVisitor, implementationContext));
        StackManipulation returnOp =  MethodReturn.VOID;
        size = size.aggregate(returnOp.apply(methodVisitor, implementationContext));
        return new Size(size.getMaximalSize(), instrumentedMethod.getStackSize());
    }

    public StackManipulation assignOperation() {
        //enqueue the new value as the item to set
        StackManipulation arg0 = MethodVariableAccess.REFERENCE.loadOffset(1);
        //load the value to be assigned
        StackManipulation val = IntegerConstant.forValue(newVal);
        //load the index
        StackManipulation indexToAssign = IntegerConstant.forValue(index);
       //set the return type
        StackManipulation.Compound compound = new StackManipulation.Compound(arg0,indexToAssign,val);
        return compound;
    }
}
