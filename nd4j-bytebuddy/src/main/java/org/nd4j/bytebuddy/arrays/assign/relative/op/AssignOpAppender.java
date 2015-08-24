package org.nd4j.bytebuddy.arrays.assign.relative.op;

import net.bytebuddy.description.method.MethodDescription;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.collection.ArrayAccess;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.jar.asm.MethodVisitor;
import net.bytebuddy.pool.TypePool;

/**
 * Handles loading the proper index
 * To assign an element to an array the byte code looks like the following:
 *   public static void main(java.lang.String[]);
 Code:
 0: invokestatic  #2                  // Method returnArr:()[I <- this is our reference
 3: astore_1 //store the variable in 1
 4: aload_1 //load the actual variable 1
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
public class AssignOpAppender implements ByteCodeAppender {
    private static TypePool typePool = TypePool.Default.ofClassPath();



    @Override
    public Size apply(MethodVisitor methodVisitor, Implementation.Context implementationContext, MethodDescription instrumentedMethod) {
        //resolve the type to store in the array and retrieve the store command
        StackManipulation store = ArrayAccess.of(typePool.describe("int").resolve()).store();
        StackManipulation.Size size = store.apply(methodVisitor, implementationContext);
        return new Size(size.getMaximalSize(), instrumentedMethod.getStackSize());
    }


}
