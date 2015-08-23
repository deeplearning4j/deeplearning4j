package org.nd4j.bytebuddy.shape;

import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import org.nd4j.bytebuddy.arithmetic.relative.op.RelativeOperationImplementation;
import org.nd4j.bytebuddy.arithmetic.stackmanipulation.OpStackManipulation;
import org.nd4j.bytebuddy.arrays.assign.relative.op.AssignOpImplementation;
import org.nd4j.bytebuddy.arrays.create.simple.SimpleCreateArrayImplementation;
import org.nd4j.bytebuddy.arrays.create.stackmanipulation.CreateIntArrayStackManipulation;
import org.nd4j.bytebuddy.arrays.retrieve.relative.RelativeRetrieveArrayImplementation;
import org.nd4j.bytebuddy.arrays.stackmanipulation.ArrayStackManipulation;
import org.nd4j.bytebuddy.constant.ConstantIntImplementation;
import org.nd4j.bytebuddy.createint.StoreIntImplementation;
import org.nd4j.bytebuddy.loadref.relative.RelativeLoadDeclaredReferenceImplementation;
import org.nd4j.bytebuddy.method.integer.relative.RelativeLoadIntParamImplementation;
import org.nd4j.bytebuddy.method.reference.LoadReferenceParamImplementation;
import org.nd4j.bytebuddy.returnref.ReturnAppender;
import org.nd4j.bytebuddy.returnref.ReturnAppenderImplementation;
import org.nd4j.bytebuddy.stackmanipulation.StackManipulationImplementation;
import org.nd4j.bytebuddy.storeint.stackmanipulation.StoreIntStackManipulation;
import org.nd4j.bytebuddy.storeref.StoreImplementation;
import org.nd4j.bytebuddy.storeref.stackmanipulation.StoreRefStackManipulation;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class ShapeMapper {

    /**
     * Get an implementation of
     * ind2sub
     * @param ordering the order to iterate in
     * @param rank the rank of the array
     * @return the implementation (in byte code) of
     * ind2subseq
     *
     */
    public static Implementation getInd2Sub2(char ordering, int rank) {


        /**
         * Given signature:
         *  int[] map(int[] shape,int index,int numIndices,char ordering);
         Load int param grabs numIndices because the instance
         variable stack indexing starts with this at zero

         4 here represents creating a variable and storing
         the value of the last argument in the method in the value

         */
        int retArrayIndex = 4;

        //the index of the variable we use to start denomination
        //the index of the last method parameter
        int linearIndexArg = 2;
        //the index of the total number of indexes
        int totalindexarg = 3;

        //"this" is 0 and the shape array is the first argument
        int arrayArgIndex = 1;
        List<StackManipulation> impls = new ArrayList<>();
        //create the return array of the specified length
        impls.add(IntegerConstant.forValue(rank));
        impls.add(new CreateIntArrayStackManipulation());
        impls.add(new StoreRefStackManipulation(retArrayIndex));

        //index of the assignment
        for(int i = rank - 1; i >= 0; i--) {
            //index /= shape[i]
            //load the linear index for divide
            impls.add(MethodVariableAccess.INTEGER.loadOffset(linearIndexArg));
            //load the array
            impls.add(MethodVariableAccess.REFERENCE.loadOffset(arrayArgIndex));
            //load index of item to divide by
            impls.add(IntegerConstant.forValue(i));
            //load the item from the array based on the index
            impls.add(ArrayStackManipulation.load());
            //divide in place
            impls.add(OpStackManipulation.div());
            //store results
            impls.add(new StoreIntStackManipulation(linearIndexArg));


            // ret[i] = index / numIndices;
            impls.add(MethodVariableAccess.REFERENCE.loadOffset(retArrayIndex));
            impls.add(IntegerConstant.forValue(i));
            impls.add(MethodVariableAccess.INTEGER.loadOffset(linearIndexArg));
            impls.add(MethodVariableAccess.INTEGER.loadOffset(totalindexarg));
            impls.add(OpStackManipulation.div());
            impls.add(ArrayStackManipulation.store());

            //   index %= denom;
            impls.add(MethodVariableAccess.INTEGER.loadOffset(linearIndexArg));
            impls.add(MethodVariableAccess.INTEGER.loadOffset(totalindexarg));
            impls.add(OpStackManipulation.mod());
            impls.add(new StoreIntStackManipulation(linearIndexArg));

        }

        impls.add(MethodVariableAccess.REFERENCE.loadOffset(1));
        impls.add(MethodReturn.REFERENCE);
        return new StackManipulationImplementation(
                new StackManipulation.Compound(impls.toArray(new StackManipulation[impls.size()]))
        );
    }
    /**
     * Get an implementation of
     * ind2sub
     * @param ordering the order to iterate in
     * @param rank the rank of the array
     * @return the implementation (in byte code) of
     * ind2subseq
     *
     */
    public static Implementation getInd2Sub(char ordering, int rank) {


        /**
         * Given signature:
         *  int[] map(int[] shape,int index,int numIndices,char ordering);
         Load int param grabs numIndices because the instance
         variable stack indexing starts with this at zero

         4 here represents creating a variable and storing
         the value of the last argument in the method in the value

         */
        int retArrayIndex = 4;

        //the index of the variable we use to start denomination
        //the index of the last method parameter
        int linearIndexArg = 2;
        //the index of the total number of indexes
        int totalIndexArg = 3;

        //"this" is 0 and the shape array is the first argument
        int arrayArgIndex = 1;
        List<Implementation> impls = new ArrayList<>();
        //create the return array
        impls.add(new SimpleCreateArrayImplementation(rank));
        impls.add(new StoreImplementation(retArrayIndex));
        //index of the assignment
        for(int i = rank - 1; i >= 0; i--) {
            //index to assign
            impls.add( new RelativeLoadDeclaredReferenceImplementation(retArrayIndex));
            impls.add(new ConstantIntImplementation(i));
            //-------------------------------------------------
            //PUT SHAPE ASSIGNMENT LOGIC HERE
            //numIndices /= shape[i]
            //IN HERE SOMEWHERE
            //shape[i]
            impls.add(new LoadReferenceParamImplementation(arrayArgIndex));
            impls.add(new RelativeRetrieveArrayImplementation(i));
            //numIndices
            impls.add(new RelativeLoadIntParamImplementation(linearIndexArg));
            // /=
            impls.add(new RelativeOperationImplementation(RelativeOperationImplementation.Operation.DIV));
            //store in index
            impls.add(new StoreIntImplementation(linearIndexArg));
            //IN HERE SOMEWHERE
            impls.add(new RelativeLoadIntParamImplementation(totalIndexArg));
            impls.add(new RelativeOperationImplementation(RelativeOperationImplementation.Operation.DIV));

            impls.add( new RelativeLoadDeclaredReferenceImplementation(retArrayIndex));

            //------------------------------------------------
            impls.add(new AssignOpImplementation());
        }


        impls.add( new RelativeLoadDeclaredReferenceImplementation(4));
        impls.add(new ReturnAppenderImplementation(ReturnAppender.ReturnType.REFERENCE));

        return new Implementation.Compound(
                impls.toArray(new Implementation[impls.size()])
        );
    }

}
