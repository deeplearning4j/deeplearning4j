package org.nd4j.bytebuddy.shape;

import net.bytebuddy.implementation.Implementation;
import org.nd4j.bytebuddy.arithmetic.relative.op.RelativeOperationImplementation;
import org.nd4j.bytebuddy.arrays.assign.relative.novalue.RelativeAssignNoValueImplementation;
import org.nd4j.bytebuddy.arrays.assign.relative.op.AssignOpImplementation;
import org.nd4j.bytebuddy.arrays.assign.relative.withvalue.RelativeArrayAssignWithValueImplementation;
import org.nd4j.bytebuddy.arrays.create.noreturn.IntArrayCreation;
import org.nd4j.bytebuddy.arrays.create.relative.RelativeIntArrayCreation;
import org.nd4j.bytebuddy.arrays.create.simple.SimpleCreateArrayImplementation;
import org.nd4j.bytebuddy.arrays.retrieve.relative.RelativeRetrieveArrayImplementation;
import org.nd4j.bytebuddy.constant.ConstantIntImplementation;
import org.nd4j.bytebuddy.createint.StoreIntImplementation;
import org.nd4j.bytebuddy.dup.DuplicateImplementation;
import org.nd4j.bytebuddy.load.LoadIntegerImplementation;
import org.nd4j.bytebuddy.loadref.LoadReferenceImplementation;
import org.nd4j.bytebuddy.loadref.relative.RelativeLoadDeclaredReferenceImplementation;
import org.nd4j.bytebuddy.method.integer.LoadIntParamImplementation;
import org.nd4j.bytebuddy.method.integer.relative.RelativeLoadIntParamImplementation;
import org.nd4j.bytebuddy.method.reference.LoadReferenceParamImplementation;
import org.nd4j.bytebuddy.returnref.ReturnAppender;
import org.nd4j.bytebuddy.returnref.ReturnAppenderImplementation;
import org.nd4j.bytebuddy.storeref.StoreImplementation;
import org.nd4j.bytebuddy.storeref.StoreRef;

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
