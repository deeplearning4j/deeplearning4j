package org.nd4j.bytebuddy.shape;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;
import net.bytebuddy.implementation.bytecode.member.MethodReturn;
import net.bytebuddy.implementation.bytecode.member.MethodVariableAccess;
import net.bytebuddy.jar.asm.Label;
import net.bytebuddy.matcher.ElementMatchers;
import org.nd4j.bytebuddy.arithmetic.ByteBuddyIntArithmetic;
import org.nd4j.bytebuddy.arithmetic.stackmanipulation.OpStackManipulation;
import org.nd4j.bytebuddy.arrays.create.stackmanipulation.CreateIntArrayStackManipulation;
import org.nd4j.bytebuddy.arrays.stackmanipulation.ArrayStackManipulation;
import org.nd4j.bytebuddy.branching.stackmanipulation.IfeqNotEquals;
import org.nd4j.bytebuddy.frame.VisitFrameFullInt;
import org.nd4j.bytebuddy.frame.VisitFrameSameInt;
import org.nd4j.bytebuddy.gotoop.GoToOp;
import org.nd4j.bytebuddy.labelvisit.LabelVisitorStackManipulation;
import org.nd4j.bytebuddy.stackmanipulation.StackManipulationImplementation;
import org.nd4j.bytebuddy.storeint.stackmanipulation.StoreIntStackManipulation;
import org.nd4j.bytebuddy.storeref.stackmanipulation.StoreRefStackManipulation;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class ShapeMapper {





    /**
     * Get an ind2sub instance
     * based on the ordering and rank
     * @param ordering the ordering
     * @param rank the rank
     * @return the ind2sub instance
     */
    public static IndexMapper getInd2SubInstance(char ordering,int rank) {
        Implementation impl = ShapeMapper.getInd2Sub(ordering, rank);
        DynamicType.Unloaded<IndexMapper> c = new ByteBuddy()
                .subclass(IndexMapper.class).method(ElementMatchers.isDeclaredBy(IndexMapper.class))
                .intercept(impl)
                .make();

        Class<IndexMapper> dynamicType = (Class<IndexMapper>)
                c.load(IndexMapper.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                        .getLoaded();
        try {
            return dynamicType.newInstance();
        } catch (Exception e) {
            throw new IllegalStateException("Unable to get index mapper for rank " + rank);
        }

    }


    /**
     * Get an ind2sub instance
     * based on the ordering and rank
     * @param rank the rank
     * @return the ind2sub instance
     */
    public static OffsetMapper getOffsetMapperInstance(int rank) {
        Implementation impl = ShapeMapper.getOffsetMapper(rank);
        DynamicType.Unloaded<OffsetMapper> c = new ByteBuddy()
                .subclass(OffsetMapper.class).method(ElementMatchers.isDeclaredBy(OffsetMapper.class))
                .intercept(impl)
                .make();

        Class<OffsetMapper> dynamicType = (Class<OffsetMapper>)
                c.load(OffsetMapper.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                        .getLoaded();
        try {
            return dynamicType.newInstance();
        } catch (Exception e) {
            throw new IllegalStateException("Unable to get index mapper for rank " + rank);
        }

    }

    /**
     * Get the offset mapper bytecode
     * for a particular rank
     * @param rank the rank of array
     *             to generate the offset mapper byte code for
     * @return the implementation of the offset mapper bytecode
     *
     */
    public static Implementation getOffsetMapper(int rank) {
        /**
         * Given:
         * int getOffset(int baseOffset,int[] shape,int[] stride,int[] indices);
         */
        //start offset is the base index to start at
        int startOffsetIndex = 1;
        //shape index is the index of the argument for shape
        int shapeIndex = 2;
        //stride index is the index of the argument for shape
        int strideIndex = 3;
        //indicesindex is the index for the indices
        int indicesIndex = 4;
        List<StackManipulation> impls = new ArrayList<>();

        for(int i = 0; i < rank; i++) {
            Label label = new Label();
            Label goToLabel = new Label();
            impls.add(MethodVariableAccess.INTEGER.loadOffset(startOffsetIndex));
            //load the array
            impls.add(MethodVariableAccess.REFERENCE.loadOffset(shapeIndex));
            //from the array load the current index
            impls.add(IntegerConstant.forValue(i));
            impls.add(ArrayStackManipulation.load());
            impls.add(new IfeqNotEquals(label));
            //load the stride  for the current index
            impls.add(MethodVariableAccess.REFERENCE.loadOffset(strideIndex));
            impls.add(IntegerConstant.forValue(i));
            impls.add(ArrayStackManipulation.load());
            //load the indices array at the current index
            impls.add(MethodVariableAccess.REFERENCE.loadOffset(indicesIndex));
            impls.add(IntegerConstant.forValue(i));
            impls.add(ArrayStackManipulation.load());

            impls.add(ByteBuddyIntArithmetic.IntegerMultiplication.INSTANCE);
            impls.add(new GoToOp(goToLabel));
            impls.add(new LabelVisitorStackManipulation(label));
            impls.add(new VisitFrameSameInt(0,1));
            impls.add(IntegerConstant.forValue(i));
            impls.add(new LabelVisitorStackManipulation(goToLabel));
            impls.add(new VisitFrameFullInt(5,2));
            //add to the offset +=
            impls.add(ByteBuddyIntArithmetic.IntegerAddition.INSTANCE);
            impls.add(new StoreIntStackManipulation(startOffsetIndex));

        }

        impls.add(MethodVariableAccess.INTEGER.loadOffset(startOffsetIndex));
        impls.add(MethodReturn.INTEGER);
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
        int totalindexarg = 3;

        //"this" is 0 and the shape array is the first argument
        int arrayArgIndex = 1;
        List<StackManipulation> impls = new ArrayList<>();
        //create the return array of the specified length
        impls.add(IntegerConstant.forValue(rank));
        impls.add(new CreateIntArrayStackManipulation());
        impls.add(new StoreRefStackManipulation(retArrayIndex));
        if(ordering == 'f') {
            //linearIndex of the assignment
            for(int i = rank - 1; i >= 0; i--) {
                //index /= shape[i]
                //load the linear index for divide
                impls.add(MethodVariableAccess.INTEGER.loadOffset(totalindexarg));
                //load the array
                impls.add(MethodVariableAccess.REFERENCE.loadOffset(arrayArgIndex));
                //load index of item to divide by
                impls.add(IntegerConstant.forValue(i));
                //load the item from the array based on the index
                impls.add(ArrayStackManipulation.load());
                //divide in place
                impls.add(OpStackManipulation.div());
                //store results
                impls.add(new StoreIntStackManipulation(totalindexarg));


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



        }
        else {
            //index of the assignment
            for(int i = 0; i < rank; i++) {
                //index /= shape[i]
                //load the linear index for divide
                impls.add(MethodVariableAccess.INTEGER.loadOffset(totalindexarg));
                //load the array
                impls.add(MethodVariableAccess.REFERENCE.loadOffset(arrayArgIndex));
                //load index of item to divide by
                impls.add(IntegerConstant.forValue(i));
                //load the item from the array based on the index
                impls.add(ArrayStackManipulation.load());
                //divide in place
                impls.add(OpStackManipulation.div());
                //store results
                impls.add(new StoreIntStackManipulation(totalindexarg));


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
        }


        impls.add(MethodVariableAccess.REFERENCE.loadOffset(retArrayIndex));
        impls.add(MethodReturn.REFERENCE);
        return new StackManipulationImplementation(
                new StackManipulation.Compound(impls.toArray(new StackManipulation[impls.size()]))
        );
    }


}
