//
// Created by yurii@skymind.io on 01.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>
#include <array>

namespace nd4j {
    namespace ops {
		CUSTOM_OP_IMPL(stack, -1, 1, false, 0, 1) {
    		NDArray<T>* input = INPUT_VARIABLE(0);
    		NDArray<T>* output = OUTPUT_VARIABLE(0);

    		int dim = INT_ARG(0);
    		if(dim < 0)
    			dim += input->rankOf();             

			// input validation
			// check whether shapes of all input array are the same		
			bool allScalars = true;
			auto first = INPUT_VARIABLE(0);
			for (int i = 0; i < (int) block.width(); ++i) {
				auto array = INPUT_VARIABLE(i);
				REQUIRE_TRUE(shape::equalsSoft(array->getShapeInfo(), first->getShapeInfo()), 0, "CUSTOM_OP stack: the shapes of input arrays are different !");
				allScalars &= array->isScalar();
			}

			// scalar is special case, that produces row vector
			if (allScalars) {
				for (int e = 0; e < block.width(); e++) {
					auto arr = INPUT_VARIABLE(e);
					output->putScalar(e, arr->getScalar(0));
				}
			
				return ND4J_STATUS_OK;
			}

   			REQUIRE_TRUE(dim < input->rankOf(), 0, "CUSTOM_OP stack: the input dimension is greater/equal than rank of input input arrays shapes !");

			std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(output->rankOf(), {dim});	
			ResultSet<T>* list = NDArrayFactory<T>::allTensorsAlongDimension(output, dimsToExclude);		// list.size() == block.width()

			for(int i=0; i<list->size(); ++i)
				list->at(i)->assign(INPUT_VARIABLE(i));
	
			// remove unity from output shape if input arrays are vectors 
			if(input->isVector())	{
				std::vector<int> outShape(output->shapeOf(), output->shapeOf() + output->rankOf());		
				outShape.erase(find(outShape.begin(), outShape.end(), 1));
				output->reshapei(output->ordering(), outShape);
				if(dim != 0 && (int)block.width() == 1)			// such is implemented by tensorFlow
					output->permutei({1, 0});
				output->getShapeInfo()[output->rankOf()*2 + 2] = 1;		
			}
	

    		STORE_RESULT(*output);
			delete list;
    		return ND4J_STATUS_OK;
		}
		DECLARE_SYN(pack, stack);
		DECLARE_SYN(Pack, stack);

		DECLARE_SHAPE_FN(stack) {
	
			// check whether input dimension is within rank range
			int* inShapeInfo = inputShape->at(0);
			int rank = inShapeInfo[0];
			int dim = INT_ARG(0);

			int elements = inputShape->size();
			if(dim < 0 ) dim += rank;

			{ // special cases for 0D concat
                bool allScalars = true;
                bool realScalars = false;
				int *newShape;
                for (int e = 0; e < elements; e++) {
                    allScalars &= shape::isScalar(inputShape->at(e));
                    realScalars |= shape::rank(inputShape->at(e)) == 0;
                }


				// any scalar
                if (allScalars && realScalars) {
                    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), int);
                    int length = shape::length(inputShape->at(0));
                    for (int i = 1; i < elements; i++) {
                       length += 1;
                    }

					std::array<int, 1> shape = {{length}};
                    shape::shapeBuffer(1, shape.data(), newShape);
                    return new ShapeList(newShape);
                } else if (allScalars) {
					// all scalars
                    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);

                    std::array<int, 2> shape = {{1, elements}};
                    shape::shapeBuffer(2, shape.data(), newShape);
                    return new ShapeList(newShape);
                }
            }

			//the rank of output ShapeInfo is larger by one compared to input ShapeInfo
			std::vector<int> outShape(inShapeInfo + 1, inShapeInfo + 1 + rank);
			// insert (int) block.width() at dim position of input shape to get output shape	
			outShape.insert(outShape.begin() + dim, (int) block.width());
			// if input arrays are vectors remove unity from shape			

			// evaluate output ShapeInfo
			int newRank = outShape.size();
			int* outShapeInfo = nullptr;
    		ALLOCATE(outShapeInfo, block.getWorkspace(), newRank*2+4, int);
    		outShapeInfo[0] = newRank;
    		for(int i=1; i <= newRank; ++i)
    			outShapeInfo[i] = outShape[i-1];
	
    		shape::updateStrides(outShapeInfo, shape::order(inShapeInfo));    

    		return new ShapeList(outShapeInfo);
		}
// 1) 1х4 + 1х4 = 2х1х4 (along dim=0) = 2x4 
// 2) 1х4 + 1х4 = 1х2х4 (along dim=1) = 2x4 
// 3) 4х1 + 4х1 = 2х4x1 (along dim=0) = 2x4 
// 4) 4х1 + 4х1 = 4х2x1 (along dim=1) = 4x2 
	}
}