//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 16.07.2018
//

#include <GradCheck.h>


namespace nd4j {

//////////////////////////////////////////////////////////////////////////
bool GradCheck::checkGrad(const ops::DeclarableOp<double>& opFF, const ops::DeclarableOp<double>& opBP, const OpArgsHolder<double>& argsHolderFF, const OpArgsHolder<double>& argsHolderBP, const LossFunc loss) {

	const int numInArrs     = argsHolderFF.getNumInArrs();
	const int numInGradArrs = argsHolderBP.getNumInArrs() - numInArrs;  		// because argsHolderBP.getNumInArrs() = numInArrs + numInGradArrs
	const std::vector<NDArray<double>*>& bpInArrs = argsHolderBP.getInArrs();

	// fill input gradient arrays in accordance to type of loss function
	fillGradArrays(loss, std::vector<NDArray<double>*>(&bpInArrs[numInArrs], &bpInArrs[numInArrs + numInGradArrs]));
	
	// also set corresponding opNum - number of reduce operation, that is loss function
	// NativeOpExcutioner<T>::execReduceScalar(1, _buffer, _shapeInfo, nullptr);

		return false;
	}


//////////////////////////////////////////////////////////////////////////
void GradCheck::fillGradArrays(const LossFunc loss, const std::vector<NDArray<double>*>& gradArrs) {

	const int numInGradArrs = gradArrs.size();

	// fill input gradient arrays in accordance to type of loss function	
	switch(loss) {

		case MEAN: 
#pragma omp parallel for if(numInGradArrs > 1) schedule(guided)
			for(int i = 0; i < numInGradArrs; ++i) 				
				*gradArrs[i] = 1. / gradArrs[i]->lengthOf();			
			break;

		case SUM: 
#pragma omp parallel for if(numInGradArrs > 1) schedule(guided)		
			for(int i = 0; i < numInGradArrs; ++i) 
				*gradArrs[i] = 1.;
			break;
			 
		default:				
			throw std::invalid_argument("GradCheck::fillGradArrays: invalid type of loss function !");        		
	}
}


}


