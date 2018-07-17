//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 16.07.2018
//

#include <GradCheck.h>


namespace nd4j {

//////////////////////////////////////////////////////////////////////////
bool GradCheck::checkGrad(ops::DeclarableOp<double>& opFF, ops::DeclarableOp<double>& opBP, const OpArgsHolder<double>& argsHolderFF, const OpArgsHolder<double>& argsHolderBP, const LossFunc loss) {

	const int numInArrsFF     = argsHolderFF.getNumInArrs();
	const int numInGradArrsBP = argsHolderBP.getNumInArrs() - numInArrsFF;  		// because argsHolderBP.getNumInArrs() = numInArrsFF + numInGradArrsBP
	const std::vector<NDArray<double>*>& inArrsFF = argsHolderFF.getInArrs();
	const std::vector<NDArray<double>*>& inArrsBP = argsHolderBP.getInArrs();

	// fill input gradient arrays in accordance to type of loss function
	fillGradArrays(loss, std::vector<NDArray<double>*>(&inArrsBP[numInArrsFF], &inArrsBP[numInArrsFF + numInGradArrsBP]));

	// beck prop pass	
	ResultSet<double>* bpOutArrs = opBP.execute(argsHolderBP);		// number of output arrays in back prop = numInArrsFF;

	for(int i = 0; i < numInArrsFF; ++i) {							// loop through input array
		for(int j = 0; j < inArrsFF[i]->lengthOf(); ++j) {			// loop through all elements for current array

			double& elem = (*inArrsFF[i])(j);
			const double orig = elem;

			// add epsilon, feed forward
			elem = orig + EPSILON;
			ResultSet<double>* ffOutArrs = opFF.execute(argsHolderFF);
			int numOutArrs = ffOutArrs->size();
			double scorePlus = 0.;
			for(int k = 0; k < numOutArrs; ++k)				// loop through output array
				scorePlus += NativeOpExcutioner<double>::execReduceScalar(loss, ffOutArrs->at(k)->getBuffer(), ffOutArrs->at(k)->getShapeInfo(), nullptr);
			delete ffOutArrs;

			// minus epsilon, feed forward
			elem = orig - EPSILON;
			ffOutArrs = opFF.execute(argsHolderFF);
			double scoreMinus = 0.;
			for(int k = 0; k < numOutArrs; ++k)				// loop through output array
				scoreMinus += NativeOpExcutioner<double>::execReduceScalar(loss, ffOutArrs->at(k)->getBuffer(), ffOutArrs->at(k)->getShapeInfo(), nullptr);
			delete ffOutArrs;

			// restore initial element value
			elem = orig;

			// calculate numerical gradient
			const double numericalGrad = (scorePlus - scoreMinus) / (2 * EPSILON);
			if(std::isnan(numericalGrad) || std::isinf(numericalGrad)) {
				printf("GradCheck::checkGrad: got wrong value for numerical gradient for input array # %i and its element at position # %i \n !", i, j);
				throw std::runtime_error("");
			}

			// get analytical gradient 
			const double analyticGrad = (*bpOutArrs->at(i))(j);
			if(std::isnan(analyticGrad) || std::isinf(analyticGrad)) {
				printf("GradCheck::checkGrad: got wrong value for analytical gradient for input array # %i and its element at position # %i \n !", i, j);
				throw std::runtime_error("");
			}

			// verify result
			double relError;
			if(numericalGrad == 0. && analyticGrad == 0.)
				relError = 0.;
			else
                relError = math::nd4j_abs<double>(analyticGrad - numericalGrad) / (math::nd4j_abs<double>(analyticGrad) + math::nd4j_abs<double>(numericalGrad));            

            if(relError > MAXRELERR || std::isnan(relError)) {

            	if(math::nd4j_abs<double>(analyticGrad - numericalGrad) < MINABSERR);
            		continue;

            	printf("GradCheck::checkGrad: got RELERROR = %f (> MAXRELERROR) for input array # %i and its element at position # %i \n !", relError , i, j);
            	delete bpOutArrs;
            	return false;
            }
		}
	}
	
	delete bpOutArrs;
	return true;
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


