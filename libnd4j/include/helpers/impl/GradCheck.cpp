/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 16.07.2018
//

#include <GradCheck.h>
#include <NDArrayFactory.h>


namespace nd4j {

//////////////////////////////////////////////////////////////////////////
void GradCheck::fillGradArrays(const LossFunc loss, const std::vector<NDArray*>& gradArrs) {

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

//////////////////////////////////////////////////////////////////////////
bool GradCheck::checkGrad(ops::DeclarableOp& opFF, ops::DeclarableOp& opBP, const OpArgsHolder& argsHolderFF, const OpArgsHolder& argsHolderBP,
	                      const std::vector<bool>& whatArrsToCheck, const std::vector<double>& idxRange, const LossFunc loss ) {

	const int numInArrsFF     = argsHolderFF.getNumInArrs();						// also numInArrsFF = number of output arrays in opBP
	const int numInGradArrsBP = argsHolderBP.getNumInArrs() - numInArrsFF;  		// because argsHolderBP.getNumInArrs() = numInArrsFF + numInGradArrsBP
	const std::vector<NDArray*>& inArrsFF = argsHolderFF.getInArrs();
	const std::vector<NDArray*>& inArrsBP = argsHolderBP.getInArrs();

	// fill input gradient arrays in accordance to type of loss function
	fillGradArrays(loss, std::vector<NDArray*>(&inArrsBP[numInArrsFF], &inArrsBP[numInArrsFF + numInGradArrsBP]));

	// beck prop pass	
	ResultSet* outArrsBP = opBP.execute(argsHolderBP);		// number of output arrays in back prop = numInArrsFF;

	for(int i = 0; i < numInArrsFF; ++i) {							// loop through input array
		
		if(!whatArrsToCheck.empty() && whatArrsToCheck[i] == false)
			continue;

		const Nd4jLong idxStart = static_cast<Nd4jLong>(idxRange[0] * inArrsFF[i]->lengthOf());
		const Nd4jLong idxEnd   = static_cast<Nd4jLong>(idxRange[1] * inArrsFF[i]->lengthOf());
		
		for(Nd4jLong j = idxStart; j < idxEnd; ++j) {			// loop through all elements for current array

			auto f = inArrsFF[i]->e<double>(j);
			double& elem = f;
			const double orig = elem;

			// add epsilon, feed forward
			elem = orig + EPSILON;
			ResultSet* outArrsFF = opFF.execute(argsHolderFF);
			int numOutArrs = outArrsFF->size();
			double scorePlus = 0.;
			auto tmpScalar = NDArrayFactory::create(0.0);
			for(int k = 0; k < numOutArrs; ++k) {                // loop through output array
				NativeOpExcutioner::execReduceScalar(loss, outArrsFF->at(k)->getBuffer(), outArrsFF->at(k)->getShapeInfo(), nullptr, tmpScalar.buffer(), tmpScalar.shapeInfo());
				scorePlus += tmpScalar.e<double>(0);
			}
			delete outArrsFF;

			// subtract epsilon, feed forward
			elem = orig - EPSILON;
			outArrsFF = opFF.execute(argsHolderFF);
			double scoreMinus = 0.;

			for(int k = 0; k < numOutArrs; ++k) {            // loop through output array
				NativeOpExcutioner::execReduceScalar(loss, outArrsFF->at(k)->getBuffer(), outArrsFF->at(k)->getShapeInfo(), nullptr, tmpScalar.buffer(), tmpScalar.shapeInfo());
				scoreMinus += tmpScalar.e<double>(0);
			}
			delete outArrsFF;

			// restore initial element value
			elem = orig;

			// calculate numerical gradient
			const double numericalGrad = (scorePlus - scoreMinus) / (2 * EPSILON);
			if(std::isnan(numericalGrad) || std::isinf(numericalGrad)) {
				printf("GradCheck::checkGrad: got wrong value for numerical gradient for input array # %i and its element at position %lld ! \n", i, j);
				throw std::runtime_error("");
			}

			// get analytical gradient 
			const double analyticGrad = outArrsBP->at(i)->e<double>(j);
			if(std::isnan(analyticGrad) || std::isinf(analyticGrad)) {
				printf("GradCheck::checkGrad: got wrong value for analytical gradient for input array # %i and its element at position %lld ! \n", i, j);
				throw std::runtime_error("");
			}

			// printf("num = %.5f, ana = %.5f\n", numericalGrad, analyticGrad);
			
			// calculate relative error
			double relError;
			if(numericalGrad == 0. && analyticGrad == 0.)
				relError = 0.;
			else
                relError = math::nd4j_abs<double>(analyticGrad - numericalGrad) / (math::nd4j_abs<double>(analyticGrad) + math::nd4j_abs<double>(numericalGrad));            

            // verify result
            if(relError > MAXRELERR || std::isnan(relError)) {

            	if(math::nd4j_abs<double>(analyticGrad - numericalGrad) < MINABSERR)
            		continue;
            	printf("numericalGrad = %f,  analyticGrad = %f \n", numericalGrad, analyticGrad);
            	printf("GradCheck::checkGrad: got RELERROR = %f > MAXRELERROR(%f) for input array # %i and its element at position %lld ! \n", relError, MAXRELERR, i, j);
            	delete outArrsBP;
            	return false;
            }
		}
	}
	
	delete outArrsBP;
	return true;
}



}


