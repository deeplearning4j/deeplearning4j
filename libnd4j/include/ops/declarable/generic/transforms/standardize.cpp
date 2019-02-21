/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Paul Dubs
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_standardize)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>


namespace nd4j {
namespace ops  {

    CONFIGURABLE_OP_IMPL(standardize, 1, 1, true, 0, -2) {
        
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);
        
        std::vector<int> axis;        

        if (block.width() > 1)            
            axis = INPUT_VARIABLE(1)->template asVectorT<int>();
        else if (block.numI() > 0) 
            axis = *block.getIArguments();        

        if(axis.empty()) {      // do not perform standardization, as element-wise each element is already at zero mean unit variance
            output->assign(input);
        }
        else {
            shape::checkDimensions(input->rankOf(), axis);

            auto means = input->reduceAlongDims(reduce::Mean, axis, true);
            auto stddev = input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, false, axis);
            stddev->reshapei(means.getShapeAsVector());

            output->assign((*input - means) / *stddev);
        }
   
        return Status::OK();
    }


    DECLARE_TYPES(standardize) {
        getOpDescriptor()->setAllowedInputTypes(0, DataType::ANY);
        getOpDescriptor()->setAllowedInputTypes(1, {DataType::INT32, DataType::INT64});
        getOpDescriptor()->setAllowedOutputTypes(0, DataType::INHERIT);
    }

    CUSTOM_OP_IMPL(standardize_bp, 2, 1, false, 0, -2) {
        auto input = INPUT_VARIABLE(0);
        auto eps = block.width() == 3 ? INPUT_VARIABLE(2) : INPUT_VARIABLE(1);

        auto output = OUTPUT_VARIABLE(0);
        std::vector<int> axis;

        if (block.width() == 3)             
            axis = INPUT_VARIABLE(1)->template asVectorT<int>();
        else if (block.numI() > 0) 
            axis = *block.getIArguments();        

        if(axis.empty()) {      // nothing to do in this case
            output->assign(eps);
        }
        else {
            shape::checkDimensions(input->rankOf(), axis);
            auto longAxis = ArrayUtils::toLongVector(axis);

            auto means = input->reduceAlongDims(reduce::Mean, axis, true);
            auto stdev = *input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, false, axis);
            stdev.reshapei(means.getShapeAsVector());

            auto dldx = *eps / stdev;
            output->assign(dldx);

            auto dldu_sum =(-dldx).reduceAlongDims(reduce::Sum, axis, true);
            nd4j::ops::reduce_mean_bp meanBp;
            auto dldx_u = *meanBp.execute({input, &dldu_sum}, {}, longAxis)->at(0);
            *output += dldx_u;

            auto dlds_sum = (*eps * (means - *input) / (stdev * stdev)).reduceAlongDims(reduce::Sum, axis, true);
            nd4j::ops::reduce_stdev_bp stdevBp;
            auto dldx_s = *stdevBp.execute({input, &dlds_sum}, {}, longAxis)->at(0);
            *output += dldx_s;
        }

        return Status::OK();
    }

    DECLARE_TYPES(standardize_bp) {
        getOpDescriptor()
                ->setAllowedInputTypes(nd4j::DataType::ANY)
                ->setAllowedOutputTypes({ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(standardize_bp) {
        auto in = inputShape->at(0);
        Nd4jLong *out;
        COPY_SHAPE(in, out);

        return SHAPELIST(out);
    }

}
}

#endif