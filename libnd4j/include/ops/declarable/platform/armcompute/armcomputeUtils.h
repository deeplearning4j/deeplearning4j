/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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


#ifndef DEV_TESTSARMCOMPUTEUTILS_H
#define DEV_TESTSARMCOMPUTEUTILS_H


#include <legacy/NativeOps.h>
#include <array/NDArray.h> 
#include <graph/Context.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Strides.h>
#include <arm_compute/core/Helpers.h>
#include <arm_compute/core/ITensor.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/core/Validate.h>
#include <arm_compute/core/Window.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/TensorAllocator.h> 
#include <iostream>

using namespace samediff;


namespace sd {
    namespace ops {
        namespace platforms {

            using Arm_DataType = arm_compute::DataType;
            using Arm_Tensor = arm_compute::Tensor;
            using Arm_ITensor = arm_compute::ITensor;            
            using Arm_TensorInfo = arm_compute::TensorInfo;
            using Arm_TensorShape = arm_compute::TensorShape;
            using Arm_Strides = arm_compute::Strides;
            using Arm_WeightsInfo = arm_compute::WeightsInfo;
            using Arm_PermutationVector = arm_compute::PermutationVector;
            using Arm_DataLayout =  arm_compute::DataLayout;
            /**
             * Here we actually declare our platform helpers
             */
             

            DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);
 
            DECLARE_PLATFORM(avgpool2d, ENGINE_CPU);

            DECLARE_PLATFORM(conv2d, ENGINE_CPU);

            DECLARE_PLATFORM(deconv2d, ENGINE_CPU);            

            //utils
            Arm_DataType getArmType(const sd::DataType& dType);

            Arm_TensorInfo getArmTensorInfo(int rank, Nd4jLong* bases, sd::DataType ndArrayType, Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

            Arm_TensorInfo getArmTensorInfo(const NDArray& arr, Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

            Arm_Tensor getArmTensor(const NDArray& arr, Arm_DataLayout layout = Arm_DataLayout::UNKNOWN);

            void copyFromTensor(const Arm_Tensor& inTensor, NDArray& output);
            void copyToTensor(const NDArray& input, Arm_Tensor& outTensor);
            void print_tensor(Arm_ITensor& tensor, const char* msg);
            bool isArmcomputeFriendly(const NDArray& arr);


            template<typename F>
            class ArmFunction {
            public:

               template<typename ...Args>
               void configure(NDArray *input , NDArray *output, Arm_DataLayout layout, Args&& ...args) {
                   
                   auto inInfo = getArmTensorInfo(*input, layout);
                   auto outInfo = getArmTensorInfo(*output, layout);  
                   in.allocator()->init(inInfo);
                   out.allocator()->init(outInfo);
                   armFunction.configure(&in,&out,std::forward<Args>(args) ...);
                   if (in.info()->has_padding()) {
                       //allocate and copy
                       in.allocator()->allocate();
                       //copy 
                       copyToTensor(*input, in);

                   }
                   else {
                       //import buffer
                       void* buff = input->buffer();
                       in.allocator()->import_memory(buff);
                   } 
                   if (out.info()->has_padding()) {
                       //store pointer to our array to copy after run
                       out.allocator()->allocate();
                       outNd = output;
                   }
                   else {
                       //import
                       void* buff = output->buffer();
                       out.allocator()->import_memory(buff);
                   }

               }

               void run() {
                   armFunction.run();
                   if (outNd) {
                       copyFromTensor(out, *outNd);
                   }
               }

               private:
                   Arm_Tensor in;
                   Arm_Tensor out;
                   NDArray *outNd=nullptr;
                   F armFunction{};
            };


     template<typename F>
     class ArmFunctionWeighted {
     public:
          
         template<typename ...Args>
         void configure(const NDArray* input, const NDArray* weights, const NDArray* biases, NDArray* output, Arm_DataLayout layout, arm_compute::PermutationVector permuteVector, Args&& ...args) {

             auto inInfo = getArmTensorInfo(*input, layout);
             auto wInfo = getArmTensorInfo(*weights, layout);
             Arm_Tensor* bias_ptr = nullptr;
             auto outInfo = getArmTensorInfo(*output, layout);
             in.allocator()->init(inInfo);
             w.allocator()->init(wInfo);

             if (biases) {
                 auto bInfo = getArmTensorInfo(*biases, layout);
                 b.allocator()->init(bInfo);
                 bias_ptr = &b;
             }
             out.allocator()->init(outInfo);

             if (permuteVector.num_dimensions()==0){
                 armFunction.configure(&in, &w, bias_ptr, &out, std::forward<Args>(args)...); 
             }
             else {
                 //configure with permute kernel
                 Arm_TensorShape shape;
                 int rank = permuteVector.num_dimensions();
                 shape.set_num_dimensions(rank);
                 for (int i = 0; i < rank; i++) {
                     shape[i] = wInfo.dimension(permuteVector[i]);
                 }
                 for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
                     shape[i] = 1;
                 }

                 Arm_TensorInfo wPermInfo(shape, 1, wInfo.data_type(), layout);
                 wPerm.allocator()->init(wPermInfo); 
                 permuter.configure(&w, &wPerm, permuteVector);
                 armFunction.configure(&in, &wPerm, bias_ptr, &out, std::forward<Args>(args)...); 
                 wPerm.allocator()->allocate();
                 runPerm = true;
             }


             if (in.info()->has_padding()) {
                 //allocate and copy
                 in.allocator()->allocate();
                 //copy 
                 copyToTensor(*input, in);
#if 0
    nd4j_printf("input copied %d\n",0);
#endif      
             }
             else {
                 //import buffer
                 auto buff = input->buffer();
                 in.allocator()->import_memory((void*)buff);
#if 0
    nd4j_printf("input imported %d\n",0);
#endif                       
             }

             if (w.info()->has_padding()) {
                 //allocate and copy
                 w.allocator()->allocate();
                 //copy 
                 copyToTensor(*weights, w);
#if 0
    nd4j_printf("weight copied %d\n",0);
#endif      
             }
             else {
                 //import buffer
                 auto buff = weights->buffer();
                 w.allocator()->import_memory((void*)buff);
#if 0
    nd4j_printf("weight imported %d\n",0);
#endif                  
             }

             if (bias_ptr) {
                 //check bias also
                 if (b.info()->has_padding()) {
                     //allocate and copy
                     b.allocator()->allocate();
                     //copy 
                     copyToTensor(*biases, b);
#if 0
    nd4j_printf("b copied %d\n",0);
#endif    
                 }
                 else {
                     //import buffer
                     auto buff = biases->buffer();
                     b.allocator()->import_memory((void*)buff);
#if 0
    nd4j_printf("b imported %d\n",0);
#endif                      
                 }
             }
             if (out.info()->has_padding()) {
                 //store pointer to our array to copy after run
                 out.allocator()->allocate();
                 outNd = output;
#if 0
    nd4j_printf("out copied %d\n",0);
#endif                  
             }
             else {
                 //import
                 void* buff = output->buffer();
                 out.allocator()->import_memory(buff);
#if 0
    nd4j_printf("out imported %d\n",0);
#endif                  
             }

         }

         void run() {
             if(runPerm){
                 permuter.run();
             }
             armFunction.run();
             if (outNd) {
                 copyFromTensor(out, *outNd);
             }
         }

     private:
         bool runPerm=false;
         Arm_Tensor in;
         Arm_Tensor b;
         Arm_Tensor w;
         Arm_Tensor wPerm;
         Arm_Tensor out;
         NDArray* outNd = nullptr;
         arm_compute::NEPermute permuter;
         F armFunction{};
     };   

        }
    }
}



#endif //DEV_TESTSARMCOMPUTEUTILS_H
