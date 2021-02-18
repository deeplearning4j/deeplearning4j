/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

 //
 // @author Yurii Shyrma, created on 26.02.2018
 //
 //
 // @author AbdelRauf 
 //

#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <execution/Threads.h>
#include <execution/ThreadPool.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/addBias.h>

#if defined(__GNUC__) 
#define align32 __attribute__((aligned(32)))
#elif defined(_MSC_VER)
#define align32 __declspec(align(32))
#else
#define align32 
#endif 

namespace sd {
	namespace ops {
		namespace helpers {

			template <typename T>
			static FORCEINLINE  void _add(const T* __restrict xx, const T* __restrict yy, T* __restrict zz, const size_t& N) {
				PRAGMA_OMP_SIMD
					for (size_t c = 0; c < N; c++)
						zz[c] = xx[c] + yy[c];
			}

			template <typename T>
			static FORCEINLINE  void _add_inplace(T* __restrict xx, const T* __restrict yy, const size_t& N) {
				PRAGMA_OMP_SIMD
					for (size_t c = 0; c < N; c++)
						xx[c] = xx[c] + yy[c];
			}

			template <typename T>
			static FORCEINLINE  void _add_broadcast_inplace(T* __restrict xx, const T  yy, const size_t& N) {
				PRAGMA_OMP_SIMD
					for (size_t c = 0; c < N; c++)
						xx[c] = xx[c] + yy;
			}

			template <typename T>
			static FORCEINLINE  void _add_broadcast(const T* __restrict xx, const T  yy, T* __restrict zz, const size_t& N) {
				PRAGMA_OMP_SIMD
					for (size_t c = 0; c < N; c++)
						zz[c] = xx[c] + yy;
			}

			static constexpr size_t MIN_NN = 32;
			static constexpr size_t MIN_NN_K = 2;

			template<typename X, typename Y>
			static typename std::enable_if<std::is_same<X, Y>::value, const X*>::type
				flattened_bias(const Y* b_real, X* b_stack, const size_t b_stack_size, std::unique_ptr<X[]>& b_heap, const Nd4jLong num, Nd4jLong yStrideC)
			{
				//best results when buffer used much , may result bad perf if buffer is used once
				X* b_new = nullptr;
				if (yStrideC != 1) {
					if (num > b_stack_size) {
						b_heap.reset(new X[num]);
						b_new = b_heap.get();
					}
					else {
						b_new = b_stack;
					}
					for (size_t i = 0; i < num; i++) {
						b_new[i] = b_real[i * yStrideC];
					}
				}
				else {
					//no need , just pass normal bias
					return static_cast<const X*>(b_real);
				}
				return const_cast<const X*>(b_new);
			}

			template<typename X, typename Y>
			static typename std::enable_if<!std::is_same<X, Y>::value, const X*>::type
				flattened_bias(const Y* b_real, X* b_stack, const size_t b_stack_size, std::unique_ptr<X[]>& b_heap, const Nd4jLong num, Nd4jLong yStrideC)
			{
				//best results when buffer used much , may result bad perf if buffer is used once
				X* b_new = nullptr;
				if (num > b_stack_size) {
					b_heap.reset(new X[num]);
					b_new = b_heap.get();
				}
				else {
					b_new = b_stack;
				}
				if (yStrideC != 1) {
					for (size_t i = 0; i < num; i++) {
						b_new[i] = static_cast<X>(b_real[i * yStrideC]);
					}
				}
				else {
					for (size_t i = 0; i < num; i++) {
						b_new[i] = static_cast<X>(b_real[i]);
					}
				}
				return const_cast<const X*>(b_new);
			}

			template<typename T, size_t constRank>
			static void channel_atTheEnd_stride1_C(const Nd4jLong*& x_strides, const Nd4jLong*& bases, T* x, const T* b, T* z, const bool& inplace, const Nd4jLong& start, const Nd4jLong& stop, const Nd4jLong& inc)
			{
				size_t loop_count = (stop - start) / inc;
				sd::CoordsState<constRank - 1> cst;
				size_t offset = sd::init_coords<constRank>(cst, start, bases, x_strides);

				if (!inplace) {
					for (size_t i = 0; i < loop_count; i++) {
						_add(&(x[offset]), b, &(z[offset]), inc);
						offset = sd::inc_coords<constRank - 1>(cst, offset);
					}
				}
				else {
					for (size_t i = 0; i < loop_count; i++) {
						_add_inplace(&(x[offset]), b, inc);
						offset = sd::inc_coords<constRank - 1>(cst, offset);
					}
				}
			}


			template<typename T, size_t constRank >
			static void channel_atTheEnd_generic_C(const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong* z_strides, const bool& inplaceOp, const bool same_stride, const bool same_order, T* x, const T* b, T* z, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {

				//just ensure that passed sameStride is correct,  because when bases are equal orders matters 
				bool sameOrderStride = same_order && same_stride;
				if (sameOrderStride && x_strides[constRank - 1] == 1) {
					channel_atTheEnd_stride1_C<T, constRank>(x_strides, bases, x, b, z, inplaceOp, start, stop, inc);
				}
				else {
					size_t loop_count = (stop - start) / inc;
					sd::ZipCoordsState<constRank - 1> cst;
					sd::zip_size_t offset = sd::init_coords<constRank>(cst, start, bases, x_strides, z_strides);
					Nd4jLong x_stride = ZIP_STRIDE1(cst, constRank - 1);
					Nd4jLong z_stride = ZIP_STRIDE2(cst, constRank - 1);

					if (same_order && x_stride == 1 && z_stride == 1) {
						/* bases are equal with different strides , but the last one is 1. So we can still vectorize it  */
						for (size_t i = 0; i < loop_count; i++) {
							_add(&(x[offset.first]), b, &(z[offset.second]), inc);
							offset = sd::inc_coords<constRank - 1>(cst, offset);
						}
					}
					else {
						for (size_t i = 0; i < loop_count; i++) {
							T* xx = &(x[offset.first]);
							T* zz = &(z[offset.second]);
							for (size_t j = 0; j < inc; j++)
								zz[j * z_stride] = xx[j * x_stride] + b[j];
							offset = sd::inc_coords<constRank - 1>(cst, offset);
						}
					}
				}

			}

			/**
			* this is our main optimization which  benefits from everything for the continuous last_channel C order case
			* as it is intended for full continous we do not need any rank info
			*/
			template<typename T>
			void channel_atTheEnd_continous_C(T* x, const T* b, T* z, bool inplaceOp, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {
				size_t nums = (stop - start);
				size_t num_inc = nums - nums % inc;
				if (inplaceOp) {

					size_t offset_p = start;
					for (size_t i = 0; i < num_inc; i += inc) {
						_add_inplace<T>(&(x[offset_p]), b, inc);
						offset_p += inc;
					}
					if (nums > num_inc)
						_add_inplace<T>(&(x[offset_p]), b, nums - num_inc);
				}
				else {
					size_t offset_p = start;
					for (size_t i = 0; i < num_inc; i += inc) {
						_add<T>(&(x[offset_p]), b, &(z[offset_p]), inc);
						offset_p += inc;
					}
					if (nums > num_inc)
						_add<T>(&(x[offset_p]), b, &(z[offset_p]), nums - num_inc);
				}
			}

			template<typename T, typename T2, size_t constRank>
			static void channel_NC_stride1_C(const Nd4jLong*& x_strides, const Nd4jLong*& bases, T* x, const T2* b, T* z, const bool& inplace, const Nd4jLong yStrideC, const Nd4jLong& start, const Nd4jLong& stop, const Nd4jLong& inc)
			{
				size_t loop_count = (stop - start) / inc;
				sd::CoordsState<constRank - 1> cst;
				size_t offset = sd::init_coords<constRank>(cst, start, bases, x_strides);

				if (!inplace) {
					for (size_t i = 0; i < loop_count; i++) {
						T yy = static_cast<T>(b[COORDS(cst, 1) * yStrideC]);
						_add_broadcast(&(x[offset]), yy, &(z[offset]), inc);
						offset = sd::inc_coords<constRank - 1>(cst, offset);
					}
				}
				else {
					for (size_t i = 0; i < loop_count; i++) {
						T yy = static_cast<T>(b[COORDS(cst, 1) * yStrideC]);
						_add_broadcast_inplace(&(x[offset]), yy, inc);
						offset = sd::inc_coords<constRank - 1>(cst, offset);
					}
				}
			}

			template<typename T, typename T2, size_t constRank >
			void channel_NC_generic_C(const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong* z_strides, const bool& inplaceOp, const bool same_stride, const bool same_order, const Nd4jLong yStrideC, T* x, const T2* b, T* z, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {

				//just ensure that passed sameStride is correct,  because when bases are equal orders matters 

				bool sameOrderStride = same_order && same_stride;

				if (sameOrderStride && x_strides[constRank - 1] == 1) {
					channel_NC_stride1_C<T, T2, constRank>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, inc);
				}
				else {

					// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
					size_t loop_count = (stop - start) / inc;
					sd::ZipCoordsState<constRank - 1> cst;
					sd::zip_size_t offset = sd::init_coords<constRank>(cst, start, bases, x_strides, z_strides);
					Nd4jLong x_stride = ZIP_STRIDE1(cst, constRank - 1);
					Nd4jLong z_stride = ZIP_STRIDE2(cst, constRank - 1);
					if (same_order && z_stride == 1 && x_stride == 1) {
						/* bases are equal with different strides , but the last one is 1. So we can still vectorize it  */
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[ZIP_COORDS(cst, 1) * yStrideC]);
							_add_broadcast(&(x[offset.first]), yy, &(z[offset.second]), inc);
							offset = sd::inc_coords<constRank - 1>(cst, offset);
						}
					}
					else {
						for (size_t i = 0; i < loop_count; i++) {
							T* xx = &(x[offset.first]);
							T* zz = &(z[offset.second]);
							T yy = static_cast<T>(b[ZIP_COORDS(cst, 1) * yStrideC]);
							for (size_t j = 0; j < inc; j++)
								zz[j * z_stride] = xx[j * x_stride] + yy;
							offset = sd::inc_coords<constRank - 1>(cst, offset);
						}
					}
				}
			}

			///
			template<typename T, typename T2>
			void channel_NC_continous_numHW_C(Nd4jLong rank, const Nd4jLong* bases, const Nd4jLong* x_strides, T* x, const T2* b, T* z, bool inplaceOp, const Nd4jLong yStrideC, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {

				// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
				size_t loop_count = (stop - start) / inc;

				sd::CoordsState<1> cst;
				//note: we had to manually pass index
				size_t offset_p = sd::init_coords<2>(cst, start / inc, bases, x_strides);

				//partitioning was done using numHW, so we can increment from rank 2
				if (inplaceOp) {
					for (size_t i = 0; i < loop_count; i++) {
						T yy = static_cast<T>(b[COORDS(cst, 1) * yStrideC]);
						_add_broadcast_inplace(&(x[offset_p]), yy, inc);
						offset_p = sd::inc_coords<2>(cst, offset_p);
					}
				}
				else {
					if (yStrideC == 1) {
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[COORDS(cst, 1)]);
							_add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
							offset_p = sd::inc_coords<2>(cst, offset_p);
						}
					}
					else {
						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[COORDS(cst, 1) * yStrideC]);
							_add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
							offset_p = sd::inc_coords<2>(cst, offset_p);
						}
					}
				}
			}

			//
			template<typename T, typename T2, size_t constRank, size_t b_index, size_t skip>
			static void channel_generic_stride_skip_F(const Nd4jLong*& x_strides, const Nd4jLong*& bases, T* x, const T2* b, T* z, const bool& inplace, const Nd4jLong yStrideC, const Nd4jLong& start, const Nd4jLong& stop, const Nd4jLong& inc)
			{
				// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size
				size_t loop_count = (stop - start) / inc;
				sd::CoordsState<constRank - 1> cst;
				size_t offset_p = sd::init_coords<constRank, 0, false>(cst, start, bases, x_strides);
				if (!inplace) {
					for (size_t i = 0; i < loop_count; i++) {
						T yy = static_cast<T>(b[COORDS(cst, b_index) * yStrideC]);
						_add_broadcast(&(x[offset_p]), yy, &(z[offset_p]), inc);
						offset_p = sd::inc_coords<constRank, skip, false>(cst, offset_p);
					}
				}
				else {
					for (size_t i = 0; i < loop_count; i++) {
						T yy = static_cast<T>(b[COORDS(cst, b_index) * yStrideC]);
						_add_broadcast_inplace(&(x[offset_p]), yy, inc);
						offset_p = sd::inc_coords<constRank, skip, false>(cst, offset_p);
					}
				}
			}

			///
			template<typename T, typename T2, size_t constRank, size_t b_index>
			void channel_generic_F(const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong* z_strides, const bool& inplaceOp, const bool same_stride, const bool same_order, const Nd4jLong yStrideC, T* x, const T2* b, T* z, Nd4jLong start, Nd4jLong stop, Nd4jLong inc) {
				//just ensure that passed sameStride is correct,  because when bases are equal orders matters 
				bool sameOrderStride = same_order && same_stride;
				if (sameOrderStride && x_strides[0] == 1) {
					channel_generic_stride_skip_F<T, T2, constRank, b_index, 1>(x_strides, bases, x, b, z, inplaceOp, yStrideC, start, stop, inc);
				}
				else {
					// (stop-start) % inc == 0 because  we  handled inside partitioning using the channel size

					size_t loop_count = (stop - start) / inc;
					sd::ZipCoordsState<constRank - 1> cst;
					sd::zip_size_t offset = sd::init_coords<constRank, 0, false>(cst, start, bases, x_strides, z_strides);
					Nd4jLong x_stride = ZIP_STRIDE1(cst, 0);
					Nd4jLong z_stride = ZIP_STRIDE2(cst, 0);
					if (same_order && z_stride == 1 && x_stride == 1) {

						for (size_t i = 0; i < loop_count; i++) {
							T yy = static_cast<T>(b[ZIP_COORDS(cst, b_index) * yStrideC]);
							_add_broadcast(&(x[offset.first]), yy, &(z[offset.second]), inc);
							offset = sd::inc_coords<constRank, 1, false>(cst, offset);
						}
					}
					else {
						for (size_t i = 0; i < loop_count; i++) {
							T* xx = &(x[offset.first]);
							T* zz = &(z[offset.second]);
							T yy = static_cast<T>(b[ZIP_COORDS(cst, b_index) * yStrideC]);
							for (size_t j = 0; j < inc; j++)
								zz[j * z_stride] = xx[j * x_stride] + yy;
							offset = sd::inc_coords<constRank, 1, false>(cst, offset);
						}
					}
				}
			}


			template <typename X, typename Y>
			static void addBias_(const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW) {
			   /*
			    if (input.rankOf() == 2 && bias.rankOf() == 1 && input.sizeAt(1) == bias.sizeAt(0) && input.ordering() == 'c') {
			        int rows = input.sizeAt(0);
			        int biasLen = bias.lengthOf();

                    auto inB = input.bufferAsT<X>();
                    auto bB = bias.bufferAsT<Y>();
                    auto outB = output.bufferAsT<X>();

			        for (int e = 0; e < rows; e++) {
			            auto row = inB + (e * biasLen);
                        auto out = outB + (e * biasLen);

			            for (int t = 0; t < biasLen; t++) {
			                out[t] = row[t] + bB[t];
			            }
			        }

                    return;
			    }
			    */

				auto x_shapeInfo = input.shapeInfo();
				auto z_shapeInfo = output.shapeInfo();
				auto x = input.bufferAsT<X>();
				auto z = output.bufferAsT<X>();
				auto b = bias.bufferAsT<Y>();
				const Nd4jLong  rank = x_shapeInfo[0];
				auto bases = &(x_shapeInfo[1]);
				auto x_strides = &(x_shapeInfo[rank + 1]);
				auto z_strides = &(z_shapeInfo[rank + 1]);
				const bool inplaceOp = (x == z);
				const bool same_order = inplaceOp || (input.ordering() == output.ordering());
				const bool channel_atTheEnd = !isNCHW;
				const bool same_stride = inplaceOp || shape::strideEquals(x_shapeInfo, z_shapeInfo);
				bool isContinuous = false;
				int posOfNonUnityDim;
				bias.isCommonVector(posOfNonUnityDim);
				const Nd4jLong yStrideC = bias.strideAt(posOfNonUnityDim);
				char order = input.ordering();

				//for rank>5 
				if (rank > 5) {
					const int channelDim = isNCHW ? 1 : input.rankOf() - 1;      // second or last
					const_cast<NDArray&>(input).applyBroadcast(sd::broadcast::Add, { channelDim }, bias, output);
					return;
				}

				if (same_order && same_stride) {
					isContinuous = shape::elementWiseStride(x_shapeInfo) == 1 && shape::elementWiseStride(z_shapeInfo) == 1;
					// check_continuity(order, bases, x_strides, rank);
				}//if ( sameOrder && same_stride)

				bool treat_as_lastC = false;
				//
				if (rank == 2 && isNCHW) {
					//we believe we better treat it as channel at the end case;
					treat_as_lastC = true;
				}
				if (channel_atTheEnd || treat_as_lastC) {
					//N..HWC case here
					//flattened bias variables
					constexpr size_t BSIZE1 = 3 * MIN_NN * MIN_NN;
					constexpr size_t BSIZE2 = BSIZE1 + MIN_NN * MIN_NN;
					X  flatBias_stack[BSIZE2] align32;
					std::unique_ptr<X[]> flatBias_heap;
					const X* bias_new;
					X* bias_extra = nullptr;
					size_t total_num = 1;
					for (Nd4jLong i = 0; i < rank; i++) {
						total_num *= bases[i];
					}
					Nd4jLong inc;
					size_t rank_skip = 1;
					if (order == 'c') {
						size_t b_stack_size = BSIZE2;
						inc = bases[rank - 1];
						if (isContinuous) {
							//for continous we need extra stack memory
							// to create vectorizable bias from small size
							b_stack_size = BSIZE1;
							bias_extra = &(flatBias_stack[BSIZE1]);
						}
						bias_new = flattened_bias(b, (X*)flatBias_stack, b_stack_size, flatBias_heap, inc, yStrideC);
						if (isContinuous && inc < MIN_NN_K * MIN_NN && total_num > inc * MIN_NN_K) {
							//for small size where total_num is sufficient  we need to recreate vectorizable buffer
							size_t old_inc = inc;
							//sizeof bias_extra is MIN_NN * MIN_NN 
							size_t new_inc = inc < MIN_NN ? inc * MIN_NN : inc * MIN_NN / MIN_NN_K;
							//if there is a room then lets multiply
							new_inc = (new_inc * MIN_NN_K <= total_num && new_inc < MIN_NN * MIN_NN / MIN_NN_K) ? MIN_NN_K * new_inc : new_inc;
							for (size_t i = 0; i < new_inc; i += inc) {
								//copy to our buffer
								X* cp = &(bias_extra[i]);
								for (size_t j = 0; j < inc; j++) {
									cp[j] = bias_new[j];
								}
							}
							//vectorizable buffer
							inc = new_inc;
							bias_new = bias_extra;
						}
					}
					else {
						inc = bases[0];
						if (isContinuous) {
							//we can choose other inc and index for that case
							//but for now lets choose all till the last one
							uint32_t req_numThreads = sd::Environment::getInstance().maxMasterThreads();
							isContinuous = false;
							if (rank > 2) {
								if (req_numThreads < 2 || bases[rank - 1] >= req_numThreads) {
									inc = total_num / bases[rank - 1];
									isContinuous = true;
									rank_skip = rank - 1;
								}
								else if (rank > 3 && bases[rank - 1] * bases[rank - 2] >= req_numThreads) {
									inc = total_num / bases[rank - 1] / bases[rank - 2]; //for continuous case it is its stride
									rank_skip = rank - 2;
									isContinuous = true;
								}
							}
						}
					}

					FUNC_1D func = [order, isContinuous, rank, x, b, bias_new, z, x_shapeInfo, z_shapeInfo, same_stride, same_order, yStrideC, rank_skip]
					(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
						const Nd4jLong  rank = x_shapeInfo[0];
						auto bases = &(x_shapeInfo[1]);
						auto x_strides = &(x_shapeInfo[rank + 1]);
						auto z_strides = &(z_shapeInfo[rank + 1]);
						const bool inplaceOp = (x == z);
						if (order == 'c') {
							if (isContinuous) {
								channel_atTheEnd_continous_C(const_cast<X*>(x), bias_new, z, inplaceOp, start, stop, increment);
							}
							// rank is in [2,5]
							else if (rank == 4) {
								channel_atTheEnd_generic_C<X, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, const_cast<X*>(x), bias_new, z, start, stop, increment);

							}
							else if (rank == 5) {
								channel_atTheEnd_generic_C<X, 5>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, const_cast<X*>(x), bias_new, z, start, stop, increment);
							}
							else if (rank == 2) {
								channel_atTheEnd_generic_C<X, 2>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, const_cast<X*>(x), bias_new, z, start, stop, increment);
							}
							else if (rank == 3) {
								channel_atTheEnd_generic_C<X, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, const_cast<X*>(x), bias_new, z, start, stop, increment);
							}
						}
						else {
							//generic F case  
							if (isContinuous) {
								if (rank == 4) {
									if (rank_skip == rank - 2) {
										channel_generic_stride_skip_F<X, Y, 4, 3, 2>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp, yStrideC, start, stop, increment);
									}
									else {
										channel_generic_stride_skip_F<X, Y, 4, 3, 3>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp, yStrideC, start, stop, increment);
									}
								}
								else if (rank == 5) {
									if (rank_skip == rank - 2) {
										//skip==3
										channel_generic_stride_skip_F<X, Y, 5, 4, 3>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp, yStrideC, start, stop, increment);
									}
									else {
										channel_generic_stride_skip_F<X, Y, 5, 4, 4>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp, yStrideC, start, stop, increment);
									}
								}
								else if (rank == 3) {
									channel_generic_stride_skip_F<X, Y, 3, 2, 2>(x_strides, bases, const_cast<X*>(x), b, z, inplaceOp, yStrideC, start, stop, increment);
								}
							}
							else if (rank == 4) {
								channel_generic_F<X, Y, 4, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}
							else if (rank == 5) {
								channel_generic_F<X, Y, 5, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}
							else if (rank == 2) {
								channel_generic_F<X, Y, 2, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}
							else if (rank == 3) {
								channel_generic_F<X, Y, 3, 2>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}

						}
					};
					//
					samediff::Threads::parallel_aligned_increment(func, 0, total_num, inc);
				}
				else {
					//NC...HW case here
					size_t numNC = 1;
					size_t numHW = 1;
					for (size_t i = 0; i < 2; i++) {
						numNC *= bases[i];
					}
					for (Nd4jLong i = 2; i < rank; i++) {
						numHW *= bases[i];
					}
					Nd4jLong total_num = numNC * numHW;
					Nd4jLong inc = (order == 'c') ? bases[rank - 1] : bases[0];
					if (order == 'c' && isContinuous) {
						//sometimes last dimension is too big and multithreading could suffer using unfair partitioning
						//so we will do it only when inc is smaller our value or multithreading turned off
						uint32_t req_numThreads = sd::Environment::getInstance().maxMasterThreads();
						if (req_numThreads < 2 || numNC >= req_numThreads || inc <= 2 * 8196 || rank == 3) {
							inc = numHW;
						}
						else {
							//treat it as stride1c case
							isContinuous = false;
						}
					}
					FUNC_1D func = [order, isContinuous, rank, x, b, z, x_shapeInfo, z_shapeInfo, same_stride, same_order, yStrideC]
					(uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
						const Nd4jLong  rank = x_shapeInfo[0];
						const Nd4jLong* bases = &(x_shapeInfo[1]);
						const Nd4jLong* x_strides = &(x_shapeInfo[rank + 1]);
						const Nd4jLong* z_strides = &(z_shapeInfo[rank + 1]);
						const bool inplaceOp = (x == z);
						if (order == 'c') {
							if (isContinuous) {
								channel_NC_continous_numHW_C<X, Y>(rank, bases, x_strides, const_cast<X*>(x), b, z, inplaceOp, yStrideC, start, stop, increment);
							}
							// rank is in [3,5]
							else if (rank == 4) {
								channel_NC_generic_C<X, Y, 4>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);

							}
							else if (rank == 5) {
								channel_NC_generic_C<X, Y, 5>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}
							else if (rank == 3) {
								channel_NC_generic_C<X, Y, 3>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}
						}
						else {
							//the same can be applied for NCHW case
							//generic F case 
							//continous case is missing

							if (rank == 4) {
								channel_generic_F<X, Y, 4, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}
							else if (rank == 5) {
								channel_generic_F<X, Y, 5, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}
							else if (rank == 3) {
								channel_generic_F<X, Y, 3, 1>(bases, x_strides, z_strides, inplaceOp, same_stride, same_order, yStrideC, const_cast<X*>(x), b, z, start, stop, increment);
							}
						}
					};
					//
					samediff::Threads::parallel_aligned_increment(func, 0, total_num, inc);
				}
			}
			//////////////////////////////////////////////////////////////////////////
			void addBias(sd::graph::Context& block, const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW) {

			    // bias.rankOf() == 1 ? bias : bias.reshape(bias.ordering(), {bias.lengthOf()})
			    BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBias_, (input, bias, output, isNCHW), FLOAT_TYPES, FLOAT_TYPES);
			}


			BUILD_DOUBLE_TEMPLATE(template void addBias_, (const NDArray& input, const NDArray& bias, NDArray& output, const bool isNCHW), FLOAT_TYPES, FLOAT_TYPES);
		}
	}
}
