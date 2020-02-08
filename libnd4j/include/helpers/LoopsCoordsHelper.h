/*******************************************************************************
 *
 * Copyright (c) 2019 Konduit K.K.
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
 // @author AbdelRauf 
 //
#ifndef LIBND4J_LOOPCOORDSHELPER_H
#define LIBND4J_LOOPCOORDSHELPER_H

#include <cstddef>
#include <type_traits>
#include <utility>
#include <pointercast.h>
#include <op_boilerplate.h>
namespace nd4j {

#if defined(__GNUC__)
#define likely(x) __builtin_expect( (x), 1)	 
#define unlikely(x) __builtin_expect( (x), 0)
#else
#define likely(x)  (x)
#define unlikely(x)  (x)
#endif

	using zip_size_t = std::pair<size_t, size_t>;

	template<size_t Index>
	struct CoordsState :CoordsState<Index - 1> {
		Nd4jLong coord;
		Nd4jLong last_num;
		Nd4jLong stride;
		Nd4jLong adjust;
		CoordsState() :CoordsState<Index - 1>() {}
	};

	template<>
	struct CoordsState<0> {
		Nd4jLong coord;
		Nd4jLong last_num;
		Nd4jLong stride;
		Nd4jLong adjust;
		CoordsState() {}
	};


	template<size_t Index>
	struct ZipCoordsState :ZipCoordsState<Index - 1> {
		Nd4jLong coord;
		Nd4jLong last_num;
		Nd4jLong stride1;
		Nd4jLong stride2;
		Nd4jLong adjust1;
		Nd4jLong adjust2;
		ZipCoordsState() : ZipCoordsState<Index - 1>() {}
	};

	template<>
	struct ZipCoordsState<0> {
		Nd4jLong coord;
		Nd4jLong last_num;
		Nd4jLong stride1;
		Nd4jLong stride2;
		Nd4jLong adjust1;
		Nd4jLong adjust2;
		ZipCoordsState() {}
	};

#define COORDS(x,index)          ((x).::nd4j::CoordsState<(index)>::coord)
#define STRIDE(x,index)          ((x).::nd4j::CoordsState<(index)>::stride)
#define LAST_NUM(x,index)        ((x).::nd4j::CoordsState<(index)>::last_num)
#define OF_ADJUST(x,index)       ((x).::nd4j::CoordsState<(index)>::adjust)
#define ZIP_LAST_NUM(x,index)    ((x).::nd4j::ZipCoordsState<(index)>::last_num)
#define ZIP_COORDS(x,index)      ((x).::nd4j::ZipCoordsState<(index)>::coord)
#define ZIP_STRIDE1(x,index)     ((x).::nd4j::ZipCoordsState<(index)>::stride1)
#define ZIP_STRIDE2(x,index)     ((x).::nd4j::ZipCoordsState<(index)>::stride2)
#define ZIP_OF_ADJUST1(x,index)  ((x).::nd4j::ZipCoordsState<(index)>::adjust1)
#define ZIP_OF_ADJUST2(x,index)  ((x).::nd4j::ZipCoordsState<(index)>::adjust2)


	FORCEINLINE void   index2coords_C(Nd4jLong index, const Nd4jLong rank, const Nd4jLong* bases, Nd4jLong* coords) {
		for (size_t i = rank - 1; i > 0; --i) {
			coords[i] = index % bases[i];
			index /= bases[i];
		}
		coords[0] = index;      // last iteration 
	}

	FORCEINLINE void   index2coords_F(Nd4jLong index, const Nd4jLong rank, const Nd4jLong* bases, Nd4jLong* coords) {

		for (size_t i = 0; i < rank - 1; i++) {
			coords[i] = index % bases[i];
			index /= bases[i];
		}
		coords[rank - 1] = index;      // last iteration
	}

	FORCEINLINE size_t offset_from_coords(const Nd4jLong* strides, const Nd4jLong* coords, const  Nd4jLong& rank) {

		size_t offset = 0;
		size_t rank_4 = rank & -4;
		for (int i = 0; i < rank_4; i += 4) {
			offset = offset
				+ coords[i] * strides[i]
				+ coords[i + 1] * strides[i + 1]
				+ coords[i + 2] * strides[i + 2]
				+ coords[i + 3] * strides[i + 3];
		}
		for (int i = rank_4; i < rank; i++) {
			offset += coords[i] * strides[i];
		}
		return offset;
	}


	FORCEINLINE zip_size_t offset_from_coords(const Nd4jLong*& x_strides, const Nd4jLong*& z_strides, const Nd4jLong* coords, const Nd4jLong& rank) {

		zip_size_t offset = { 0,0 };
		size_t rank_4 = rank & -4;
		for (int i = 0; i < rank_4; i += 4) {
			offset.first = offset.first
				+ coords[i] * x_strides[i]
				+ coords[i + 1] * x_strides[i + 1]
				+ coords[i + 2] * x_strides[i + 2]
				+ coords[i + 3] * x_strides[i + 3];
			offset.second = offset.second
				+ coords[i] * z_strides[i]
				+ coords[i + 1] * z_strides[i + 1]
				+ coords[i + 2] * z_strides[i + 2]
				+ coords[i + 3] * z_strides[i + 3];
		}
		for (int i = rank_4; i < rank; i++) {
			offset.first += coords[i] * x_strides[i];
			offset.second += coords[i] * z_strides[i];
		}
		return offset;
	}

	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	constexpr size_t StridesOrderInd() {
		return Last_Index_Faster ? Rank - Index - 1 : Index;
	}

	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == Index), size_t>::type
		coord_inc_n(CoordsState<Rank - 1>& cbs, size_t last_offset) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, Last_Index_Faster>();

		if (likely(COORDS(cbs, Ind) < LAST_NUM(cbs, Ind))) {
			last_offset += cbs.CoordsState<Ind>::stride;
			COORDS(cbs, Ind) = COORDS(cbs, Ind) + 1;
			return last_offset;
		}
		//overflow case should not happen
		COORDS(cbs, Ind) = 0;
		//last_offset = 0;// last_offset + strides[Ind] - adjust_stride;
		return 0;
	}

	template<size_t Rank, size_t Index, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != Index), size_t >::type
		coord_inc_n(CoordsState<Rank - 1>& cbs, size_t last_offset) {

		constexpr size_t Ind = StridesOrderInd<Rank, Index, Last_Index_Faster>();

		if (likely(COORDS(cbs, Ind) < LAST_NUM(cbs, Ind))) {
			last_offset = last_offset + cbs.CoordsState<Ind>::stride;
			COORDS(cbs, Ind) = COORDS(cbs, Ind) + 1;
		}
		else {
			//lets adjust offset
			last_offset -= OF_ADJUST(cbs, Ind);
			COORDS(cbs, Ind) = 0;
			last_offset = coord_inc_n<Rank, Index + 1, Last_Index_Faster>(cbs, last_offset);
		}

		return last_offset;

	}

	template<size_t Rank, size_t Index = 0, bool Last_Index_Faster = true>
	FORCEINLINE size_t inc_coords(CoordsState<Rank - 1>& cbs, size_t last_offset) {

		return coord_inc_n<Rank, Index, Last_Index_Faster>(cbs,/* 1,*/ last_offset/*, 0*/);
	}

	template<size_t Rank, size_t rankIndex = 0, bool Last_Index_Faster = true>
	FORCEINLINE size_t inc_coords_ews(CoordsState<Rank - 1>& cbs, size_t last_offset, size_t ews) {
		if (ews == 1) {
			constexpr size_t Ind = StridesOrderInd<Rank, rankIndex, Last_Index_Faster>();
			return last_offset + STRIDE(cbs, Ind);
		}
		return coord_inc_n<Rank, rankIndex, Last_Index_Faster>(cbs,/* 1,*/ last_offset/*, 0*/);
	}

	template<size_t Rank, size_t rankIndex, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == rankIndex), zip_size_t>::type
		coord_inc_n(ZipCoordsState<Rank - 1>& cbs, zip_size_t last_offset) {

		constexpr size_t Ind = StridesOrderInd<Rank, rankIndex, Last_Index_Faster>();

		if (likely(ZIP_COORDS(cbs, Ind) < ZIP_LAST_NUM(cbs, Ind))) {
			last_offset.first += ZIP_STRIDE1(cbs, Ind);
			last_offset.second += ZIP_STRIDE2(cbs, Ind);
			ZIP_COORDS(cbs, Ind) = ZIP_COORDS(cbs, Ind) + 1;
			return last_offset;
		}
		//overflow case should not happen
		ZIP_COORDS(cbs, Ind) = 0;
		//last_offset = 0;// last_offset + strides[Ind] - adjust_stride;
		return { 0,0 };
	}

	template<size_t Rank, size_t rankIndex, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != rankIndex), zip_size_t >::type
		coord_inc_n(ZipCoordsState<Rank - 1>& cbs, zip_size_t last_offset) {

		constexpr size_t Ind = StridesOrderInd<Rank, rankIndex, Last_Index_Faster>();

		if (likely(ZIP_COORDS(cbs, Ind) < ZIP_LAST_NUM(cbs, Ind))) {
			last_offset.first += ZIP_STRIDE1(cbs, Ind);
			last_offset.second += ZIP_STRIDE2(cbs, Ind);
			ZIP_COORDS(cbs, Ind) = ZIP_COORDS(cbs, Ind) + 1;
		}
		else {

			//lets adjust offset
			last_offset.first -= ZIP_OF_ADJUST1(cbs, Ind);
			last_offset.second -= ZIP_OF_ADJUST2(cbs, Ind);
			ZIP_COORDS(cbs, Ind) = 0;
			last_offset = coord_inc_n<Rank, rankIndex + 1, Last_Index_Faster>(cbs, last_offset);
		}

		return last_offset;

	}

	template<size_t Rank, size_t rankIndex = 0, bool Last_Index_Faster = true>
	FORCEINLINE zip_size_t inc_coords(ZipCoordsState<Rank - 1>& cbs, zip_size_t last_offset) {

		return coord_inc_n<Rank, rankIndex, Last_Index_Faster>(cbs, last_offset);
	}


	template<size_t Rank, size_t rankIndex = 0, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == rankIndex), size_t>::type
		init_coords(CoordsState<Rank - 1>& cbs, const Nd4jLong index, const Nd4jLong* bases, const Nd4jLong* strides, size_t offset = 0) {
		constexpr size_t Ind = StridesOrderInd<Rank, rankIndex, Last_Index_Faster>();
		COORDS(cbs, Ind) = index % bases[Ind];
		LAST_NUM(cbs, Ind) = bases[Ind] - 1;
		STRIDE(cbs, Ind) = strides[Ind];
		OF_ADJUST(cbs, Ind) = bases[Ind] * strides[Ind] - strides[Ind];
		offset += COORDS(cbs, Ind) * strides[Ind];
		return offset;
	}



	template<size_t Rank, size_t rankIndex = 0, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != rankIndex), size_t>::type
		init_coords(CoordsState<Rank - 1>& cbs, const Nd4jLong index, const Nd4jLong* bases, const Nd4jLong* strides, size_t offset = 0) {
		constexpr size_t Ind = StridesOrderInd<Rank, rankIndex, Last_Index_Faster>();
		COORDS(cbs, Ind) = index % bases[Ind];
		LAST_NUM(cbs, Ind) = bases[Ind] - 1;
		STRIDE(cbs, Ind) = strides[Ind];
		OF_ADJUST(cbs, Ind) = bases[Ind] * strides[Ind] - strides[Ind];
		offset += COORDS(cbs, Ind) * strides[Ind];
		return init_coords<Rank, rankIndex + 1, Last_Index_Faster>(cbs, index / bases[Ind], bases, strides, offset);
	}




	template<size_t Rank, size_t rankIndex = 0, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == rankIndex), bool>::type
		eq_coords(CoordsState<Rank - 1>& cbs, const Nd4jLong* coords) {
		return COORDS(cbs, rankIndex) == coords[rankIndex];
	}

	template<size_t Rank, size_t rankIndex = 0>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != rankIndex), bool>::type
		eq_coords(CoordsState<Rank - 1>& cbs, const Nd4jLong* coords) {
		return COORDS(cbs, rankIndex) == coords[rankIndex] && eq_coords<Rank, rankIndex + 1>(cbs, coords);
	}


	template<size_t Rank, size_t rankIndex = 0, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == rankIndex), bool>::type
		eq_zip_coords(ZipCoordsState<Rank - 1>& cbs, const Nd4jLong* coords) {
		return ZIP_COORDS(cbs, rankIndex) == coords[rankIndex];
	}

	template<size_t Rank, size_t rankIndex = 0>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != rankIndex), bool>::type
		eq_zip_coords(ZipCoordsState<Rank - 1>& cbs, const Nd4jLong* coords) {
		return ZIP_COORDS(cbs, rankIndex) == coords[rankIndex] && eq_zip_coords<Rank, rankIndex + 1>(cbs, coords);
	}

	template<size_t Rank, size_t rankIndex = 0, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 == rankIndex), zip_size_t>::type
		init_coords(ZipCoordsState<Rank - 1>& cbs, const Nd4jLong index, const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong* z_strides, zip_size_t offset = {}) {
		constexpr size_t Ind = StridesOrderInd<Rank, rankIndex, Last_Index_Faster>();
		ZIP_COORDS(cbs, Ind) = index % bases[Ind];
		ZIP_LAST_NUM(cbs, Ind) = bases[Ind] - 1;
		ZIP_STRIDE1(cbs, Ind) = x_strides[Ind];
		ZIP_STRIDE2(cbs, Ind) = z_strides[Ind];
		ZIP_OF_ADJUST1(cbs, Ind) = ZIP_LAST_NUM(cbs, Ind) * ZIP_STRIDE1(cbs, Ind);
		ZIP_OF_ADJUST2(cbs, Ind) = ZIP_LAST_NUM(cbs, Ind) * ZIP_STRIDE2(cbs, Ind);
		offset.first += ZIP_COORDS(cbs, Ind) * ZIP_STRIDE1(cbs, Ind);
		offset.second += ZIP_COORDS(cbs, Ind) * ZIP_STRIDE2(cbs, Ind);
		return offset;
	}

	template<size_t Rank, size_t rankIndex = 0, bool Last_Index_Faster = true>
	FORCEINLINE
		typename std::enable_if<(Rank - 1 != rankIndex), zip_size_t>::type
		init_coords(ZipCoordsState<Rank - 1>& cbs, const Nd4jLong index, const Nd4jLong* bases, const Nd4jLong* x_strides, const Nd4jLong* z_strides, zip_size_t offset = {}) {
		constexpr size_t Ind = StridesOrderInd<Rank, rankIndex, Last_Index_Faster>();
		ZIP_COORDS(cbs, Ind) = index % bases[Ind];
		ZIP_LAST_NUM(cbs, Ind) = bases[Ind] - 1;
		ZIP_STRIDE1(cbs, Ind) = x_strides[Ind];
		ZIP_STRIDE2(cbs, Ind) = z_strides[Ind];
		ZIP_OF_ADJUST1(cbs, Ind) = ZIP_LAST_NUM(cbs, Ind) * ZIP_STRIDE1(cbs, Ind);
		ZIP_OF_ADJUST2(cbs, Ind) = ZIP_LAST_NUM(cbs, Ind) * ZIP_STRIDE2(cbs, Ind);
		offset.first += ZIP_COORDS(cbs, Ind) * ZIP_STRIDE1(cbs, Ind);
		offset.second += ZIP_COORDS(cbs, Ind) * ZIP_STRIDE2(cbs, Ind);
		return init_coords<Rank, rankIndex + 1, Last_Index_Faster>(cbs, index / bases[Ind], bases, x_strides, z_strides, offset);
	}


	//inc coords for non constant Ranks
	template<bool Last_Index_Faster = true>
	FORCEINLINE size_t inc_coords(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong* coords, size_t last_offset, const size_t rank, const size_t skip = 0) {

		Nd4jLong  val;
		for (int i = rank - skip - 1; i >= 0; i--) {
			val = coords[i] + 1;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset += strides[i];
				break;
			}
			else {
				last_offset -= coords[i] * strides[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}

	template<>
	FORCEINLINE size_t inc_coords<false>(const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong* coords, size_t last_offset, const size_t rank, const size_t skip) {

		Nd4jLong  val;
		for (int i = skip; i < rank; i++) {
			val = coords[i] + 1;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset += strides[i];
				break;
			}
			else {
				last_offset -= coords[i] * strides[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}


	template<bool Last_Index_Faster = true>
	FORCEINLINE zip_size_t inc_coords(const Nd4jLong* bases, const Nd4jLong* x_strides, const  Nd4jLong* z_strides, Nd4jLong* coords, zip_size_t last_offset, const size_t rank, const size_t skip = 0) {

		Nd4jLong  val = 0;
		for (int i = rank - skip - 1; i >= 0; i--) {
			val = coords[i] + 1;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset.first += x_strides[i];
				last_offset.second += z_strides[i];
				break;
			}
			else {
				last_offset.first -= coords[i] * x_strides[i];
				last_offset.second -= coords[i] * z_strides[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}

	template<>
	FORCEINLINE zip_size_t inc_coords<false>(const Nd4jLong* bases, const Nd4jLong* x_strides, const  Nd4jLong* z_strides, Nd4jLong* coords, zip_size_t last_offset, const size_t rank, const size_t skip) {

		Nd4jLong  val = 0;
		for (int i = skip; i < rank; i++) {
			val = coords[i] + 1;
			if (likely(val < bases[i])) {
				coords[i] = val;

				last_offset.first += x_strides[i];
				last_offset.second += z_strides[i];
				break;
			}
			else {
				last_offset.first -= coords[i] * x_strides[i];
				last_offset.second -= coords[i] * z_strides[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}

}

#endif