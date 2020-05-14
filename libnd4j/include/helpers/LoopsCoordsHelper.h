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
#include <vector>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <system/pointercast.h>
#include <system/op_boilerplate.h>
#include <helpers/shape.h>
namespace sd {

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

#define COORDS(x,index)          ((x).::sd::CoordsState<(index)>::coord)
#define STRIDE(x,index)          ((x).::sd::CoordsState<(index)>::stride)
#define LAST_NUM(x,index)        ((x).::sd::CoordsState<(index)>::last_num)
#define OF_ADJUST(x,index)       ((x).::sd::CoordsState<(index)>::adjust)
#define ZIP_LAST_NUM(x,index)    ((x).::sd::ZipCoordsState<(index)>::last_num)
#define ZIP_COORDS(x,index)      ((x).::sd::ZipCoordsState<(index)>::coord)
#define ZIP_STRIDE1(x,index)     ((x).::sd::ZipCoordsState<(index)>::stride1)
#define ZIP_STRIDE2(x,index)     ((x).::sd::ZipCoordsState<(index)>::stride2)
#define ZIP_OF_ADJUST1(x,index)  ((x).::sd::ZipCoordsState<(index)>::adjust1)
#define ZIP_OF_ADJUST2(x,index)  ((x).::sd::ZipCoordsState<(index)>::adjust2)


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


	FORCEINLINE zip_size_t offset_from_coords(const Nd4jLong* x_strides, const Nd4jLong* z_strides, const Nd4jLong* coords, const Nd4jLong& rank) {

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


	struct triple_size_t {
		size_t first;
		size_t second;
		size_t third;
	};


	template<bool Last_Index_Faster = true>
	FORCEINLINE triple_size_t inc_coords(const Nd4jLong* bases, const Nd4jLong* x_strides, const  Nd4jLong* y_strides, const  Nd4jLong* z_strides, Nd4jLong* coords, triple_size_t last_offset, const size_t rank, const size_t skip = 0) {

		Nd4jLong  val = 0;
		for (int i = rank - skip - 1; i >= 0; i--) {
			val = coords[i] + 1;
			if (likely(val < bases[i])) {
				coords[i] = val;
				last_offset.first += x_strides[i];
				last_offset.second += y_strides[i];
				last_offset.third += z_strides[i];
				break;
			}
			else {
				last_offset.first -= coords[i] * x_strides[i];
				last_offset.second -= coords[i] * y_strides[i];
				last_offset.third -= coords[i] * z_strides[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}

	template<>
	FORCEINLINE triple_size_t inc_coords<false>(const Nd4jLong* bases, const Nd4jLong* x_strides, const  Nd4jLong* y_strides, const  Nd4jLong* z_strides, Nd4jLong* coords, triple_size_t last_offset, const size_t rank, const size_t skip) {

		Nd4jLong  val = 0;
		for (int i = skip; i < rank; i++) {
			val = coords[i] + 1;
			if (likely(val < bases[i])) {
				coords[i] = val;

				last_offset.first += x_strides[i];
				last_offset.second += y_strides[i];
				last_offset.third += z_strides[i];
				break;
			}
			else {
				last_offset.first -= coords[i] * x_strides[i];
				last_offset.second -= coords[i] * y_strides[i];
				last_offset.third -= coords[i] * z_strides[i];
				coords[i] = 0;
			}
		}
		return last_offset;
	}

	FORCEINLINE triple_size_t offset_from_coords(const Nd4jLong* x_strides, const  Nd4jLong* y_strides, const  Nd4jLong* z_strides, const Nd4jLong* coords, const Nd4jLong& rank) {

		triple_size_t offset = { 0,0 ,0 };
		size_t rank_4 = rank & -4;
		for (int i = 0; i < rank_4; i += 4) {
			offset.first = offset.first
				+ coords[i] * x_strides[i]
				+ coords[i + 1] * x_strides[i + 1]
				+ coords[i + 2] * x_strides[i + 2]
				+ coords[i + 3] * x_strides[i + 3];
			offset.second = offset.second
				+ coords[i] * y_strides[i]
				+ coords[i + 1] * y_strides[i + 1]
				+ coords[i + 2] * y_strides[i + 2]
				+ coords[i + 3] * y_strides[i + 3];
			offset.third = offset.third
				+ coords[i] * z_strides[i]
				+ coords[i + 1] * z_strides[i + 1]
				+ coords[i + 2] * z_strides[i + 2]
				+ coords[i + 3] * z_strides[i + 3];
		}
		for (int i = rank_4; i < rank; i++) {
			offset.first += coords[i] * x_strides[i];
			offset.second += coords[i] * y_strides[i];
			offset.third += coords[i] * z_strides[i];
		}
		return offset;
	}


	template<bool Last_Index_Faster = true>
	FORCEINLINE Nd4jLong getLength(const Nd4jLong* bases, int rank, int skip = 0)
	{
		if (skip < 0 || skip >= rank) skip = 0;
		Nd4jLong total = 1;
		for (int i = 0; i < rank - skip; i++) {
			total *= bases[i];
		}
		return total;
	}


	template<>
	FORCEINLINE Nd4jLong getLength<false>(const Nd4jLong* bases, int rank, int skip)
	{
		if (skip < 0 || skip >= rank) skip = 0;
		Nd4jLong total = 1;
		for (int i = skip; i < rank; i++) {
			total *= bases[i];
		}

		return total;
	}


	template<bool Last_Index_Faster = true>
	FORCEINLINE Nd4jLong getLength(const Nd4jLong* bases, int rank, int skip, Nd4jLong& outSkippedLength)
	{
		if (skip < 0 || skip >= rank) skip = 0;
		Nd4jLong total = 1;
		for (int i = 0; i < rank - skip; i++) {
			total *= bases[i];
		}
		if (skip > 0) {
			outSkippedLength = 1;
			for (int i = rank - skip; i < rank; i++) {
				outSkippedLength *= bases[i];
			}
		}
		else {
			outSkippedLength = 0;
		}
		return total;
	}


	template<>
	FORCEINLINE Nd4jLong getLength<false>(const Nd4jLong* bases, int rank, int skip, Nd4jLong& outSkippedLength)
	{
		if (skip < 0 || skip >= rank) skip = 0;
		if (skip > 0) {
			outSkippedLength = 1;
			for (int i = 0; i < skip; i++) {
				outSkippedLength *= bases[i];
			}
		}
		else {
			outSkippedLength = 0;
		}
		Nd4jLong total = 1;
		for (int i = skip; i < rank; i++) {
			total *= bases[i];
		}

		return total;
	}

	/*
	for ODR rule it willbe declared as inline
	rePartition for reductions and et cet
	Indices mentioned in the dimension list will be moved to the tail
	This way it will be splitted into two parts
	the first part will contain output part,the second tail part will be used for reductions and other purposes
	if squash is True then  it will attempt to minimize the output ( for both orders) and the tail
*/

	FORCEINLINE void rePartition(char order, const std::vector<int>& dimensions, const size_t rank, const Nd4jLong* bases, const Nd4jLong* strides, Nd4jLong(&new_bases)[MAX_RANK], Nd4jLong(&new_strides)[MAX_RANK], int& first_begin, int& first_end, int& second_begin, int& second_end, bool first_squash = false, bool second_squash = true) {

		bool indices[MAX_RANK] = {};
		int ind = 0;
		size_t second_rank;
		if (dimensions.size() == 0 || (dimensions.size() == 1 && dimensions.at(0) == sd::DataTypeUtils::max<int>())){
			first_end = 0;
			first_begin = 0;
			//treat it as the whole
			for (int i = 0; i < rank; i++) {
				new_bases[i] = bases[i];
				new_strides[i] = strides[i];
			}
			second_rank = rank;
			second_end = rank;
			second_begin = 0;

		}
		else {
			for (int index : dimensions) {
				if (index < 0) index = rank + index;
				if (index >= 0 && index < rank) {
					indices[index] = true;
				}
			}


			//move output ones and
			for (int i = 0; i < rank; i++) {

				if (!indices[i]) {

					new_bases[ind] = bases[i];
					new_strides[ind] = strides[i];
					ind++;
				}
			}


			int first_rank = ind;

			first_end = ind;
			first_begin = 0;
			//nd4j_printf("rffrr ss & %d ind-- %d %d\n", first_rank, first_begin, first_end);
			//squash output rank 
			if (first_squash && first_rank > 1) {

				if (order == 'c') {
					int uniq_ind = first_end-1;
					for (int i = first_end - 2; i >= first_begin; i--) {
						if (new_strides[i] == new_bases[uniq_ind] * new_strides[uniq_ind]) {
							new_bases[uniq_ind] = new_bases[i] * new_bases[uniq_ind];
							new_strides[uniq_ind] = new_strides[uniq_ind];
							--first_rank;
						}
						else {
							--uniq_ind;
							new_bases[uniq_ind] = new_bases[i];
							new_strides[uniq_ind] = new_strides[i];
						}
					}
					first_begin = first_end - first_rank;
				}
				else {
					//squash fortran 
					int uniq_ind = 0;
					for (int i = 1; i < first_end; i++) {
						if (new_strides[i] == new_bases[uniq_ind] * new_strides[uniq_ind]) {
							new_bases[uniq_ind] = new_bases[i] * new_bases[uniq_ind];
							new_strides[uniq_ind] = new_strides[uniq_ind];
							--first_rank;
						}
						else {
							uniq_ind++;
							new_bases[uniq_ind] = new_bases[i];
							new_strides[uniq_ind] = new_strides[i];
						}
					}
					first_end = first_begin + first_rank;

				} 
				ind = first_end;
			}

			//nd4j_printf("rffrr ss & %d ind-- %d %d\n", first_rank, first_begin, first_end);
			//move process indices
			for (int i = 0; i < rank; i++) {
				if (indices[i]) {
					new_bases[ind] = bases[i];
					new_strides[ind] = strides[i];
					ind++;
				}
			}

			second_rank = ind - first_end;
			second_end = ind;
			second_begin = first_end;

		}
		

		if (second_squash && second_rank > 1) {

			if (order == 'c') {
				int uniq_ind = second_end - 1;
				for (int i = second_end - 2; i >= second_begin; i--) {
					if (new_strides[i] == new_bases[uniq_ind] * new_strides[uniq_ind]) {
						new_bases[uniq_ind] = new_bases[i] * new_bases[uniq_ind];
						new_strides[uniq_ind] = new_strides[uniq_ind];
						--second_rank;
					}
					else {
						--uniq_ind;
						new_bases[uniq_ind] = new_bases[i];
						new_strides[uniq_ind] = new_strides[i];
					}
				}
				second_begin = second_end - second_rank;
			}
			else {
				int uniq_ind = second_begin;
				for (int i = second_begin+1; i < second_end; i++) {
					if (new_strides[i] == new_bases[uniq_ind] * new_strides[uniq_ind]) {
						new_bases[uniq_ind] = new_bases[i] * new_bases[uniq_ind];
						new_strides[uniq_ind] = new_strides[uniq_ind];
						--second_rank;
					}
					else {
						uniq_ind++;
						new_bases[uniq_ind] = new_bases[i];
						new_strides[uniq_ind] = new_strides[i];
					}
				}
				second_end = second_begin + second_rank;

			}
			
		}

		return;
	}

	//basic CRTP static polymorphism classes for offset increments

	template<typename Derived>
	struct CoordsBaseMovement {
		void init(const Nd4jLong* bases, const Nd4jLong* strides1, const Nd4jLong* strides2, int rank, int start = 0) {
			static_cast<Derived*>(this)->initImpl(bases, strides1, strides2, rank, start);
		}

		void increment(int skipRank = 0) {
			static_cast<Derived*>(this)->incrementImpl(skipRank);
		}

		Nd4jLong  First() { return static_cast<Derived*>(this)->FirstImpl(); };
		Nd4jLong  Second() { return static_cast<Derived*>(this)->SecondImpl(); };
	};


	struct ZipGenericCoordsRank1Stride1 : CoordsBaseMovement<ZipGenericCoordsRank1Stride1> {

		size_t offset1;
		size_t offset2;


		void initImpl(const Nd4jLong* bases, const Nd4jLong* strides1, const Nd4jLong* strides2, int rank, int start = 0) {
			offset1 = start;
			offset2 = start;
		}

		void incrementImpl(int skipRank = 0) {
			offset1 += 1;
			offset2 += 1;
		}

		Nd4jLong  FirstImpl() { return offset1; };
		Nd4jLong  SecondImpl() { return offset2; };

	};

	struct ZipGenericCoordsRank1BothStrideN : CoordsBaseMovement<ZipGenericCoordsRank1BothStrideN> {
		size_t stride1;
		size_t stride2;
		size_t offset1;
		size_t offset2;


		void initImpl(const Nd4jLong* bases, const Nd4jLong* strides1, const Nd4jLong* strides2, int rank, int start = 0) {
			stride1 = strides1[0];
			stride2 = strides2[0];
			offset1 = start * stride1;
			offset2 = start * stride2;
		}

		void incrementImpl(int skipRank = 0) {
			offset1 += stride1;
			offset2 += stride2;
		}

		Nd4jLong  FirstImpl() { return offset1; };
		Nd4jLong  SecondImpl() { return offset2; };

	};

	template<int ConstRank, bool LastIndexFaster = true>
	struct ZipGenericCoordsConstMovementSecondStride1 : CoordsBaseMovement<ZipGenericCoordsConstMovementSecondStride1<ConstRank, LastIndexFaster>> {
		sd::CoordsState<ConstRank - 1> cst;
		Nd4jLong coords[MAX_RANK];
		size_t offset1;
		size_t offset2;
		int _rank;

		void initImpl(const Nd4jLong* bases, const Nd4jLong* strides1, const Nd4jLong* strides2, int rank, int start = 0) {
			offset1 = sd::init_coords<ConstRank, 0, LastIndexFaster>(cst, start, bases, strides1);
			offset2 = start * 1;
		}

		void incrementImpl(int skipRank = 0) {
			offset1 = sd::inc_coords<ConstRank, 0, LastIndexFaster>(cst, offset1);
			offset2 += 1;
		}

		Nd4jLong  FirstImpl() { return offset1; };
		Nd4jLong  SecondImpl() { return offset2; };

	};

	template<int ConstRank, bool LastIndexFaster = true>
	struct ZipGenericCoordsConstMovementSecondStrideN : CoordsBaseMovement<ZipGenericCoordsConstMovementSecondStrideN<ConstRank, LastIndexFaster>> {
		sd::CoordsState<ConstRank - 1> cst;
		Nd4jLong _stride2;
		Nd4jLong coords[MAX_RANK];
		size_t offset1;
		size_t offset2;
		int _rank;

		void initImpl(const Nd4jLong* bases, const Nd4jLong* strides1, const Nd4jLong* strides2, int rank, int start = 0) {
			_stride2 = strides2[0];
			offset1 = sd::init_coords<ConstRank, 0, LastIndexFaster>(cst, start, bases, strides1);
			offset2 = start * _stride2;
		}

		void incrementImpl(int skipRank = 0) {
			offset1 = sd::inc_coords<ConstRank, 0, LastIndexFaster>(cst, offset1);
			offset2 += _stride2;
		}

		Nd4jLong  FirstImpl() { return offset1; };
		Nd4jLong  SecondImpl() { return offset2; };

	};

	template<bool LastIndexFaster = true>
	struct ZipGenericCoordsMovementSecondStrideN : CoordsBaseMovement<ZipGenericCoordsMovementSecondStrideN<LastIndexFaster>> {
		const Nd4jLong* _bases;
		const Nd4jLong* _strides1;
		Nd4jLong _stride2;
		Nd4jLong coords[MAX_RANK];
		zip_size_t offset;
		int _rank;

		void initImpl(const Nd4jLong* bases, const Nd4jLong* strides1, const Nd4jLong* strides2, int rank, int start = 0) {

			_bases = bases;
			_strides1 = strides1;
			_stride2 = strides2[0];
			_rank = rank;
			if (start == 0) {
				for (int i = 0; i < MAX_RANK; i++) {
					coords[i] = 0;
				}
				offset = { 0,0 };

			}
			else {
				if (LastIndexFaster) {
					sd::index2coords_C(start, rank, bases, (Nd4jLong*)&coords);
				}
				else {
					sd::index2coords_F(start, rank, bases, (Nd4jLong*)&coords);
				}
				offset.first = sd::offset_from_coords(strides1, (Nd4jLong*)&coords, rank);
				offset.second = start * _stride2;
			}

		}

		void incrementImpl(int skipRank = 0) {
			offset.first = inc_coords<LastIndexFaster>(_bases, _strides1, (Nd4jLong*)&coords, offset.first, _rank, skipRank);
			offset.second += _stride2;
		}

		Nd4jLong  FirstImpl() { return offset.first; };
		Nd4jLong  SecondImpl() { return offset.second; };

	};

	template<bool LastIndexFaster = true>
	struct ZipGenericCoordsMovement : CoordsBaseMovement<ZipGenericCoordsMovement<LastIndexFaster>> {
		const Nd4jLong* _bases;
		const Nd4jLong* _strides1;
		const Nd4jLong* _strides2;
		Nd4jLong coords[MAX_RANK];
		zip_size_t offset;
		int _rank;

		void initImpl(const Nd4jLong* bases, const Nd4jLong* strides1, const Nd4jLong* strides2, int rank, int start = 0) {

			_bases = bases;
			_strides1 = strides1;
			_strides2 = strides2;
			_rank = rank;
			if (start == 0) {
				for (int i = 0; i < MAX_RANK; i++) {
					coords[i] = 0;
				}
				offset = { 0,0 };

			}
			else {
				if (LastIndexFaster) {
					sd::index2coords_C(start, rank, bases, (Nd4jLong*)&coords);
				}
				else {
					sd::index2coords_F(start, rank, bases, (Nd4jLong*)&coords);
				}
				offset = sd::offset_from_coords(strides1, strides2, (Nd4jLong*)&coords, rank);
			}

		}

		void incrementImpl(int skipRank = 0) {
			offset = inc_coords<LastIndexFaster>(_bases, _strides1, _strides2, (Nd4jLong*)&coords, offset, _rank, skipRank);
		}

		Nd4jLong  FirstImpl() { return offset.first; };
		Nd4jLong  SecondImpl() { return offset.second; };

	};

}



#endif