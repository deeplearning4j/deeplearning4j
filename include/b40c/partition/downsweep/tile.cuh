/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Abstract tile-processing functionality for partitioning downsweep scan
 * kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>
#include <b40c/util/device_intrinsics.cuh>

namespace b40c {
namespace partition {
namespace downsweep {


/**
 * Tile
 *
 * Abstract class
 */
template <
	typename KernelPolicy,
	typename DerivedTile>
struct Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType 				ValueType;
	typedef typename KernelPolicy::SizeT 					SizeT;

	typedef DerivedTile Dispatch;

	enum {
		LOAD_VEC_SIZE 				= KernelPolicy::LOAD_VEC_SIZE,
		LOADS_PER_CYCLE 			= KernelPolicy::LOADS_PER_CYCLE,
		CYCLES_PER_TILE 			= KernelPolicy::CYCLES_PER_TILE,
		TILE_ELEMENTS_PER_THREAD 	= KernelPolicy::TILE_ELEMENTS_PER_THREAD,
		SCAN_LANES_PER_CYCLE		= KernelPolicy::SCAN_LANES_PER_CYCLE,

		INVALID_BIN					= -1,
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------


	// The keys (and values) this thread will read this cycle
	KeyType 	keys[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];
	ValueType 	values[TILE_ELEMENTS_PER_THREAD];

	int 		local_ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];		// The local rank of each key
	int 		key_bins[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];			// The bin for each key
	SizeT 		scatter_offsets[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];	// The global rank of each key
	int 		counter_offsets[LOADS_PER_CYCLE][LOAD_VEC_SIZE];					// The (byte) counter offset for each key

	// Counts of my bin in each load in each cycle, valid in threads [0,BINS)
	int 		bin_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE];


	//---------------------------------------------------------------------
	// Abstract Interface
	//---------------------------------------------------------------------

	/**
	 * Returns whether or not the key is valid.
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ SizeT ValidElements(Cta *cta, const SizeT &guarded_elements)
	{
		return guarded_elements;
	}

	/**
	 * Returns the bin into which the specified key is to be placed.
	 *
	 * To be overloaded
	 */
	template <typename Cta>
	__device__ __forceinline__ int DecodeBin(KeyType key, Cta *cta);


	/**
	 * Returns whether or not the key is valid.
	 *
	 * To be overloaded.
	 */
	template <int CYCLE, int LOAD, int VEC>
	__device__ __forceinline__ bool IsValid();


	/**
	 * Loads keys into the tile
	 *
	 * Can be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				(KeyType (*)[KernelPolicy::LOAD_VEC_SIZE]) keys,
				cta->d_in_keys,
				cta_offset,
				guarded_elements);
	}


	/**
	 * Scatter keys from the tile
	 *
	 * Can be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterKeys(
		Cta *cta,
		const SizeT &guarded_elements)
	{
		// Scatter keys to global bin partitions
		util::io::ScatterTile<
			KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
			0,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				cta->d_out_keys,
				(KeyType (*)[1]) keys,
				(SizeT (*)[1]) scatter_offsets,
				guarded_elements);
	}


	/**
	 * Loads values into the tile
	 *
	 * Can be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadValues(
		Cta *cta,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		// Read values
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				(ValueType (*)[KernelPolicy::LOAD_VEC_SIZE]) values,
				cta->d_in_values,
				cta_offset,
				guarded_elements);
	}


	/**
	 * Scatter values from the tile
	 *
	 * Can be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterValues(
		Cta *cta,
		const SizeT &guarded_elements)
	{
		// Scatter values to global bin partitions
		util::io::ScatterTile<
			KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
			0,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				cta->d_out_values,
				(ValueType (*)[1]) values,
				(SizeT (*)[1]) scatter_offsets,
				guarded_elements);
	}


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Computes the number of previously-binned keys owned by the calling thread
	 * that have been marked for the specified bin.
	 */
	struct SameBinCount
	{
		// Inspect previous vec-element
		template <int CYCLE, int LOAD, int VEC>
		struct Iterate
		{
			static __device__ __forceinline__ int Invoke(Tile *tile, int current_bin)
			{
				return (current_bin == tile->key_bins[CYCLE][LOAD][VEC - 1]) +
					Iterate<CYCLE, LOAD, VEC - 1>::Invoke(tile, current_bin);
			}
		};

		// Terminate (0th vec-element has no previous elements)
		template <int CYCLE, int LOAD>
		struct Iterate<CYCLE, LOAD, 0>
		{
			static __device__ __forceinline__ int Invoke(Tile *tile, int current_bin)
			{
				return 0;
			}
		};
	};


	//---------------------------------------------------------------------
	// Cycle Methods
	//---------------------------------------------------------------------


	/**
	 * DecodeKeys
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void DecodeKeys(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch *) this;

		// Update composite-counter
		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {

			const int PADDED_BYTES_PER_LANE 	= KernelPolicy::Grid::ROWS_PER_LANE * KernelPolicy::Grid::PADDED_PARTIALS_PER_ROW * 4;
			const int LOAD_OFFSET_BYTES 		= LOAD * KernelPolicy::SCAN_LANES_PER_LOAD * PADDED_BYTES_PER_LANE;
			const KeyType COUNTER_BYTE_MASK 	= (KernelPolicy::LOG_BINS < 2) ? 0x1 : 0x3;

			// Decode the bin for this key
			key_bins[CYCLE][LOAD][VEC] = dispatch->DecodeBin(keys[CYCLE][LOAD][VEC], cta);

			// Decode composite-counter lane and sub-counter from bin
			int lane = key_bins[CYCLE][LOAD][VEC] >> 2;										// extract composite counter lane
			int sub_counter = key_bins[CYCLE][LOAD][VEC] & COUNTER_BYTE_MASK;				// extract 8-bit counter offset

			// Compute partial (because we overwrite, we need to accommodate all previous
			// vec-elements if they have the same bin)
			int partial = 1 + SameBinCount::template Iterate<CYCLE, LOAD, VEC>::Invoke(
				dispatch,
				key_bins[CYCLE][LOAD][VEC]);

			// Counter offset in bytes from this thread's "base_composite_counter" location
			counter_offsets[LOAD][VEC] =
				LOAD_OFFSET_BYTES +
				util::FastMul(lane, PADDED_BYTES_PER_LANE) +
				sub_counter;

			// Overwrite partial
			unsigned char *base_partial_chars = (unsigned char *) cta->base_composite_counter;
			base_partial_chars[counter_offsets[LOAD][VEC]] = partial;

		} else {

			key_bins[CYCLE][LOAD][VEC] = INVALID_BIN;
		}
	}


	/**
	 * ExtractRanks
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void ExtractRanks(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch *) this;

		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {

			unsigned char *base_partial_chars = (unsigned char *) cta->base_composite_counter;

			local_ranks[CYCLE][LOAD][VEC] = base_partial_chars[counter_offsets[LOAD][VEC]] +
				SameBinCount::template Iterate<CYCLE, LOAD, VEC>::Invoke(
					dispatch,
					key_bins[CYCLE][LOAD][VEC]);
		} else {

			// Put invalid keys just after the end of the valid swap exchange.
			local_ranks[CYCLE][LOAD][VEC] = KernelPolicy::TILE_ELEMENTS;
		}
	}


	/**
	 * UpdateRanks
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void UpdateRanks(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch *) this;

		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {
			// Update this key's rank with the bin-prefix for it's bin
			local_ranks[CYCLE][LOAD][VEC] +=
				cta->smem_storage.bin_prefixes[CYCLE][LOAD][key_bins[CYCLE][LOAD][VEC]];
		}
	}


	/**
	 * UpdateGlobalOffsets
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch *) this;

		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {
			// Update this key's global scatter offset with its
			// cycle rank and with the bin-prefix for it's bin
			scatter_offsets[CYCLE][LOAD][VEC] =
				local_ranks[CYCLE][LOAD][VEC] +
				cta->smem_storage.bin_prefixes[CYCLE][LOAD][key_bins[CYCLE][LOAD][VEC]];
		}
	}


	/**
	 * ResetLanes
	 */
	template <int LANE, typename Cta>
	__device__ __forceinline__ void ResetLanes(Cta *cta)
	{
		cta->base_composite_counter[LANE][0] = 0;
	}


	//---------------------------------------------------------------------
	// IterateCycleLanes Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next lane
	 */
	template <int LANE, int dummy = 0>
	struct IterateCycleLanes
	{
		// ResetLanes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ResetLanes(Cta *cta, Tile *tile)
		{
			tile->ResetLanes<LANE>(cta);
			IterateCycleLanes<LANE + 1>::ResetLanes(cta, tile);
		}
	};

	/**
	 * Terminate lane iteration
	 */
	template <int dummy>
	struct IterateCycleLanes<SCAN_LANES_PER_CYCLE, dummy>
	{
		// ResetLanes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ResetLanes(Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// IterateCycleElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int CYCLE, int LOAD, int VEC, int dummy = 0>
	struct IterateCycleElements
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile)
		{
			tile->DecodeKeys<CYCLE, LOAD, VEC>(cta);
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::DecodeKeys(cta, tile);
		}

		// ExtractRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ExtractRanks(Cta *cta, Tile *tile)
		{
			tile->ExtractRanks<CYCLE, LOAD, VEC>(cta);
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::ExtractRanks(cta, tile);
		}

		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile)
		{
			tile->UpdateRanks<CYCLE, LOAD, VEC>(cta);
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::UpdateRanks(cta, tile);
		}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile)
		{
			tile->UpdateGlobalOffsets<CYCLE, LOAD, VEC>(cta);
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::UpdateGlobalOffsets(cta, tile);
		}
	};


	/**
	 * IterateCycleElements next load
	 */
	template <int CYCLE, int LOAD, int dummy>
	struct IterateCycleElements<CYCLE, LOAD, LOAD_VEC_SIZE, dummy>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, LOAD + 1, 0>::DecodeKeys(cta, tile);
		}

		// ExtractRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ExtractRanks(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, LOAD + 1, 0>::ExtractRanks(cta, tile);
		}

		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, LOAD + 1, 0>::UpdateRanks(cta, tile);
		}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, LOAD + 1, 0>::UpdateGlobalOffsets(cta, tile);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int CYCLE, int dummy>
	struct IterateCycleElements<CYCLE, LOADS_PER_CYCLE, 0, dummy>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile) {}

		// ExtractRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ExtractRanks(Cta *cta, Tile *tile) {}

		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile) {}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile) {}
	};



	//---------------------------------------------------------------------
	// Tile Internal Methods
	//---------------------------------------------------------------------

	/**
	 * Scan Cycle
	 */
	template <int CYCLE, typename Cta>
	__device__ __forceinline__ void ScanCycle(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Reset smem composite counters
		IterateCycleLanes<0>::ResetLanes(cta, dispatch);

		// Decode bins and update 8-bit composite counters for the keys in this cycle
		IterateCycleElements<CYCLE, 0, 0>::DecodeKeys(cta, dispatch);

		__syncthreads();

		// Use our raking threads to, in aggregate, scan the composite counter lanes
		if (threadIdx.x < KernelPolicy::Grid::RAKING_THREADS) {

			// Upsweep rake
			int partial = util::reduction::SerialReduce<KernelPolicy::Grid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment);

			int warpscan_lane 		= threadIdx.x >> KernelPolicy::Grid::LOG_RAKING_THREADS_PER_LANE;
			int warpscan_tid 		= threadIdx.x & (KernelPolicy::Grid::RAKING_THREADS_PER_LANE - 1);

			// Inclusive warpscan in bin warpscan_lane
			int inclusive_prefix 	= util::scan::WarpScan<KernelPolicy::Grid::LOG_RAKING_THREADS_PER_LANE, false>::Invoke(
				partial,
				cta->smem_storage.lanes_warpscan[warpscan_lane],
				warpscan_tid);
			int exclusive_prefix 	= inclusive_prefix - partial;

			// Save off each lane's warpscan total for this cycle
			if (warpscan_tid == KernelPolicy::Grid::RAKING_THREADS_PER_LANE - 1) {
				cta->smem_storage.lane_totals[CYCLE][warpscan_lane][0] = exclusive_prefix;
				cta->smem_storage.lane_totals[CYCLE][warpscan_lane][1] = partial;
			}

			// Downsweep rake
			util::scan::SerialScan<KernelPolicy::Grid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment,
				exclusive_prefix);
		}

		__syncthreads();

		// Extract the local ranks of each key
		IterateCycleElements<CYCLE, 0, 0>::ExtractRanks(cta, dispatch);
	}


	/**
	 * RecoverBinCounts
	 *
	 * Called by threads [0, KernelPolicy::BINS)
	 */
	template <int CYCLE, int LOAD, typename Cta>
	__device__ __forceinline__ void RecoverBinCounts(
		int my_base_lane, int my_quad_byte, Cta *cta)
	{
		bin_counts[CYCLE][LOAD] =
			cta->smem_storage.lane_totals_c[CYCLE][LOAD][my_base_lane][0][my_quad_byte] +
			cta->smem_storage.lane_totals_c[CYCLE][LOAD][my_base_lane][1][my_quad_byte];
	}


	/**
	 * UpdateBinPrefixes
	 *
	 * Called by threads [0, KernelPolicy::BINS)
	 */
	template <int CYCLE, int LOAD, typename Cta>
	__device__ __forceinline__ void UpdateBinPrefixes(int bin_prefix, Cta *cta)
	{
		cta->smem_storage.bin_prefixes[CYCLE][LOAD][threadIdx.x] = bin_counts[CYCLE][LOAD] + bin_prefix;
	}


	/**
	 * DecodeGlobalOffsets
	 */
	template <int ELEMENT, typename Cta>
	__device__ __forceinline__ void DecodeGlobalOffsets(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch*) this;

		KeyType *linear_keys 	= (KeyType *) keys;
		SizeT *linear_offsets 	= (SizeT *) scatter_offsets;

		int bin = dispatch->DecodeBin(linear_keys[ELEMENT], cta);

		linear_offsets[ELEMENT] =
			cta->smem_storage.bin_carry[bin] +
			(KernelPolicy::THREADS * ELEMENT) + threadIdx.x;
	}


	//---------------------------------------------------------------------
	// IterateCycles Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next cycle
	 */
	template <int CYCLE, int dummy = 0>
	struct IterateCycles
	{
		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, 0, 0>::UpdateRanks(cta, tile);
			IterateCycles<CYCLE + 1>::UpdateRanks(cta, tile);
		}

		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, 0, 0>::UpdateGlobalOffsets(cta, tile);
			IterateCycles<CYCLE + 1>::UpdateGlobalOffsets(cta, tile);
		}

		// ScanCycles
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScanCycles(Cta *cta, Tile *tile)
		{
			tile->ScanCycle<CYCLE>(cta);
			IterateCycles<CYCLE + 1>::ScanCycles(cta, tile);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateCycles<CYCLES_PER_TILE, dummy>
	{
		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile) {}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile) {}

		// ScanCycles
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScanCycles(Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// IterateCycleLoads Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next load
	 */
	template <int CYCLE, int LOAD, int dummy = 0>
	struct IterateCycleLoads
	{
		// RecoverBinCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverBinCounts(
			int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile)
		{
			tile->template RecoverBinCounts<CYCLE, LOAD>(my_base_lane, my_quad_byte, cta);
			IterateCycleLoads<CYCLE, LOAD + 1>::RecoverBinCounts(my_base_lane, my_quad_byte, cta, tile);
		}

		// UpdateBinPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateBinPrefixes(
			int bin_prefix, Cta *cta, Tile *tile)
		{
			tile->template UpdateBinPrefixes<CYCLE, LOAD>(bin_prefix, cta);
			IterateCycleLoads<CYCLE, LOAD + 1>::UpdateBinPrefixes(bin_prefix, cta, tile);
		}
	};


	/**
	 * Iterate next cycle
	 */
	template <int CYCLE, int dummy>
	struct IterateCycleLoads<CYCLE, LOADS_PER_CYCLE, dummy>
	{
		// RecoverBinCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverBinCounts(
			int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile)
		{
			IterateCycleLoads<CYCLE + 1, 0>::RecoverBinCounts(my_base_lane, my_quad_byte, cta, tile);
		}

		// UpdateBinPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateBinPrefixes(
			int bin_prefix, Cta *cta, Tile *tile)
		{
			IterateCycleLoads<CYCLE + 1, 0>::UpdateBinPrefixes(bin_prefix, cta, tile);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateCycleLoads<CYCLES_PER_TILE, 0, dummy>
	{
		// RecoverBinCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverBinCounts(int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile) {}

		// UpdateBinPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateBinPrefixes(int bin_prefix, Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// IterateElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next tile element
	 */
	template <int ELEMENT, int dummy = 0>
	struct IterateElements
	{
		// DecodeGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeGlobalOffsets(Cta *cta, Tile *tile)
		{
			tile->DecodeGlobalOffsets<ELEMENT>(cta);
			IterateElements<ELEMENT + 1>::DecodeGlobalOffsets(cta, tile);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateElements<TILE_ELEMENTS_PER_THREAD, dummy>
	{
		// DecodeGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeGlobalOffsets(Cta *cta, Tile *tile) {}
	};



	//---------------------------------------------------------------------
	// Partition/scattering specializations
	//---------------------------------------------------------------------


	template <
		ScatterStrategy SCATTER_STRATEGY,
		int dummy = 0>
	struct PartitionTile;



	/**
	 * Specialized for two-phase scatter, keys-only
	 */
	template <
		ScatterStrategy SCATTER_STRATEGY,
		int dummy>
	struct PartitionTile
	{
		enum {
			MEM_BANKS 					= 1 << B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__),
			DIGITS_PER_SCATTER_PASS 	= KernelPolicy::WARPS * (B40C_WARP_THREADS(__B40C_CUDA_ARCH__) / (MEM_BANKS)),
			SCATTER_PASSES 				= KernelPolicy::BINS / DIGITS_PER_SCATTER_PASS,
		};

		template <typename T>
		static __device__ __forceinline__ void Nop(T &t) {}

		/**
		 * Warp based scattering that does not cross alignment boundaries, e.g., for SM1.0-1.1
		 * coalescing rules
		 */
		template <int PASS, int SCATTER_PASSES>
		struct WarpScatter
		{
			template <typename T, void Transform(T&), typename Cta>
			static __device__ __forceinline__ void ScatterPass(
				Cta *cta,
				T *exchange,
				T *d_out,
				const SizeT &valid_elements)
			{
				const int LOG_STORE_TXN_THREADS = B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__);
				const int STORE_TXN_THREADS = 1 << LOG_STORE_TXN_THREADS;

				int store_txn_idx = threadIdx.x & (STORE_TXN_THREADS - 1);
				int store_txn_digit = threadIdx.x >> LOG_STORE_TXN_THREADS;

				int my_digit = (PASS * DIGITS_PER_SCATTER_PASS) + store_txn_digit;

				if (my_digit < KernelPolicy::BINS) {

					int my_exclusive_scan = cta->smem_storage.bin_warpscan[1][my_digit - 1];
					int my_inclusive_scan = cta->smem_storage.bin_warpscan[1][my_digit];
					int my_digit_count = my_inclusive_scan - my_exclusive_scan;

					int my_carry = cta->smem_storage.bin_carry[my_digit] + my_exclusive_scan;
					int my_aligned_offset = store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));

					while (my_aligned_offset < my_digit_count) {

						if ((my_aligned_offset >= 0) && (my_exclusive_scan + my_aligned_offset < valid_elements)) {

							T datum = exchange[my_exclusive_scan + my_aligned_offset];
							Transform(datum);
							d_out[my_carry + my_aligned_offset] = datum;
						}
						my_aligned_offset += STORE_TXN_THREADS;
					}
				}

				WarpScatter<PASS + 1, SCATTER_PASSES>::template ScatterPass<T, Transform>(
					cta,
					exchange,
					d_out,
					valid_elements);
			}
		};

		// Terminate
		template <int SCATTER_PASSES>
		struct WarpScatter<SCATTER_PASSES, SCATTER_PASSES>
		{
			template <typename T, void Transform(T&), typename Cta>
			static __device__ __forceinline__ void ScatterPass(
				Cta *cta,
				T *exchange,
				T *d_out,
				const SizeT &valid_elements) {}
		};

		template <bool KEYS_ONLY, int dummy2 = 0>
		struct ScatterValues
		{
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Invoke(
				SizeT cta_offset,
				const SizeT &guarded_elements,
				const SizeT &valid_elements,
				Cta *cta,
				Tile *tile)
			{
				// Load values
				tile->LoadValues(cta, cta_offset, guarded_elements);

				// Scatter values to smem by local rank
				util::io::ScatterTile<
					KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
					0,
					KernelPolicy::THREADS,
					util::io::st::NONE>::Scatter(
						cta->smem_storage.value_exchange,
						(ValueType (*)[1]) tile->values,
						(int (*)[1]) tile->local_ranks);

				__syncthreads();

				if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE) {

					WarpScatter<0, SCATTER_PASSES>::template ScatterPass<ValueType, Nop<ValueType> >(
						cta,
						cta->smem_storage.value_exchange,
						cta->d_out_values,
						valid_elements);

					__syncthreads();

				} else {

					// Gather values linearly from smem (vec-1)
					util::io::LoadTile<
						KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
						0,
						KernelPolicy::THREADS,
						util::io::ld::NONE,
						false>::LoadValid(									// No need to check alignment
							(ValueType (*)[1]) tile->values,
							cta->smem_storage.value_exchange,
							0);

					__syncthreads();

					// Scatter values to global bin partitions
					tile->ScatterValues(cta, valid_elements);
				}
			}
		};

		template <int dummy2>
		struct ScatterValues<true, dummy2>
		{
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Invoke(
					SizeT cta_offset,
					const SizeT &guarded_elements,
					const SizeT &valid_elements,
					Cta *cta,
					Tile *tile) {}
		};

		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(
			SizeT cta_offset,
			const SizeT &guarded_elements,
			Cta *cta,
			Tile *tile)
		{
			// Load keys
			tile->LoadKeys(cta, cta_offset, guarded_elements);

			// Scan cycles
			IterateCycles<0>::ScanCycles(cta, tile);

			// Scan across bins
			if (threadIdx.x < KernelPolicy::BINS) {

				// Recover bin-counts from lane totals
				int my_base_lane = threadIdx.x >> 2;
				int my_quad_byte = threadIdx.x & 3;
				IterateCycleLoads<0, 0>::RecoverBinCounts(
					my_base_lane, my_quad_byte, cta, tile);

				// Scan across my bin counts for each load
				int tile_bin_total = util::scan::SerialScan<KernelPolicy::LOADS_PER_TILE>::Invoke(
					(int *) tile->bin_counts, 0);

				// Add the previous tile's inclusive-scan to the running bin-carry
				SizeT my_carry = cta->smem_storage.bin_carry[threadIdx.x] +
					cta->smem_storage.bin_warpscan[1][threadIdx.x];

				// Perform overflow-free inclusive SIMD Kogge-Stone across bins
				int tile_bin_inclusive = util::scan::WarpScan<KernelPolicy::LOG_BINS, false>::Invoke(
					tile_bin_total,
					cta->smem_storage.bin_warpscan);

				// Save inclusive scan in bin_warpscan
				cta->smem_storage.bin_warpscan[1][threadIdx.x] = tile_bin_inclusive;

				// Calculate exclusive scan
				int tile_bin_exclusive = tile_bin_inclusive - tile_bin_total;

				// Subtract the bin prefix from the running carry (to offset threadIdx during scatter)
				cta->smem_storage.bin_carry[threadIdx.x] = my_carry - tile_bin_exclusive;

				// Compute the bin prefixes for this tile for each load
				IterateCycleLoads<0, 0>::UpdateBinPrefixes(tile_bin_exclusive, cta, tile);
			}

			__syncthreads();

			// Update the local ranks in each load with the bin prefixes for the tile
			IterateCycles<0>::UpdateRanks(cta, tile);

			// Scatter keys to smem by local rank
			util::io::ScatterTile<
				KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
				0,
				KernelPolicy::THREADS,
				util::io::st::NONE>::Scatter(
					cta->smem_storage.key_exchange,
					(KeyType (*)[1]) tile->keys,
					(int (*)[1]) tile->local_ranks);

			__syncthreads();

			SizeT valid_elements = tile->ValidElements(cta, guarded_elements);

			if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE) {

				WarpScatter<0, SCATTER_PASSES>::template ScatterPass<KeyType, KernelPolicy::PostprocessKey>(
					cta,
					cta->smem_storage.key_exchange,
					cta->d_out_keys,
					valid_elements);

				__syncthreads();

			} else {

				// Gather keys linearly from smem (vec-1)
				util::io::LoadTile<
					KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
					0,
					KernelPolicy::THREADS,
					util::io::ld::NONE,
					false>::LoadValid(									// No need to check alignment
						(KeyType (*)[1]) tile->keys,
						cta->smem_storage.key_exchange,
						0);

				__syncthreads();

				// Compute global scatter offsets for gathered keys
				IterateElements<0>::DecodeGlobalOffsets(cta, tile);

				// Scatter keys to global bin partitions
				tile->ScatterKeys(cta, valid_elements);

			}

			// Partition values
			ScatterValues<KernelPolicy::KEYS_ONLY>::Invoke(
				cta_offset, guarded_elements, valid_elements, cta, tile);
		}
	};


	/**
	 * Specialized for direct scatter
	 */
	template <int dummy>
	struct PartitionTile<SCATTER_DIRECT, dummy>
	{
		template <bool KEYS_ONLY, int dummy2 = 0>
		struct ScatterValues
		{
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Invoke(
				SizeT cta_offset,
				const SizeT &guarded_elements,
				const SizeT &valid_elements,
				Cta *cta,
				Tile *tile)
			{
				// Load values
				tile->LoadValues(cta, cta_offset, guarded_elements);

				// Scatter values to global bin partitions
				tile->ScatterValues(cta, valid_elements);
			}
		};

		template <int dummy2>
		struct ScatterValues<true, dummy2>
		{
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Invoke(
					SizeT cta_offset,
					const SizeT &guarded_elements,
					const SizeT &valid_elements,
					Cta *cta,
					Tile *tile) {}
		};

		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(
			SizeT cta_offset,
			const SizeT &guarded_elements,
			Cta *cta,
			Tile *tile)
		{
			// Load keys
			tile->LoadKeys(cta, cta_offset, guarded_elements);

			// Scan cycles
			IterateCycles<0>::ScanCycles(cta, tile);

			// Scan across bins
			if (threadIdx.x < KernelPolicy::BINS) {

				// Recover bin-counts from lane totals
				int my_base_lane = threadIdx.x >> 2;
				int my_quad_byte = threadIdx.x & 3;
				IterateCycleLoads<0, 0>::RecoverBinCounts(
					my_base_lane, my_quad_byte, cta, tile);

				// Scan across my bin counts for each load
				int tile_bin_total = util::scan::SerialScan<KernelPolicy::LOADS_PER_TILE>::Invoke(
					(int *) tile->bin_counts, 0);

				// Add the previous tile's inclusive-scan to the running bin-carry
				SizeT my_carry = cta->smem_storage.bin_carry[threadIdx.x];

				// Update bin prefixes with the incoming carry
				IterateCycleLoads<0, 0>::UpdateBinPrefixes(my_carry, cta, tile);

				// Update carry
				cta->smem_storage.bin_carry[threadIdx.x] = my_carry + tile_bin_total;
			}

			__syncthreads();

			SizeT valid_elements = tile->ValidElements(cta, guarded_elements);

			// Update the scatter offsets in each load with the bin prefixes for the tile
			IterateCycles<0>::UpdateGlobalOffsets(cta, tile);

			// Scatter keys to global bin partitions
			tile->ScatterKeys(cta, valid_elements);

			// Partition values
			ScatterValues<KernelPolicy::KEYS_ONLY>::Invoke(
				cta_offset, guarded_elements, valid_elements, cta, tile);
		}
	};



	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Loads, decodes, and scatters a tile into global partitions
	 */
	template <typename Cta>
	__device__ __forceinline__ void Partition(
		SizeT cta_offset,
		const SizeT &guarded_elements,
		Cta *cta)
	{
		PartitionTile<KernelPolicy::SCATTER_STRATEGY>::Invoke(
			cta_offset,
			guarded_elements,
			cta,
			(Dispatch *) this);
	}

};


} // namespace downsweep
} // namespace partition
} // namespace b40c

