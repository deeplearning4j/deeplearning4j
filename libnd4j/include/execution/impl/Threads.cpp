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
 // @author raver119@gmail.com
 //
#include <execution/Threads.h>
#include <execution/ThreadPool.h>
#include <vector>
#include <thread>
#include <helpers/logger.h>
#include <math/templatemath.h>
#include <helpers/shape.h>

#ifdef _OPENMP

#include <omp.h>

#endif


namespace samediff {

	int ThreadsHelper::numberOfThreads(int maxThreads, uint64_t numberOfElements) {
		// let's see how many threads we actually need first
		auto optimalThreads = sd::math::nd4j_max<uint64_t>(1, numberOfElements / 1024);

		// now return the smallest value
		return sd::math::nd4j_min<int>(optimalThreads, maxThreads);
	}

	Span3::Span3(int64_t startX, int64_t stopX, int64_t incX, int64_t startY, int64_t stopY, int64_t incY, int64_t startZ, int64_t stopZ, int64_t incZ) {
		_startX = startX;
		_startY = startY;
		_startZ = startZ;
		_stopX = stopX;
		_stopY = stopY;
		_stopZ = stopZ;
		_incX = incX;
		_incY = incY;
		_incZ = incZ;
	}

	Span3 Span3::build(int loop, uint64_t threadID, uint64_t numThreads, int64_t startX, int64_t stopX, int64_t incX, int64_t startY, int64_t stopY, int64_t incY, int64_t startZ, int64_t stopZ, int64_t incZ) {
		switch (loop) {
		case 1: {
			auto span = (stopX - startX) / numThreads;
			auto s = span * threadID;
			auto e = s + span;
			if (threadID == numThreads - 1)
				e = stopX;

			return Span3(s, e, incX, startY, stopY, incY, startZ, stopZ, incZ);
		}
			  break;
		case 2: {
			auto span = (stopY - startY) / numThreads;
			auto s = span * threadID;
			auto e = s + span;
			if (threadID == numThreads - 1)
				e = stopY;

			return Span3(startX, stopX, incX, s, e, incY, startZ, stopZ, incZ);
		}
			  break;
		case 3: {
			auto span = (stopZ - startZ) / numThreads;
			auto s = span * threadID;
			auto e = s + span;
			if (threadID == numThreads - 1)
				e = stopZ;

			return Span3(startX, stopX, incX, startY, stopY, incY, s, e, incZ);
		}
			  break;
		default:
			throw std::runtime_error("");
		}
		return Span3(startX, stopX, incX, startY, stopY, incY, startZ, stopZ, incZ);
	}

	Span::Span(int64_t startX, int64_t stopX, int64_t incX) {
		_startX = startX;
		_stopX = stopX;
		_incX = incX;
	}

	Span Span::build(uint64_t threadID, uint64_t numThreads, int64_t startX, int64_t stopX, int64_t incX) {
		auto span = (stopX - startX) / numThreads;
		auto s = span * threadID;
		auto e = s + span;
		if (threadID == numThreads - 1)
			e = stopX;

		return Span(s, e, incX);
	}

	Span2::Span2(int64_t startX, int64_t stopX, int64_t incX, int64_t startY, int64_t stopY, int64_t incY) {
		_startX = startX;
		_startY = startY;
		_stopX = stopX;
		_stopY = stopY;
		_incX = incX;
		_incY = incY;
	}


	Span2 Span2::build(int loop, uint64_t threadID, uint64_t numThreads, int64_t startX, int64_t stopX, int64_t incX, int64_t startY, int64_t stopY, int64_t incY) {

		switch (loop) {
		case 1: {
			auto span = (stopX - startX) / numThreads;
			auto s = span * threadID;
			auto e = s + span;
			if (threadID == numThreads - 1)
				e = stopX;

			return Span2(s, e, incX, startY, stopY, incY);
		}
			  break;
		case 2: {
			auto span = (stopY - startY) / numThreads;
			auto s = span * threadID;
			auto e = s + span;
			if (threadID == numThreads - 1)
				e = stopY;

			return Span2(startX, stopX, incX, s, e, incY);
		}
			  break;
		default:
			throw std::runtime_error("");
		}
	}

	int64_t Span::startX() const {
		return _startX;
	}

	int64_t Span::stopX() const {
		return _stopX;
	}

	int64_t Span::incX() const {
		return _incX;
	}

	int64_t Span2::startX() const {
		return _startX;
	}

	int64_t Span2::startY() const {
		return _startY;
	}

	int64_t Span2::stopX() const {
		return _stopX;
	}

	int64_t Span2::stopY() const {
		return _stopY;
	}

	int64_t Span2::incX() const {
		return _incX;
	}

	int64_t Span2::incY() const {
		return _incY;
	}

	int64_t Span3::startX() const {
		return _startX;
	}

	int64_t Span3::startY() const {
		return _startY;
	}

	int64_t Span3::startZ() const {
		return _startZ;
	}

	int64_t Span3::stopX() const {
		return _stopX;
	}

	int64_t Span3::stopY() const {
		return _stopY;
	}

	int64_t Span3::stopZ() const {
		return _stopZ;
	}

	int64_t Span3::incX() const {
		return _incX;
	}

	int64_t Span3::incY() const {
		return _incY;
	}

	int64_t Span3::incZ() const {
		return _incZ;
	}

	int ThreadsHelper::pickLoop2d(int numThreads, uint64_t itersX, uint64_t itersY) {
		// if one of dimensions is definitely too small - we just pick the other one
		if (itersX < numThreads && itersY >= numThreads)
			return 2;
		if (itersY < numThreads && itersX >= numThreads)
			return 1;

		// next step - we pick the most balanced dimension
		auto remX = itersX % numThreads;
		auto remY = itersY % numThreads;
		auto splitY = itersY / numThreads;

		// if there's no remainder left in some dimension - we're picking that dimension, because it'll be the most balanced work distribution
		if (remX == 0)
			return 1;
		if (remY == 0)
			return 2;

		// if there's no loop without a remainder - we're picking one with smaller remainder
		if (remX < remY)
			return 1;
		if (remY < remX && splitY >= 64) // we don't want too small splits over last dimension, or vectorization will fail
			return 2;
		// if loops are equally sized - give the preference to the first thread
		return 1;
	}


	static int threads_(int maxThreads, uint64_t elements) {

		if (elements == maxThreads) {
			return maxThreads;
		}
		else if (elements > maxThreads) {
			// if we have full load across thread, or at least half of threads can be utilized
			auto rem = elements % maxThreads;
			if (rem == 0 || rem >= maxThreads / 3)
				return maxThreads;
			else
				return threads_(maxThreads - 1, elements);

		}
		else if (elements < maxThreads) {
			return elements;
		}

		return 1;
	}

	int ThreadsHelper::numberOfThreads2d(int maxThreads, uint64_t iters_x, uint64_t iters_y) {
		// in some cases there's nothing to think about, part 1
		if (iters_x < maxThreads && iters_y < maxThreads)
			return sd::math::nd4j_max<int>(iters_x, iters_y);

		auto remX = iters_x % maxThreads;
		auto remY = iters_y % maxThreads;

		// in some cases there's nothing to think about, part 2
		if ((iters_x >= maxThreads && remX == 0) || (iters_y >= maxThreads && remY == 0))
			return maxThreads;

		// at this point we suppose that there's no loop perfectly matches number of our threads
		// so let's pick something as equal as possible
		if (iters_x > maxThreads || iters_y > maxThreads)
			return maxThreads;
		else
			return numberOfThreads2d(maxThreads - 1, iters_x, iters_y);
	}

	int ThreadsHelper::numberOfThreads3d(int maxThreads, uint64_t itersX, uint64_t itersY, uint64_t itersZ) {
		// we don't want to run underloaded threads
		if (itersX * itersY * itersZ <= 32)
			return 1;

		auto remX = itersX % maxThreads;
		auto remY = itersY % maxThreads;
		auto remZ = itersZ % maxThreads;

		// if we have perfect balance across one of dimensions - just go for it
		if ((itersX >= maxThreads && remX == 0) || (itersY >= maxThreads && remY == 0) || (itersZ >= maxThreads && remZ == 0))
			return maxThreads;

		int threadsX = 0, threadsY = 0, threadsZ = 0;

		// now we look into possible number of
		threadsX = threads_(maxThreads, itersX);
		threadsY = threads_(maxThreads, itersY);
		threadsZ = threads_(maxThreads, itersZ);

		// we want to split as close to outer loop as possible, so checking it out first
		if (threadsX >= threadsY && threadsX >= threadsZ)
			return threadsX;
		else if (threadsY >= threadsX && threadsY >= threadsZ)
			return threadsY;
		else if (threadsZ >= threadsX && threadsZ >= threadsY)
			return threadsZ;

		return 1;
	}

	int ThreadsHelper::pickLoop3d(int numThreads, uint64_t itersX, uint64_t itersY, uint64_t itersZ) {
		auto remX = itersX % numThreads;
		auto remY = itersY % numThreads;
		auto remZ = itersZ % numThreads;

		auto splitX = itersX / numThreads;
		auto splitY = itersY / numThreads;
		auto splitZ = itersZ / numThreads;

		// if there's no remainder left in some dimension - we're picking that dimension, because it'll be the most balanced work distribution
		if (remX == 0)
			return 1;
		else if (remY == 0)
			return 2;
		else if (remZ == 0) // TODO: we don't want too smal splits over last dimension? or we do?
			return 3;

		if (itersX > numThreads)
			return 1;
		else if (itersY > numThreads)
			return 2;
		else if (itersZ > numThreads)
			return 3;

		return 1;
	}

#ifdef _OPENMP

    std::mutex Threads::gThreadmutex;
	uint64_t Threads::_nFreeThreads = sd::Environment::getInstance()->maxThreads();

     bool   Threads::tryAcquire(int numThreads){
		 std::lock_guard<std::mutex> lock( gThreadmutex );
		 auto nThreads = _nFreeThreads - numThreads;
		 if(nThreads >= 0){
			 _nFreeThreads = nThreads;

			 return true;
		 }
		 return false;
	 }

	 bool  Threads::freeThreads(int numThreads){
		 std::lock_guard<std::mutex> lock( gThreadmutex );
         _nFreeThreads += numThreads;
		 // check if correct number of threads
		 return _nFreeThreads > sd::Environment::getInstance()->maxThreads();
	 }
#endif

	int Threads::parallel_tad(FUNC_1D function, int64_t start, int64_t stop, int64_t increment, uint32_t numThreads) {
		if (start > stop)
			throw std::runtime_error("Threads::parallel_for got start > stop");

		auto delta = (stop - start);

		if (numThreads > delta)
			numThreads = delta;

		if (numThreads == 0)
			return 0;

		// shortcut
		if (numThreads == 1) {
			function(0, start, stop, increment);
			return 1;
		}

#ifdef _OPENMP

		if (tryAcquire(numThreads)) {
			#pragma omp parallel for
		    for (int e = start; e < stop; e++) {
			    function(omp_get_thread_num(), e, e + 1, increment);
		    }
			freeThreads(numThreads);
			return numThreads;
		}
		else {
		    // if there were no threads available - we'll execute function right within current thread
		    function(0, start, stop, increment);

		    // we tell that parallelism request declined
		    return 1;
		}
#else

        sd::Environment::getInstance()->maxThreads();
		auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
		if (ticket != nullptr) {
			// if we got our threads - we'll run our jobs here
			auto span = delta / numThreads;

			for (uint32_t e = 0; e < numThreads; e++) {
				auto start_ = span * e + start;
				auto stop_ = start_ + span;

				// last thread will process tail
				if (e == numThreads - 1)
					stop_ = stop;

				// putting the task into the queue for a given thread
				ticket->enqueue(e, numThreads, function, start_, stop_, increment);
			}

			// block and wait till all threads finished the job
			ticket->waitAndRelease();

			// we tell that parallelism request succeeded
			return numThreads;
		}
		else {
			// if there were no threads available - we'll execute function right within current thread
			function(0, start, stop, increment);

			// we tell that parallelism request declined
			return 1;
		}
#endif
	}

	int Threads::parallel_for(FUNC_1D function, int64_t start, int64_t stop, int64_t increment, uint32_t numThreads) {
		if (start > stop)
			throw std::runtime_error("Threads::parallel_for got start > stop");

		auto delta = (stop - start);

		// in some cases we just fire func as is
		if (delta == 0 || numThreads == 1) {
			function(0, start, stop, increment);
			return 1;
		}

		auto numElements = delta / increment;

		// we decide what's optimal number of threads we need here, and execute it in parallel_tad.
		numThreads = ThreadsHelper::numberOfThreads(numThreads, numElements);
		return parallel_tad(function, start, stop, increment, numThreads);
	}

	int Threads::parallel_for(FUNC_2D function, int64_t startX, int64_t stopX, int64_t incX, int64_t startY, int64_t stopY, int64_t incY, uint64_t numThreads, bool debug) {
		if (startX > stopX)
			throw std::runtime_error("Threads::parallel_for got startX > stopX");

		if (startY > stopY)
			throw std::runtime_error("Threads::parallel_for got startY > stopY");

		// number of elements per loop
		auto delta_x = (stopX - startX);
		auto delta_y = (stopY - startY);

		// number of iterations per loop
		auto itersX = delta_x / incX;
		auto itersY = delta_y / incY;

		// total number of iterations
		auto iters_t = itersX * itersY;

		// we are checking the case of number of requested threads was smaller
		numThreads = ThreadsHelper::numberOfThreads2d(numThreads, itersX, itersY);

		// basic shortcut for no-threading cases
		if (numThreads == 1) {
			function(0, startX, stopX, incX, startY, stopY, incY);
			return 1;
		}

		// We have couple of scenarios:
		// either we split workload along 1st loop, or 2nd
		auto splitLoop = ThreadsHelper::pickLoop2d(numThreads, itersX, itersY);

		// for debug mode we execute things inplace, without any threads
		if (debug) {
			for (int e = 0; e < numThreads; e++) {
				auto span = Span2::build(splitLoop, e, numThreads, startX, stopX, incX, startY, stopY, incY);

				function(e, span.startX(), span.stopX(), span.incX(), span.startY(), span.stopY(), span.incY());
			}

			// but we still mimic multithreaded execution
			return numThreads;
		}
		else {
#ifdef _OPENMP

		if (tryAcquire(numThreads)) {
			#pragma omp parallel for collapse(2)
			for (auto x = startX; x < stopX; x += incX) {
			    for (auto y = startY; y < stopY; y += incY) {
				    function(omp_get_thread_num(), x, x+1, 1, y, y+1, 1);
				}
			}
			freeThreads(numThreads);
			return numThreads;
			}
			else {
     	    // if there were no threads available - we'll execute function right within current thread
		    function(0, startX, stopX, incX, startY, stopY, incY);

		    // we tell that parallelism request declined
		    return 1;
		}

#else

			auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
			if (ticket != nullptr) {

				for (int e = 0; e < numThreads; e++) {
					auto threadId = numThreads - e - 1;
					auto span = Span2::build(splitLoop, threadId, numThreads, startX, stopX, incX, startY, stopY, incY);

					ticket->enqueue(e, numThreads, function, span.startX(), span.stopX(), span.incX(), span.startY(), span.stopY(), span.incY());
				}

				// block until all threads finish their job
				ticket->waitAndRelease();

				return numThreads;
			}
			else {
				// if there were no threads available - we'll execute function right within current thread
				function(0, startX, stopX, incX, startY, stopY, incY);

				// we tell that parallelism request declined
				return 1;
			}
#endif
		};
	}


	int Threads::parallel_for(FUNC_3D function, int64_t startX, int64_t stopX, int64_t incX, int64_t startY, int64_t stopY, int64_t incY, int64_t startZ, int64_t stopZ, int64_t incZ, uint64_t numThreads) {
		if (startX > stopX)
			throw std::runtime_error("Threads::parallel_for got startX > stopX");

		if (startY > stopY)
			throw std::runtime_error("Threads::parallel_for got startY > stopY");

		if (startZ > stopZ)
			throw std::runtime_error("Threads::parallel_for got startZ > stopZ");

		auto delta_x = stopX - startX;
		auto delta_y = stopY - startY;
		auto delta_z = stopZ - startZ;

		auto itersX = delta_x / incX;
		auto itersY = delta_y / incY;
		auto itersZ = delta_z / incZ;

		numThreads = ThreadsHelper::numberOfThreads3d(numThreads, itersX, itersY, itersZ);
		if (numThreads == 1) {
			// loop is too small - executing function as is
			function(0, startX, stopX, incX, startY, stopY, incY, startZ, stopZ, incZ);
			return 1;
		}

#ifdef _OPENMP

		if (tryAcquire(numThreads)) {
            #pragma omp parallel for collapse(3)
		    for (auto x = startX; x < stopX; x += incX) {
		        for (auto y = startY; y < stopY; y += incY) {
		            for (auto z = startZ; z < stopZ; z += incZ) {
				        function(omp_get_thread_num(), x, x+1, 1, y, y+1, 1, z, z+1, 1);
				    }
				}
			}

			freeThreads(numThreads);
			return numThreads;
		}
		else {
		    // if there were no threads available - we'll execute function right within current thread
	        function(0, startX, stopX, incX, startY, stopY, incY, startZ, stopZ, incZ);

		    // we tell that parallelism request declined
		    return 1;
		}
#else

		auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
		if (ticket != nullptr) {
			auto splitLoop = ThreadsHelper::pickLoop3d(numThreads, itersX, itersY, itersZ);

			for (int e = 0; e < numThreads; e++) {
				auto thread_id = numThreads - e - 1;
				auto span = Span3::build(splitLoop, thread_id, numThreads, startX, stopX, incX, startY, stopY, incY, startZ, stopZ, incZ);

				ticket->enqueue(e, numThreads, function, span.startX(), span.stopX(), span.incX(), span.startY(), span.stopY(), span.incY(), span.startZ(), span.stopZ(), span.incZ());
			}

			// block until we're done
			ticket->waitAndRelease();

			// we tell that parallelism request succeeded
			return numThreads;
		}
		else {
			// if there were no threads available - we'll execute function right within current thread
			function(0, startX, stopX, incX, startY, stopY, incY, startZ, stopZ, incZ);

			// we tell that parallelism request declined
			return 1;
		}
#endif
	}

	int Threads::parallel_do(FUNC_DO function, uint64_t numThreads) {

		if (numThreads == 1) {
			function(0, numThreads);
			return 1;
		}

#ifdef _OPENMP

		if (tryAcquire(numThreads)) {
			#pragma omp parallel for
		    for (int e = 0; e < numThreads; e++) {
			    function(e, numThreads);
		    }

			freeThreads(numThreads);
			return numThreads;
		}
		else {
		    // if there's no threads available - we'll execute function sequentially one by one
		    for (uint64_t e = 0; e < numThreads; e++)
			    function(e, numThreads);

		    return numThreads;
		}
#else
		auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads - 1);
		if (ticket != nullptr) {

			// submit tasks one by one
			for (uint64_t e = 0; e < numThreads - 1; e++)
				ticket->enqueue(e, numThreads, function);

			function(numThreads - 1, numThreads);

			ticket->waitAndRelease();

			return numThreads;
		}
		else {
			// if there's no threads available - we'll execute function sequentially one by one
			for (uint64_t e = 0; e < numThreads; e++)
				function(e, numThreads);

			return numThreads;
		}
#endif

		return numThreads;
	}

	int64_t Threads::parallel_long(FUNC_RL function, FUNC_AL aggregator, int64_t start, int64_t stop, int64_t increment, uint64_t numThreads) {
		if (start > stop)
			throw std::runtime_error("Threads::parallel_long got start > stop");

		auto delta = (stop - start);
		if (delta == 0 || numThreads == 1)
			return function(0, start, stop, increment);

		auto numElements = delta / increment;

		// we decide what's optimal number of threads we need here, and execute it
		numThreads = ThreadsHelper::numberOfThreads(numThreads, numElements);
		if (numThreads == 1)
			return function(0, start, stop, increment);

		// create temporary array
		int64_t intermediatery[256];
		auto span = delta / numThreads;

#ifdef _OPENMP
		if (tryAcquire(numThreads)) {
			#pragma omp parallel for num_threads(numThreads)
            for (int e = 0; e < numThreads; e++) {
			    auto start_ = span * e + start;
			    auto stop_ = span * (e + 1) + start;

			    intermediatery[e] = function(e, start_, e == numThreads - 1 ? stop : stop_, increment);
		    }
			freeThreads(numThreads);
		}
        else{
			     	// if there were no thre ads available - we'll execute function right within current thread
            return	function(0, start, stop, increment);
		}
#else
        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads - 1);
        if (ticket == nullptr)
            return function(0, start, stop, increment);

		// execute threads in parallel
		for (uint32_t e = 0; e < numThreads; e++) {
			auto start_ = span * e + start;
			auto stop_ = span * (e + 1) + start;

			if (e == numThreads - 1)
				intermediatery[e] = function(e, start_, stop, increment);
			else
				ticket->enqueue(e, numThreads, &intermediatery[e], function, start_, stop_, increment);
		}

		ticket->waitAndRelease();

#endif

		// aggregate results in single thread
		for (uint64_t e = 1; e < numThreads; e++)
			intermediatery[0] = aggregator(intermediatery[0], intermediatery[e]);

		// return accumulated result
		return intermediatery[0];
	}

	double Threads::parallel_double(FUNC_RD function, FUNC_AD aggregator, int64_t start, int64_t stop, int64_t increment, uint64_t numThreads) {
		if (start > stop)
			throw std::runtime_error("Threads::parallel_long got start > stop");

		auto delta = (stop - start);
		if (delta == 0 || numThreads == 1)
			return function(0, start, stop, increment);

		auto numElements = delta / increment;

		// we decide what's optimal number of threads we need here, and execute it
		numThreads = ThreadsHelper::numberOfThreads(numThreads, numElements);
		if (numThreads == 1)
			return function(0, start, stop, increment);

		// create temporary array
		double intermediatery[256];
		auto span = delta / numThreads;

#ifdef _OPENMP

        if (tryAcquire(numThreads)) {
			#pragma omp parallel for num_threads(numThreads)
        	for (int e = 0; e < numThreads; e++) {
				auto start_ = span * e + start;
				auto stop_ = span * (e + 1) + start;

				intermediatery[e] = function(e, start_, e == numThreads - 1 ? stop : stop_, increment);
			}
			freeThreads(numThreads);
		}
        else{
			     	// if there were no thre ads available - we'll execute function right within current thread
            return	function(0, start, stop, increment);
		}

#else

        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads - 1);
        if (ticket == nullptr)
            return function(0, start, stop, increment);

		// execute threads in parallel
		for (uint32_t e = 0; e < numThreads; e++) {
			auto start_ = span * e + start;
			auto stop_ = span * (e + 1) + start;

			if (e == numThreads - 1)
				intermediatery[e] = function(e, start_, stop, increment);
			else
				ticket->enqueue(e, numThreads, &intermediatery[e], function, start_, stop_, increment);
		}

		ticket->waitAndRelease();

#endif

		// aggregate results in single thread
		for (uint64_t e = 1; e < numThreads; e++)
			intermediatery[0] = aggregator(intermediatery[0], intermediatery[e]);

		// return accumulated result
		return intermediatery[0];
	}


	int  Threads::parallel_aligned_increment(FUNC_1D function, int64_t start, int64_t stop, int64_t increment, bool adjust, size_t type_size, uint32_t req_numThreads) {
		if (start > stop)
			throw std::runtime_error("Threads::parallel_for got start > stop");
		auto num_elements = (stop - start);
		//this way we preserve increment starts offset
		//so we will parition considering delta but not total elements
		auto delta = (stop - start) / increment;


		// in some cases we just fire func as is
		if (delta == 0 || req_numThreads == 1) {
			function(0, start, stop, increment);
			return 1;
		}
		int numThreads = 0;

		struct th_span {
			Nd4jLong start;
			Nd4jLong end;
		};
#ifdef _OPENMP
		constexpr int max_thread_count = 8;
#else
		constexpr int max_thread_count = 1024;
#endif
		th_span thread_spans[max_thread_count];

		req_numThreads = req_numThreads > max_thread_count ? max_thread_count : req_numThreads;

#ifdef _OPENMP
		int adjusted_numThreads = max_thread_count;
#else
		int adjusted_numThreads = (!adjust) ? req_numThreads : samediff::ThreadsHelper::numberOfThreads(req_numThreads, (num_elements * sizeof(double)) / (200 * type_size));
#endif

		if (adjusted_numThreads > delta)
			adjusted_numThreads = delta;
		// shortcut
		if (adjusted_numThreads <= 1) {
			function(0, start, stop, increment);
			return 1;
		}



		//take span as ceil
		auto spand = std::ceil((double)delta / (double)adjusted_numThreads);
		numThreads = static_cast<int>(std::ceil((double)delta / spand));
		auto span = static_cast<Nd4jLong>(spand);


		//tail_add is additional value of the last part
		//it could be negative or positive
		//we will spread that value across
		auto tail_add = delta - numThreads * span;
		Nd4jLong begin = 0;
		Nd4jLong end = 0;

		//we will try enqueu bigger parts first
		decltype(span) span1, span2;
		int last = 0;
		if (tail_add >= 0) {
			//for span == 1  , tail_add is  0
			last = tail_add;
			span1 = span + 1;
			span2 = span;
		}
		else {
			last = numThreads + tail_add;// -std::abs(tail_add);
			span1 = span;
			span2 = span - 1;
		}
		for (int i = 0; i < last; i++) {
			end = begin + span1 * increment;
			// putting the task into the queue for a given thread
			thread_spans[i].start = begin;
			thread_spans[i].end = end;
			begin = end;
		}
		for (int i = last; i < numThreads - 1; i++) {
			end = begin + span2 * increment;
			// putting the task into the queue for a given thread
			thread_spans[i].start = begin;
			thread_spans[i].end = end;
			begin = end;
		}
		//for last one enqueue last offset as stop
		//we need it in case our ((stop-start) % increment ) > 0
		thread_spans[numThreads - 1].start = begin;
		thread_spans[numThreads - 1].end = stop;

#ifdef _OPENMP
		if (tryAcquire(numThreads)) {
#pragma omp parallel for
			for (size_t j = 0; j < numThreads; j++) {
				function(j, thread_spans[j].start, thread_spans[j].end, increment);
			}
			freeThreads(numThreads);
			return numThreads;
		}
		else {
			function(0, start, stop, increment);
			// we tell that parallelism request declined
			return 1;
		}
#else
		auto ticket = samediff::ThreadPool::getInstance()->tryAcquire(numThreads);
		if (ticket != nullptr) {

			for (size_t j = 0; j < numThreads; j++) {
				ticket->enqueue(j, numThreads, function, thread_spans[j].start, thread_spans[j].end, increment);
			}
			// block and wait till all threads finished the job
			ticket->waitAndRelease();
			// we tell that parallelism request succeeded
			return numThreads;
		}
		else {
			// if there were no threads available - we'll execute function right within current thread
			function(0, start, stop, increment);
			// we tell that parallelism request declined
			return 1;
		}
#endif
	}
}

