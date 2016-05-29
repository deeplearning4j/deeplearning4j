#pragma once

#ifdef __CUDACC__
#define op_def inline __host__  __device__
#elif defined(__GNUC__)
#define op_def inline
#endif


namespace simdOps {
	template<typename T>
	class Add {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d1 + d2;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Subtract {
	public:

#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d1 - d2;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class ReverseSubtract {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d2 - d1;
		}
		
#pragma omp declare simd		
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Multiply {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d1 * d2;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Divide {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d1 / d2;
		}
		
#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class ReverseDivide {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d2 / d1;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};

	template<typename T>
	class Copy {
	public:
#pragma omp declare simd
		op_def static T op(T d1, T d2) {
			return d2;
		}

#pragma omp declare simd
		op_def static T op(T d1) {
			return d1;
		}
	};
}