//
// @author raver119@gmail.com
//

#ifndef PROJECT_OPS_H
#define PROJECT_OPS_H


#define no_special_ff static inline bool requiresSpecialFF() {return false;}; inline static void ffActivation(NDArray<T> *input, NDArray<T>* output) {};
#define no_special_bp static inline bool requiresSpecialBP() {return false;}; inline static void bpActivation(NDArray<T> *input, NDArray<T>* output, NDArray<T> *epsilon) {};

#define no_regular_ff static inline bool requiresSpecialFF() {return true;}; inline static T ffActivation(T value) {};
#define no_regular_bp static inline bool requiresSpecialBP() {return true;}; inline static T bpActivation(T value, T epsilon) {};

#include <layers/activations.h>

// this might be not the best idea, given limited math availability
#include <ops/ops.h>
#include <templatemath.h>

namespace nd4j {
namespace activations 
{
    template<typename T> class ReLU  {

    public:
        no_special_ff
        no_special_bp

        // This method applies activation for FF pass        
        inline static T ffActivation(T value) {
            // FIXME: ultra-bad. should consider configurable extra params here
            T extra[] = {(T) 0.0f};
            return simdOps::RELU<T>::template op(value, extra);
        }

        // This method applies activation for BP pass
        inline static T bpActivation(T value, T epsilon) {
            // FIXME: ultra-bad. should consider configurable extra params here
            T extra[] = {(T) 0.0f};
            return simdOps::Step<T>::template op(value, extra) * epsilon;
        }
};



template<typename T> class Identity  {

    public:
        no_special_ff
        no_special_bp
    
        // This method applies activation for FF pass

        inline static T ffActivation(T value) {
            return value;
        }
    
        // This method applies activation for BP pass
        inline static T bpActivation(T value, T epsilon) {
            return epsilon;
        }
};



// end of namespace brackets
}
}

#endif //PROJECT_OPS_H
