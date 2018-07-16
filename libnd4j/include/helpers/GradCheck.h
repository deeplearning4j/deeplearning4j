//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.07.2018
//

#ifndef LIBND4J_GRADCHECK_H
#define LIBND4J_GRADCHECK_H


#include <NDArray.h>
#include <ops/declarable/DeclarableOp.h>

namespace nd4j {

class GradCheck {

    public:
        
        enum LossFunc {MEAN = 0, SUM = 1};
        static bool checkGrad(const ops::DeclarableOp<double>& opFF, const ops::DeclarableOp<double>& opBP, const OpArgsHolder<double>& argsHolderFF, const OpArgsHolder<double>& argsHolderBP, const LossFunc loss);

    private:
        static constexpr double EPSILON = 1e-5;
        static constexpr double MAXRELERR = 1e-5;
        static constexpr double MINABSERR = 1e-6;        
        static void fillGradArrays(const LossFunc loss, const std::vector<NDArray<double>*>& gradArrs);

};





// //////////////////////////////////////////////////////////////////////////
// ///// IMLEMENTATION OF INLINE METHODS ///// 
// //////////////////////////////////////////////////////////////////////////

// template<typename T>
// FORCEINLINE bool ShapeUtils<T>::isPermutNecessary(const std::vector<int>& permut) {        

//     for(int i=0; i<permut.size(); ++i)
//         if(permut[i] != i)
//             return true;

//     return false;
// }



}

#endif //LIBND4J_GRADCHECK_H
