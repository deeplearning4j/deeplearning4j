//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.07.2018
//

#ifndef LIBND4J_GRADCHECK_H
#define LIBND4J_GRADCHECK_H


#include <NDArray.h>

namespace nd4j {
 
class GradCheck {

    private:
        static constexpr double EPS = 1e-5;
        static constexpr double MAXRELERR = 1e-5;
        static constexpr double MINABSERR = 1e-6;

    public:

        static bool checkGrad(const ops::DeclarableOp<double> &op, const )
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
