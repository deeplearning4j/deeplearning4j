//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.07.2018
//

#ifndef LIBND4J_OPARGSHOLDER_H
#define LIBND4J_OPARGSHOLDER_H


#include <NDArray.h>

namespace nd4j {
 
template<typename T>
class OpArgsHolder {

    private: 
        std::vector<NDArray<T>*> _inArrs = std::vector<NDArray<T>*>();
        std::vector<T>           _tArgs  = std::vector<T>();
        std::vector<int>         _iArgs  = std::vector<int>();

        int _numInArrs = _inArrs.size();
        int _numTArgs  = _tArgs.size();
        int _numIArgs  = _iArgs.size();

        std::vector<bool> _isArrAlloc = std::vector<bool>(_numInArrs, false);

    public:

        OpArgsHolder(const std::vector<NDArray<T>*>& inArrs = std::vector<NDArray<T>*>(), const std::vector<T>& tArgs = std::vector<T>(), const std::vector<int>& iArgs = std::vector<int>())
                    : _inArrs(inArrs), _tArgs(tArgs), _iArgs(iArgs) { }

        const std::vector<NDArray<T>*>& getInArrs() const
        {return _inArrs; }

        const std::vector<T>& getTArgs() const
        {return _tArgs; }

        const std::vector<int>& getIArgs() const
        {return _iArgs; }

        int getNumInArrs() const
        {return _numInArrs; }

        int getNumTArgs() const
        {return _numTArgs; }

        int getNumIArgs() const
        {return _numIArgs; }
};





}

#endif //LIBND4J_OPARGSHOLDER_H
