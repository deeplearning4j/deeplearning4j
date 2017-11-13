//
// Created by raver119 on 10.11.2017.
//

#include <helpers/logger.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {

     bool BitwiseUtils::isBE() {
        short int word = 0x0001;
        char *byte = (char *) &word;
        return(byte[0] ? false : true);
     }

    int BitwiseUtils::valueBit(int holder) {
        if (holder == 0)
            return -1;

#ifdef __LITTLE_ENDIAN__
        for (int e = 0; e < 32; e++) {
#elif __BIG_ENDIAN__
        for (int e = 32; e >= 0; e--) {
#else
        bool be = isBE();
        int start = be ? 32 : 0;
        int stop = be ? -1 : 32;
        int step = be ? -1 : 1;

        for (int e = start; e != stop; e+= step) {
#endif
            bool isOne = (holder & 1 << e) != 0;

            if (isOne)
                return e;
        }

        return -1;
    }


    std::vector<int> BitwiseUtils::valueBits(int holder) {
        std::vector<int> bits;
        if (holder == 0) {
            for (int e = 0; e < 32; e++)
                bits.emplace_back(0);

            return bits;
        }


#ifdef __LITTLE_ENDIAN__
        for (int e = 0; e < 32; e++) {
#elif __BIG_ENDIAN__
        for (int e = 32; e >= 0; e--) {
#else
        bool be = isBE();
        int start = be ? 32 : 0;
        int stop = be ? -1 : 32;
        int step = be ? -1 : 1;

        for (int e = start; e != stop; e+= step) {
#endif
            bool isOne = (holder & 1 << e) != 0;

            if (isOne)
                bits.emplace_back(1);
            else
                bits.emplace_back(0);
        }

        return bits;
    }
}
