#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" fftw
    popd
    exit
fi

if [[ $PLATFORM == windows* ]]; then
    FFTW_VERSION=3.3.4
    [[ $PLATFORM == *64 ]] && BITS=64 || BITS=32
    download ftp://ftp.fftw.org/pub/fftw/fftw-$FFTW_VERSION-dll$BITS.zip fftw-$FFTW_VERSION-dll$BITS.zip

    mkdir -p $PLATFORM
    cd $PLATFORM
    mkdir -p include lib
    unzip -o ../fftw-$FFTW_VERSION-dll$BITS.zip -d fftw-$FFTW_VERSION-dll$BITS
    cd fftw-$FFTW_VERSION-dll$BITS
else
    FFTW_VERSION=3.3.4
    download http://www.fftw.org/fftw-$FFTW_VERSION.tar.gz fftw-$FFTW_VERSION.tar.gz

    mkdir -p $PLATFORM
    cd $PLATFORM
    INSTALL_PATH=`pwd`
    tar -xzvf ../fftw-$FFTW_VERSION.tar.gz
    cd fftw-$FFTW_VERSION
fi

case $PLATFORM in
    android-arm)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --host="arm-linux-androideabi" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -fstack-protector -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16 -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib -Wl,--fix-cortex-a8" LIBS="-lgcc -ldl -lz -lm -lc" --enable-float
        make -j4
        make install-strip
        ;;
     android-x86)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc"
        make -j4
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --host="i686-linux-android" --with-sysroot="$ANDROID_ROOT" CC="$ANDROID_BIN-gcc" STRIP="$ANDROID_BIN-strip" CFLAGS="--sysroot=$ANDROID_ROOT -DANDROID -fPIC -ffunction-sections -funwind-tables -mssse3 -mfpmath=sse -fomit-frame-pointer -fstrict-aliasing -funswitch-loops -finline-limit=300" LDFLAGS="-nostdlib" LIBS="-lgcc -ldl -lz -lm -lc" --enable-float
        make -j4
        make install-strip
        ;;
    linux-x86)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m32"
        make -j4
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m32" --enable-float
        make -j4
        make install-strip
        ;;
    linux-x86_64)
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m64"
        make -j4
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-avx CC="gcc -m64" --enable-float
        make -j4
        make install-strip
        ;;
    macosx-*)
        patch -Np1 < ../../../fftw-$FFTW_VERSION-macosx.patch
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2
        make -j4
        make install-strip
        ./configure --prefix=$INSTALL_PATH --enable-shared --enable-threads --with-combined-threads --enable-sse2 --enable-float
        make -j4
        make install-strip
        ;;
    windows-x86)
        cp *.h ../include
        cp *.dll ../lib
        ;;
    windows-x86_64)
        cp *.h ../include
        cp *.dll ../lib
        ;;
    *)
        echo "Error: Platform \"$PLATFORM\" is not supported"
        ;;
esac

cd ../..