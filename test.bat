call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
          SET MSYSTEM=MINGW64
          SET "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0"
          dir "%CUDA_PATH%"
          dir "%CUDA_PATH%\lib"
          SET PATH="C:\msys64\usr\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64;%PATH%"
          echo "Running cuda build"
          cd %GITHUB_WORKSPACE%
          bash ./change-cuda-versions.sh 11.0
          SET command="mvn -Possrh -Dlibnd4j.buildthreads=${{ github.event.inputs.buildThreads }} -Djavacpp.platform=windows-x86_64 -Dlibnd4j.compute=\"5.0 5.2 5.3 6.0 6.2 8.0\" -Djavacpp.platform=windows-x86_64  -pl \":nd4j-cuda-11.0,:deeplearning4j-cuda-11.0,:libnd4j\" --also-make -Dlibnd4j.platform=windows-x86_64 -Pcuda -Dlibnd4j.chip=cuda clean --batch-mode deploy -DskipTests"
          if %PERFORM_RELEASE%==1
               bash %GITHUB_WORKSPACE}%\release-specified-component.sh  "${RELEASE_VERSION}" "${SNAPSHOT_VERSION}" "${RELEASE_REPO_ID}" %command%
          else
             mvn -Possrh -Dlibnd4j.buildthreads=${{ github.event.inputs.buildThreads }} -Djavacpp.platform=windows-x86_64 -Dlibnd4j.compute="5.0 5.2 5.3 6.0 6.2 8.0" -Djavacpp.platform=windows-x86_64  -pl ":nd4j-cuda-11.0,:deeplearning4j-cuda-11.0,:libnd4j" --also-make -Dlibnd4j.platform=windows-x86_64 -Pcuda -Dlibnd4j.chip=cuda clean  --batch-mode deploy -DskipTests
