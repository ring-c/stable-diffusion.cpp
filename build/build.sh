set -ex
export CUDACXX=/usr/local/cuda-12/bin/nvcc

cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUBLAS=ON -DSD_BUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES=OFF -DSD_BUILD_SHARED_LIBS=ON
cmake --build . --config Release -j 10
mv -f ./bin/libstable-diffusion.so /code/github/ring-c/go-web-diff/pkg/bind/deps/linux/libsd-abi.so
