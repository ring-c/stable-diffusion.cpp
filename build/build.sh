set -ex
export CUDACXX=/usr/local/cuda-12/bin/nvcc

cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=OFF
cmake --build . --config Release -j 10
mv -f ./bin/libsd-abi.so /code/github/ring-c/go-web-diff/pkg/bind/deps/linux/
