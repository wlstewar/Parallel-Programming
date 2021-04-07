#include <chrono>
#include <functional>
#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>

#include "a1.hpp"

using namespace std::chrono;


inline int hash(int x) { return x; }

int main(int argc, char* argv[]) {
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const long int N = 32 * 1024 * 1024;
    const int SEED = 113 + rank;

    std::vector<int> buf((rank + 1) * N);

    std::mt19937 re(SEED);
    std::uniform_int_distribution<int> uniform(0, 256);

    auto rng = std::bind(uniform, re);
    for (auto& x : buf) x = rng();

    MPI_Barrier(MPI_COMM_WORLD);

    auto start = high_resolution_clock::now();
    mpi_shuffle(buf, hash, MPI_INT, MPI_COMM_WORLD);
    auto stop = high_resolution_clock::now();

    using seconds_t = duration<double, std::ratio<1,1>>;
    auto duration = duration_cast<seconds_t>(stop - start);

    if (rank == 0) {
        std::cout << "p: " << size << std::endl;
        std::cout << "Tp: " << duration.count() << std::endl;
    }
    int correct = check(buf, size, rank, hash);
    std::vector<int> chk(size);
    MPI_Gather(&correct, 1, MPI_INT, chk.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << ((std::accumulate(chk.begin(), chk.end(),0) == size) ? "pass" : "fail") << std::endl;
    }

    return MPI_Finalize();
} // main
