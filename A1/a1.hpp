/*  William
 *  Stewart
 *  wlstewar
 */

#ifndef A1_HPP
#define A1_HPP

#include <algorithm>

/*
 * Distributed memory shuffle w/ Open MPI
 *
 * Function mpi_shuffle hashes elements modulus the number of PUs participating
 * in the shuffle.  We sort the data locally in ascending order of PU rank,
 * and communicate the number of elements to be sent to each processor via an
 * MPI_Alltoall().  Then we compute offsets for an MPI_Alltoallv() and allocate
 * space for the elements the individual PU will receive in the communication.
 * We call MPI_Alltoallv(), swap out the input vector for the output vector and
 * deallocate the input vector.
 */
template <typename T, typename Hash>
void mpi_shuffle(std::vector<T>& inout, Hash hash, MPI_Datatype Type, MPI_Comm Comm)
{
        int rank = 0;
        int size = 0;
        MPI_Comm_rank(Comm, &rank);
        MPI_Comm_size(Comm, &size);

        // compute number of elements to send to each PU and sort vector
        std::vector<int> cnts(size);
        for (auto &x : inout) {
                ++cnts[hash(x) % size];
        }
        std::sort(inout.begin(), inout.end(),
          [hash, size](T x, T y) {return (hash(x) % size) < (hash(y) % size);});

        // send and get counts of how many elements will come from each PU
        std::vector<int> recv_cnts(size);
        MPI_Alltoall(cnts.data(), 1, MPI_INT, recv_cnts.data(), 1, MPI_INT, 
                     Comm);

        // allocate space for the elements we're getting
        int eles = std::accumulate(recv_cnts.begin(), recv_cnts.end(), 0);
        std::vector<T> recv_vec(eles);

        // get list of where to send/recv elements
        std::vector<int> send_offsets(size);
        std::vector<int> recv_offsets(size);
        send_offsets[0] = 0;
        recv_offsets[0] = 0;
        for (int i = 1; i < size; i++) {
                recv_offsets[i] = recv_offsets[i-1] + recv_cnts[i-1];
                send_offsets[i] = send_offsets[i-1] + cnts[i-1];
        }

        // shuffle time!
        MPI_Alltoallv(inout.data(), cnts.data(), send_offsets.data(), Type,
                recv_vec.data(), recv_cnts.data(), recv_offsets.data(), Type,
                Comm);

        // clean up
        inout.swap(recv_vec);
        recv_vec.clear();
} // mpi_shuffle

template <typename T, typename Hash>
inline int check(std::vector<T>& buf, int size, int rank, Hash hash)
{
    return std::all_of(buf.begin(), buf.end(),
        [hash, rank, size](T x){return (hash(x)%size) == rank;});
}

#endif // A1_HPP