/*  William
 *  Stewart
 *  wlstewar
 */

#ifndef A0_HPP
#define A0_HPP

template <typename T>
void print_res(int n, T* x)
{
#if DEBUG
        for (int i = 0; i < n; ++i)
                std::cout << x[i] << " ";
        std::cout << std::endl;
#endif
}

int slow_add(int a, int b)
{
#if DEBUG
        volatile int x = 10;
        while(x--);
        return a + b;
#else
        return a+b;
#endif
}

/*
 * High level idea --> do two parallel passes over the array and compute their
 * partial prefixes per processor with the second pass getting the final result.
 *
 * In the general case, we chunk up the array in to n/(p+1) sections.  The first
 * pass we compute the prefixes for the first p sections.  Then in a sequential
 * block, we distribute the last prefix values of each section to their 
 * immediate neighbor in the next section while maintaining a rolling prefix.
 * 
 * In the second pass we shift each PU to their neighbor section and compute the
 * prefix.
 * 
 * In short we use the first pass to compute offsets for the rest of the
 * sections, and then the second pass computes the final result.
 * 
 * Two sequential points:
 * 
 * 1.  If n isn't evenly divisible by p+1, we finish the (n % (p+1)) prefixes
 *     outside of the parallel region.
 * 
 * 2.  If n is too small for the # of processors, we just compute the prefix
 *     sequentially, (assuming ccr resources, n <= 16 and p <= 16).
 */

template <typename T, typename Op>
void omp_scan(int n, const T* in, T* out, Op op) 
{
        int section_size = 0;
        int num_threads = 0;
        int rem = 0;
        #pragma omp parallel
        {
                #pragma omp single
                {       // collect system info
                        num_threads = omp_get_num_threads();
                        rem = n % (num_threads+1);
                        section_size = (n-rem) / (num_threads + 1);
                        out[num_threads * section_size] = in[num_threads * section_size];
                        if(section_size < 2) { //n too small
                                std::partial_sum(in, in + n, out, op);
                                rem = 0;
                        }
                }
                if (section_size > 1) {
                        int id = omp_get_thread_num();
                        int offset = id * section_size;
                        out[offset] = in[offset];
                        for (int i = offset+1; i < offset + section_size; ++i) {
                                out[i] = op(out[i-1], in[i]);
                        }
                        offset += section_size;
                        #pragma omp barrier
                        #pragma omp single
                        {
                                T partials = out[section_size - 1];
                                out[section_size] = partials;
                                for (int i = 2; i < num_threads + 1; ++i) {
                                        partials = op(partials, out[(i * section_size)- 1]);
                                        out[i * section_size] = partials;
                                }
                        }
                        out[offset] = op(out[offset], in[offset]);
                        for (int i = offset + 1; i < offset + section_size; ++i) {
                                out[i] = op(out[i-1], in[i]);
                        }
                }
        }
        int x = rem;
        while(rem) {
                int idx = (num_threads+1) * section_size + (x - rem--);
                out[idx] = op(out[idx-1], in[idx]);
        }
} // omp_scan

/*** <<WORKING NOTES>> ***/

//iteration begins at index 2^i - 1 where i = iter #
//iteration offset is always 2^i + idx(0) where i = iter # and idx(0) is the index of the first element in that iteration
//gotta resolve conflicts however with multiple iters, eg index 3 is used in both iteration 1 and iteration 2.
//easier to just do n/p work per processor

//how can we resolve all prefixes without wasting a processor
/*      we never touch the first n/p section after doing its prefix
 *      we can use it to distribute the partial prefixes across the
 *      remaining sections but that's still a 'join' area and we
 *      can't use the processor again for anything useful (waste).
 *      no matter what we have to distribute the partials before we can continue
 *      in sequence because there's no way to get that information propogated
 *      correctly in parallel...
 * 
 *      create a ghost processor, use the first one twice?
 */

//if we have p = 2, then the work is split like n/2 on p0, n/2 on p1
//the prefix from p0 is easily distributed to p2.
//if we have p = m, then we split like n/m on p0...pm assuming !(n%m)

 
 /*** <<WORKING NOTES>> ***/

#endif // A0_HPP