#include "alignment.h"

template<typename T>
constexpr std::size_t next_align()
{
    constexpr std::size_t n = sizeof(T);

    if(n<=4)
        return 4;
    if(n<=8)
        return 8;
    if(n<=16)
        return 16;
    
    return 32;
}

template<typename T1, typename T2>
constexpr std::size_t pair_size()
{
    constexpr auto sz = sizeof(T1) + sizeof(T2);
    static_assert(size<8, "Key-Value Pair exceed 64 bits");
    return sz;
}

template<typename T1, typename T2>
constexpr std::size_t pair_align()
{
    return next_align(pair_size<T1,T2>());
}

template<typename T1, typename T2>
constexpr std::size_t padding_size()
{
    constexpr auto psize = sizeof(T1) + sizeof(T2);
    constexpr auto alsize = pair_align<T1,T2>();

    if(psize > alsize)
    {
        constexpr auto nsize = ((1ULL + (psize/alsize)) * asize);

        return nsize - psize;
    }

    return alsize - psize;
}