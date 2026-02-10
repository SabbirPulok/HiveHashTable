#ifndef TSTATS_H
#define TSTATS_H

#include <cmath>
#include <ostream>
  using std::ostream;

template<typename floatType = double>
struct tStats
{
    inline tStats(void)
    {
        Init();
        return;
    }

    inline virtual ~tStats(void)
    {
        return;
    }

    inline void Init(void)
    {
        SumX = SumXX = Count = Min = Max = floatType(0.0);

        return;
    }

    inline tStats &operator +=(floatType X)
    {
        if (Count == floatType(0.0))
        {
            Min = Max = X;
        }
        else
        {
            if (X < Min)
            {
                Min = X;
            }

            if (X > Max)
            {
                Max = X;
            }
        }

        SumX  += X;
        SumXX += X * X;
        ++Count;

        return *this;
    }

    // Include (absorb) other stats into this one.
    inline tStats &Include(const tStats &OtherStats)
    {
        if (Count == floatType(0.0))
        {
            Min = Max = OtherStats.Max;
        }
        else
        {
            if (OtherStats.Min < Min)
            {
                Min = OtherStats.Min;
            }

            if (OtherStats.Max > Max)
            {
                Max = OtherStats.Max;
            }
        }

        SumX  += OtherStats.SumX;
        SumXX += OtherStats.SumXX;;
        Count += OtherStats.Count;

        return *this;
    }

    inline floatType Avg(void) const
    {
        return SumX / Count;
    }

    inline floatType SumOfSquares(void) const
    {
        auto SumSq { SumXX - SumX * SumX / Count };

        // In some cases, roundoff can produce a negative value.
        // This means the sumSq is best approximated by 0.0.
        return (SumSq < floatType(0.0)) ? floatType(0.0) : SumSq;
    }

    inline floatType PopVar(void) const
    {
        return SumOfSquares() / Count;
    }

    inline floatType PopStd(void) const
    {
        return sqrt(PopVar());
    }

    inline floatType SampVar(void) const
    {
        return SumOfSquares() / (Count - floatType(1.0));
    }

    inline floatType SampStd(void) const
    {
        return sqrt(SampVar());
    }

    inline floatType Minimum(void) const
    {
        return Min;
    }

    inline floatType Maximum(void) const
    {
        return Max;
    }

    inline floatType Range(void) const
    {
        return Max - Min;
    }

    inline floatType N(void) const
    {
        return Count;
    }

    floatType SumX;
    floatType SumXX;
    floatType Count;
    floatType Min;
    floatType Max;
};

template<typename floatType = double>
ostream &operator<<(ostream &os, const tStats<floatType> &Stats)
{ return os <<
      "N: "          << Stats.Count     <<
      ", Min: "      << Stats.Min       <<
      ", Max: "      << Stats.Max       <<
      ", Range: "    << Stats.Range()   <<
      ", Sum: "      << Stats.SumX      <<
      ", Mean: "     << Stats.Avg()     <<
      ", PopStd: "   << Stats.PopStd()  <<
      ", SampStd: "  << Stats.SampStd()
      ;
}

#endif
