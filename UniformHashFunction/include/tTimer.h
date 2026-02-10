#ifndef TTIMER_H
#define TTIMER_H

#include <chrono>
using std::chrono::duration;
using HighResClock=std::chrono::high_resolution_clock;
using HighResTimePoint=HighResClock::time_point;

template<typename floatType = double>
struct tTimer
{
    inline tTimer(void)
    {
        Start();
        return;
    }

    inline virtual ~tTimer(void)
    {
        return;
    }

    inline void Start(void)
    {
        StartTime = HighResClock::now();

        return;
    }

    inline floatType GetDuration(void) const
    {
        duration<floatType> Duration = HighResClock::now() - StartTime;

        return Duration.count();
    }

private:
    HighResTimePoint StartTime;
};

typedef tTimer<double> tHighResTimer;
#endif
