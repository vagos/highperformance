#pragma once

#include <array>

class Point : public std::array<double, PROBDIM>
{
    public:
        static const int DIM = PROBDIM;

        Point() {}

        Point(double* p)
        {

            for (std::size_t i = 0; i < PROBDIM; i++)
            {
                (*this)[i] = p[i];
            }
        }
};

