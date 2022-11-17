#include <iostream>
#include <stdlib.h>
#include <iomanip>

using namespace std;

int main()
{
    double number_in_circle = 0;
    int number_of_tosses = 700;
    for (int toss = 0; toss < number_of_tosses; toss++)
    {
        double x = 2 * rand() / (RAND_MAX + 1.0);
        double y = 2 * rand() / (RAND_MAX + 1.0);
        double distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
            number_in_circle++;
    }
    double pi_estimate = 4 * number_in_circle /(( double ) number_of_tosses);
    cout << setprecision(3) << pi_estimate;
}