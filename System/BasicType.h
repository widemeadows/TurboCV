#include <iomanip>
#include <sstream>
#include <string>
#include "String.h"
using namespace std;

namespace System
{
    class Int
    {
    public:
        static int Parse(const String& integer)
        {
            stringstream ss;
            int result;

            ss << integer;
            ss >> result;

            return result;
        }

        static String ToString(int integer)
        {
            stringstream ss;
            String result;

            ss << integer;
            ss >> result;

            return result;
        }
    };
    
    class Float
    {
    public:
        static float Parse(const String& floating)
        {
            stringstream ss;
            float result;

            ss << setprecision(7) << floating;
            ss >> result;

            return result;
        }

        static String ToString(float floating)
        {
            stringstream ss;
            String result;

            ss << setprecision(7) << floating;
            ss >> result;

            return result;
        }
    };

    class Double
    {
    public:
        static double Parse(const String& floating)
        {
            stringstream ss;
            double result;

            ss << setprecision(16) << floating;
            ss >> result;

            return result;
        }

        static String ToString(double floating)
        {
            stringstream ss;
            String result;

            ss << setprecision(16) << floating;
            ss >> result;

            return result;
        }
    };
}