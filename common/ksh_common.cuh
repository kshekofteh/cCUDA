#include <stdio.h>
#include <fstream>
#include <string.h>
#include <string>
#include <iostream>
#include <time.h>
#include <stdlib.h>

#ifdef __linux__
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_YELLOW  "\x1b[33m"
#define COLOR_BLUE    "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN    "\x1b[36m"
#define COLOR_RESET   "\x1b[0m"
#elif _WIN32
#define COLOR_RED     ""//"\33[0:31m\\]"
#define COLOR_GREEN   ""
#define COLOR_YELLOW  ""
#define COLOR_BLUE    ""
#define COLOR_MAGENTA ""
#define COLOR_CYAN    ""
#define COLOR_RESET   ""//"\33[0m\\]"
#endif



using namespace std;


std::string replaceChar(string str, char ch1, char ch2) {
  for (int i = 0; i < str.length(); ++i) {
    if (str[i] == ch1)
      str[i] = ch2;
  }

  return str;
}

const std::string currentDateTime_Old() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return replaceChar(buf, ':', '-');
}

#ifdef _WIN32
	int gettimeofday(struct timeval * tp, struct timezone * tzp)
	{
		// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
		// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
		// until 00:00:00 January 1, 1970 
		static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

		SYSTEMTIME  system_time;
		FILETIME    file_time;
		uint64_t    time;

		GetSystemTime(&system_time);
		SystemTimeToFileTime(&system_time, &file_time);
		time = ((uint64_t)file_time.dwLowDateTime);
		time += ((uint64_t)file_time.dwHighDateTime) << 32;

		tp->tv_sec = (long)((time - EPOCH) / 10000000L);
		tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
		return 0;
	}
#endif

const std::string currentDateTime() {
#ifdef __linux__
	char            fmt[64], buf[64];
#elif _WIN32
	char            fmt[64];
#endif

   struct tm       *tm; 
#ifdef __linux__
   struct timeval  tv;
   gettimeofday(&tv, NULL);
   tm = localtime(&tv.tv_sec);
#elif _WIN32
   time_t long_time;

   time(&long_time);                /* Get time as long integer. */
   tm = localtime(&long_time); /* Convert to local time. */
#endif

   strftime(fmt, sizeof fmt, "%Y-%m-%d %H:%M:%S.%%06u", tm);
#ifdef __linux__
   snprintf(buf, sizeof buf, fmt, tv.tv_usec); 
   return buf;
#elif _WIN32
   return fmt;
#endif
}

const char* strToChar(std::string str)
{
	char * writable = new char[str.size() + 1];
	std::copy(str.begin(), str.end(), writable);
	writable[str.size()] = '\0'; // don't forget the terminating 0
	return writable;
}

void shuffle(int *array, size_t n) {    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int usec = tv.tv_usec;
#ifdef __linux__
	srand48(usec);
#elif _WIN32
	srand(usec);
#endif


    if (n > 1) {
        size_t i;
        for (i = n - 1; i > 0; i--) {
#ifdef __linux__
			size_t j = (unsigned int) (drand48()*(i+1));
#elif _WIN32
			size_t j = (unsigned int)(rand()*(i + 1));
#endif
			int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
void shuffle(int array[], int n) {    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int usec = tv.tv_usec;
#ifdef __linux__
	srand48(usec);
#elif _WIN32
	srand(usec);
#endif


    if (n > 1) {
        size_t i;
        for (i = n - 1; i > 0; i--) {
#ifdef __linux__
			size_t j = (unsigned int)(drand48()*(i + 1));
#elif _WIN32
			size_t j = (unsigned int)(rand()*(i + 1));
#endif
			int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}