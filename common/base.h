#define __DELAY__TICK 100000000
#define _delay()								\
	for (int i = 0;i < __DELAY__TICK;i++);

