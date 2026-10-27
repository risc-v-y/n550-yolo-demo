/* Test-only instrumentation: execute the real cache operations as well. */
#ifndef YOLO_CACHE_AUDIT_H
#define YOLO_CACHE_AUDIT_H
#define board_clean audit_real_clean
#define board_prepare_write audit_real_prepare
#define board_finish_write audit_real_finish
#include "board/cache.h"
#undef board_clean
#undef board_prepare_write
#undef board_finish_write
void cache_audit(unsigned kind,const void *p,size_t bytes);
static inline void board_clean(const void *p,size_t n) {
    cache_audit(0,p,n); audit_real_clean(p,n);
}
static inline void board_prepare_write(const void *p,size_t n) {
    cache_audit(1,p,n); audit_real_prepare(p,n);
}
static inline void board_finish_write(const void *p,size_t n) {
    cache_audit(2,p,n); audit_real_finish(p,n);
}
#endif
