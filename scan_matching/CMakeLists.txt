set(SOURCE_FILES
    "common.h"
    "common.cu"
    "mlp.h"
    "mlp.cu"
    )

cuda_add_library(scan_matching
    ${SOURCE_FILES}
    OPTIONS -arch=sm_60
    )
