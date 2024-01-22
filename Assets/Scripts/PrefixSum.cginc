#if !defined(PREFIX_SUM_ARRAY_NAME)
    #error "Please define PREFIX_SUM_ARRAY_NAME as a proper array name."
#endif

// Original array needs to be calculated.
// Make its length round to THREADS.
RWStructuredBuffer<uint> PREFIX_SUM_ARRAY_NAME;

// Temp array with the length of ceil(length(arr) / THREADS).
RWStructuredBuffer<uint> groupArr;

// Make it power of two.
#define THREAD_NUM 1024

// Double buffered
groupshared uint tmp[THREAD_NUM*2];

// Calculate prefix sum for each groups.
[numthreads(THREAD_NUM,1,1)]
void PrefixSum1 (uint3 id : SV_DispatchThreadID)
{
    uint length, stride;
    PREFIX_SUM_ARRAY_NAME.GetDimensions(length, stride);

    uint localIndex = id.x & (THREAD_NUM-1);
    if (id.x < length) {
        // Copy data to groupshared memory.
        tmp[localIndex] = PREFIX_SUM_ARRAY_NAME[id.x];
    }

    GroupMemoryBarrierWithGroupSync();

    uint bufferIndex = 0;
    for (uint i = 1; i < THREAD_NUM; i <<= 1) {
        if (id.x < length) {
            if (localIndex >= i) {
                tmp[localIndex + (bufferIndex^1) * THREAD_NUM] = tmp[(localIndex-i) + bufferIndex * THREAD_NUM] + tmp[localIndex + bufferIndex * THREAD_NUM];
            }
            else {
                tmp[localIndex + (bufferIndex^1) * THREAD_NUM] = tmp[localIndex + bufferIndex * THREAD_NUM];
            }
        }
        bufferIndex ^= 1;

        GroupMemoryBarrierWithGroupSync();
    }

    // Write results.
    if (id.x < length) {
        PREFIX_SUM_ARRAY_NAME[id.x] = tmp[localIndex + bufferIndex * THREAD_NUM];
    }
}

// Calculate prefix sum for sum of each groups.
[numthreads(THREAD_NUM,1,1)]
void PrefixSum2 (uint3 id : SV_DispatchThreadID)
{
    uint length, stride;
    groupArr.GetDimensions(length, stride);

    uint localIndex = id.x & (THREAD_NUM-1);
    if (id.x < length) {
        // Copy data to groupshared memory.
        tmp[localIndex] = PREFIX_SUM_ARRAY_NAME[id.x * THREAD_NUM + (THREAD_NUM-1)];
    }

    GroupMemoryBarrierWithGroupSync();

    uint bufferIndex = 0;
    for (uint i = 1; i < THREAD_NUM; i <<= 1) {
        if (id.x < length) {
            if (localIndex >= i) {
                tmp[localIndex + (bufferIndex^1) * THREAD_NUM] = tmp[(localIndex-i) + bufferIndex * THREAD_NUM] + tmp[localIndex + bufferIndex * THREAD_NUM];
            }
            else {
                tmp[localIndex + (bufferIndex^1) * THREAD_NUM] = tmp[localIndex + bufferIndex * THREAD_NUM];
            }
        }
        bufferIndex ^= 1;

        GroupMemoryBarrierWithGroupSync();
    }

    // Write results.
    if (id.x < length) {
        groupArr[id.x] = tmp[localIndex + bufferIndex * THREAD_NUM];
    }
}

// Add offset to each groups and finalize the results.
[numthreads(THREAD_NUM,1,1)]
void PrefixSum3 (uint3 id : SV_DispatchThreadID)
{
    uint length, stride;
    PREFIX_SUM_ARRAY_NAME.GetDimensions(length, stride);

    if (id.x < length) {
        if (id.x >= THREAD_NUM) {
            PREFIX_SUM_ARRAY_NAME[id.x] += groupArr[id.x / THREAD_NUM - 1];
        }
    }
}