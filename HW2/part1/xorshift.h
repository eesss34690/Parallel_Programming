#include<immintrin.h>
#include<stdint.h>

struct avx_key{
	__m256i part1;
	__m256i part2;
};

static void xorshift128plus_keys(uint64_t * pos0, uint64_t * pos1) {
	uint64_t s1 = *pos0;
	const uint64_t s0 = *pos1;
	*pos0 = s0;
	s1 ^= s1 << 23; 
	*pos1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
}

static void xorshift128plus_jump(uint64_t in1, uint64_t in2, uint64_t * out1, uint64_t * out2) {
	uint64_t s0 = 0;
	uint64_t s1 = 0;
	
	for (int b = 0; b < 64; b++) {
		if (0x8a5cd789635d2dff & 1ULL << b) {
			s0 ^= in1;
	  		s1 ^= in2;
		}
		xorshift128plus_keys(&in1, &in2);
	}
	
	for (int b = 0; b < 64; b++) {
		if (0x121fd2155c472f96 & 1ULL << b) {
			s0 ^= in1;
	  		s1 ^= in2;
		}
		xorshift128plus_keys(&in1, &in2);
	}

	out1[0] = s0;
	out2[0] = s1;
}

void xorshift128plus_init(uint64_t key1, uint64_t key2, avx_key *key) {
	uint64_t s0[4];
	uint64_t s1[4];
 
	s0[0] = key1;
	s1[0] = key2;
 
	xorshift128plus_jump(*s0, *s1, s0 + 1, s1 + 1);
	xorshift128plus_jump(*(s0 + 1), *(s1 + 1), s0 + 2, s1 + 2);
	xorshift128plus_jump(*(s0 + 2), *(s1 + 2), s0 + 3, s1 + 3);
 
	key->part1 = _mm256_loadu_si256((__m256i const *) s0);
	key->part2 = _mm256_loadu_si256((__m256i const *) s1);
}

__m256 xorshift128plus(avx_key *key) {
	__m256i s1 = key->part1;
	__m256i s0 = key->part2;
	key->part1 = key->part2;
	s1 = _mm256_xor_si256(key->part2, _mm256_slli_epi64(key->part2, 23));
	key->part2 = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(s1, s0),
			_mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
	return _mm256_cvtepi32_ps(_mm256_add_epi64(key->part2, s0));
}
