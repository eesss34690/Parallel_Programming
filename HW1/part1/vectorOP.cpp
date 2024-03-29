#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
	__pp_vec_float x, result;
	__pp_vec_int y, count;
	
	__pp_mask maskAll, mask_y_not_0, y_equals_0, exp, gt_nine;

	__pp_vec_int zero = _pp_vset_int(0), vec_int_one = _pp_vset_int(1);
	__pp_vec_float float_nine = _pp_vset_float(9.999999f);
  __pp_vec_float float_one = _pp_vset_float(1.f);
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // select remain
		if ( i + VECTOR_WIDTH > N) 
      maskAll = _pp_init_ones(N-i);
		// select all
		else maskAll = _pp_init_ones();
		
		y_equals_0 = _pp_init_ones(0);
		gt_nine = _pp_init_ones(0);
		exp = _pp_init_ones(0);

		// laod values and exponents
		_pp_vload_float(x, values + i, maskAll);
		_pp_vload_int(y, exponents + i, maskAll);
		
		// if y = 0, output = 1.f
		_pp_veq_int(y_equals_0, y, zero, maskAll);
		_pp_vset_float(result, 1.f, y_equals_0);
		
		// else
		mask_y_not_0 = _pp_mask_not(y_equals_0);
		// result = x
		_pp_vload_float(result, values+i, mask_y_not_0);
		// count = y - 1
		_pp_vsub_int(count, y, vec_int_one, mask_y_not_0);
		
		_pp_vgt_int(exp, count, zero, mask_y_not_0);
		while(_pp_cntbits(exp) != 0){
			// result *= x
			_pp_vmult_float(result, result, x, exp);
			// count--
			_pp_vsub_int(count, count, vec_int_one, exp);
      // see whether there is elemets still need multiplication
			_pp_vgt_int(exp, count, zero, exp);
		}
		
		// if result > 9.999999f, result = 9.999999f
		_pp_vgt_float(gt_nine, result, float_nine, mask_y_not_0);
		_pp_vset_float(result, 9.999999f, gt_nine);
		
		// output = result
		_pp_vstore_float(output+i, result, maskAll);
  }
}

float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    __pp_vec_float x, x_cross;
    __pp_mask maskAll;
    float *output = new float[VECTOR_WIDTH];
    float sum = 0.f;
    int t = VECTOR_WIDTH;
    
    for (int i = 0; i < N; i += VECTOR_WIDTH)
    {
      t = VECTOR_WIDTH;
      // load all the values
      maskAll = _pp_init_ones();
      _pp_vload_float(x, values+i, maskAll);
        
      while(t > 1){
        // add adjacent values in the vector
        _pp_hadd_float(x, x);
        // total number of times divided by 2
        t >>= 1;
          
        // not yet finished
        if(t > 1){
          // interleave the values to let the values being added all the way down
          _pp_interleave_float(x_cross, x);
          _pp_vmove_float(x, x_cross, maskAll);
        }
      }
      // store the result
      _pp_vstore_float(output, x, maskAll);
      // all the sum is at hte first elements
      sum += output[0];
    }
    return sum;
  }
  return 0.f;
}
