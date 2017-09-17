static const char  PyGSL_wavelet_type_doc[] = "a pygsl wavelet type\n";
static const char  PyGSL_wavelet_forward_doc[] = "\n";
static const char  PyGSL_wavelet_inverse_doc[] = "\n";
static const char  PyGSL_transform_space_type_doc[] = "\
A catch all of the various space types of the transform module. Call the method\n\
'get_type' to find what object is wrapped underneath!\n\
";
static const char ww_doc[] =
"Wavelet workspace\n\
\n\
    required for storing intermediate data for the wavelet transform.\
\n\
Args:\n\
     n: Length of the data to transform";
static const char fft_module_doc[] = "\
Wrapper for the FFT Module of the GSL Library\n\
\n\
";
static const char transform_module_doc[] = "\
Wrapper for the FFT Module of the GSL Library\n\
\n\
";
static const char cws_doc[] = "\
Complex Workspace\n\
\n\
Needed as working space for mixed radix routines.\n\
\n\
Args:\n\
    n: Length of the data to transform\n\
";
static const char cwt_doc[] = "\
Complex Wavetable\n\
\n\
   Stores the precomputed trigonometric functions\n\
Args:\n\
    n: Length of the data to transform\n\
";
static const char rws_doc[] = "\
Real Workspace\n\
\n\
Needed as working space for mixed radix routines.\n\
\n\
Args:\n\
    n: Length of the data to transform\n\
";
static const char rwt_doc[] = "\
Real Wavetable\n\
\n\
   Stores the precomputed trigonometric functions\n\
Args:\n\
    n: Length of the data to transform\n\
";

static const char hcwt_doc[] = "\
Half Complex Wavetable\n\
\n\
   Stores the precomputed trigonometric functions\n\
Args:\n\
    n: Length of the data to transform\n\
";
#define TRANSFORM_INPUT "\
Args:\n\
      data   : an array of complex numbers\n\
      space : a workspace of approbriate type and size\n\
      table : a wavetable of approbriate type and size\n\
      output: array to store the output into. GSL computes the FFT\n\
              in place. So if this array is provided, the wrapper\n\
              will use this array as output array. If the input and\n\
              output array are identical no internal copy will be\n\
              made. \n\
              This works only for the complex transform types!\n\
\n\
Optional objects will be generated by the function automatically if required"

#define TRANSFORM_INPUT_RADIX2 "\
Args:\n\
      data   : an array of complex numbers\n\
      output: array to store the output into. GSL computes the FFT\n\
              in place. So if this array is provided, the wrapper\n\
              will use this array as output array. If the input and\n\
              output array are identical no internal copy will be\n\
              made. \n\
              This works only for the complex transform types!\n\
\n\
Optional objects will be generated by the function automatically if required"

#define TRANSFORM_INPUT_REAL "\
Args:\n\
      data: an array of real numbers\n\
      space : a workspace of approbriate type and size\n\
      table : a wavetable of approbriate type and size\n\
      output: array to store the output into. GSL computes the FFT\n\
              in place. So if this array is provided, the wrapper\n\
              will use this array as output array. If the input and\n\
              output array are identical no internal copy will be\n\
              made. \n\
              This works only for the complex transform types!\n\
\n\
Optional objects will be generated by the function automatically if required"

#define TRANSFORM_INPUT_REAL_RADIX2 "\
Args:\n\
      data: an array of real numbers\n\
\n\
Returns:\n\
      the transformed data in its special storage. Halfcomplex data\n\
      in an real array. Use :func:`halfcomplex_radix2_unpack` to transform it\n\
      into an approbriate complex array."

#define TRANSFORM_INPUT_HALFCOMPLEX "\
Args:\n\
      data: an array of complex numbers\n\
      n     : length of the real array. From the complex input I can not\n\
              compute the original length if it was odd or even. Thus I \n\
              allow to give the input here. If not given the routine will guess\n\
              the input length. If the last imaginary part is zero it will\n\
              assume an real output array of even length\n\
      space : a workspace of approbriate type and size\n\
      table : a wavetable of approbriate type and size\n\
      eps   : epsilon to use in the comparisons (default 1e-8)\n\
\n\
If arguments objects are not provided, they will be generated by the\n\
function automatically.\n\
\n\
"

#define TRANSFORM_INPUT_HALFCOMPLEX_RADIX2 "\
Args:\n\
      data: an array of real data containing the complex data\n\
            as required by this transform. See the GSL Reference Document\n\
\n\
"

static const char cf_doc[] = 
"Complex forward transform\n\
" TRANSFORM_INPUT;

static const char cb_doc[] = 
"Complex backward transform\n\
\n\
The output is not scaled!\n\
" TRANSFORM_INPUT;

static const char ci_doc[] = 
"Complex inverse transform\n\
\n\
The output is to scale.\n\
" TRANSFORM_INPUT;

static const char cf_doc_r2[] = 
"Complex forward radix2 transform\n\
" TRANSFORM_INPUT_RADIX2;

static const char cb_doc_r2[] = 
"Complex backward radix2 transform\n\
\n\
The output is not scaled!\n\
" TRANSFORM_INPUT_RADIX2;

static const char ci_doc_r2[] = 
"Complex inverse radix2 transform\n\
\n\
The output is to scale.\n\
" TRANSFORM_INPUT_RADIX2;

static const char cf_doc_r2_dif[] = 
"Complex forward radix2 decimation-in-frequency transform\n\
" TRANSFORM_INPUT_RADIX2;

static const char cb_doc_r2_dif[] = 
"Complex backward radix2 decimation-in-frequency transform\n\
\n\
The output is not scaled!\n\
" TRANSFORM_INPUT_RADIX2;

static const char ci_doc_r2_dif[] = 
"Complex inverse radix2 decimation-in-frequency transform\n\
\n\
The output is to scale.\n\
" TRANSFORM_INPUT_RADIX2;

static const char rt_doc[] = 
"Real transform\n\
" TRANSFORM_INPUT_REAL;

static const char hc_doc[] = 
"Half complex transform\n\
\n\
The output is not scaled!\n\
" TRANSFORM_INPUT_HALFCOMPLEX;

static const char hi_doc[] = 
"Half complex inverse\n\
" TRANSFORM_INPUT_HALFCOMPLEX;

static const char rt_doc_r2[] = 
"Real radix2 transform\n\
" TRANSFORM_INPUT_REAL_RADIX2;

static const char hc_doc_r2[] = 
"Half complex  radix2 transform\n\
\n\
The output is not scaled!\n\
" TRANSFORM_INPUT_HALFCOMPLEX_RADIX2;

static const char hi_doc_r2[] = 
"Half complex  radix2 inverse\n\
" TRANSFORM_INPUT_HALFCOMPLEX_RADIX2;

static const char un_doc_r2[] = 
"Unpack the frequency data from the output of a real radix 2 transform to an approbriate complex array.\n\
";

static const char float_doc [] = 
"Float Version. See the corresponding double version. Remove the trailing _float\n\
";  
