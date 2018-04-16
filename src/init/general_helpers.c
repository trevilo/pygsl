/* -*- C -*- */
/**
 * Author: Pierre Schnizer		
 * Date : January 2003
 */
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <pygsl/general_helpers.h>
#include <pygsl/utils.h>
#include <pygsl/profile.h>

#define PyGSL_ERROR_STRING_BFFER_N 2048
PyGSL_API_EXTERN int
PyGSL_set_error_string_for_callback(PyGSL_error_info * info)
{
     PyObject *name_o = NULL;
     PyObject *callback;
     const char * message = "";
     const char * error_description = "";
     const char * mmesg;
     char * name, msg[PyGSL_ERROR_STRING_BFFER_N];
     char *formatstring = "For the callback %s evaluted  for function %s, an error occured : %s";

     int status;

     FUNC_MESS_BEGIN();    
     callback = info->callback;
     if(info->message){
	  message = info->message;          
     }
     if(info->error_description){
	  error_description = info->error_description;
     }
     

     if (message == NULL){
	  mmesg = "Unknown";
     }else{
	  mmesg = message;
     }
     assert(callback != NULL);
     name_o = PyObject_GetAttrString(callback, "__name__");
     if (name_o == NULL){
	  name_o = PyObject_GetAttrString(callback, "func_name");
     }
     if (name_o == NULL){
	 PyErr_SetString(PyExc_AttributeError, 
			 "While I was treating an errornous callback object,"
			 " I found that it had no attribute '__name__'!");
	 DEBUG_MESS(2, "Coukd not get the name of the callback %p", (void *) callback);
	 PyGSL_ERROR("Could not get the name of the callback!", GSL_EBADFUNC);
	 goto fail;
     }
     if(!PyGSL_string_check(name_o)){
	  PyErr_SetString(PyExc_TypeError, 
			  " For an errornous callback object,"  
			  " the attribute '__name__' was not a Python string!");
	 DEBUG_MESS(2, "Nameobject of the callback  %p was not a string", (void *) name_o);
	 PyGSL_ERROR("Nameobject of the callback was not a string!", GSL_EBADFUNC);
	 goto fail;
     }     
     name = PyGSL_string_as_string(name_o);

     /* A non completly standard but safe function. */
     DEBUG_MESS(2, "making string for %s", name);
     snprintf(msg,  PyGSL_ERROR_STRING_BFFER_N, formatstring, name, mmesg, error_description);
     if(DEBUG > 2){
       DEBUG_MESS(2, "error message %s", msg);
     }
     status = GSL_EBADFUNC;
     pygsl_error(msg, __FILE__, __LINE__, status);

     FUNC_MESS_END();
     return GSL_EBADFUNC;
     /* Py_XDECREF(name_o);  ??? */

 fail:
     FUNC_MESS_FAILED();
     return status;
     /* Py_XDECREF(name_o);  ??? */
}


PyGSL_API_EXTERN int 
PyGSL_pyfloat_to_double(PyObject *object, double *result, PyGSL_error_info *info)
{
     
     PyObject *object1;
     char *msg="The object returned to the GSL Function could not be converted to float";

     FUNC_MESS_BEGIN();
     object1 = PyNumber_Float(object);
     if(object1 == NULL){
	 *result = gsl_nan();
	 if(info){
	      info->error_description = msg;
	      return PyGSL_set_error_string_for_callback(info);
	 }
	 DEBUG_MESS(2, "Not from call back treatment, normal error. info = %p", info);
	 PyGSL_ERROR(msg, GSL_EBADFUNC);
     }

     *result   = PyFloat_AsDouble(object1);
     DEBUG_MESS(3, "found a double of %f\n", *result);
     Py_DECREF(object1);

     PyGSL_INCREASE_float_transform_counter();
     FUNC_MESS_END();
     return GSL_SUCCESS;
}

PyGSL_API_EXTERN int 
PyGSL_pylong_to_uint(PyObject *object, unsigned int *result, PyGSL_error_info *info)
{
     int flag;
     unsigned long int tmp;
     flag =PyGSL_pylong_to_ulong(object, &tmp, info);
     *result = (unsigned int) tmp;
     return flag;
}


PyGSL_API_EXTERN int 
PyGSL_pylong_to_ulong(PyObject *object, unsigned long *result, PyGSL_error_info *info)
{
     
     PyObject *object1;
     char *msg="The object returned to the GSL Function could not be converted to unsigned long";


     object1 = PyNumber_Long(object);
     if(object1 == NULL){
	 *result = 0;
	 if(info){
	      info->error_description = msg;
	      return PyGSL_set_error_string_for_callback(info);
	 }
	 PyGSL_ERROR(msg, GSL_EINVAL);
     }

     *result   = PyLong_AsUnsignedLong(object1);
     if(DEBUG>2){
	  fprintf(stderr, "\t\t%s found a double of %ld\n", __FUNCTION__, *result);
     }
     Py_DECREF(object1);

     PyGSL_INCREASE_float_transform_counter();

     return GSL_SUCCESS;
}

PyGSL_API_EXTERN int 
PyGSL_pyint_to_int(PyObject *object, int *result, PyGSL_error_info *info)
{
     
     PyObject *object1;
     char *msg="The object returned to the GSL Function could not be converted to int";
     long tmp;

     FUNC_MESS_BEGIN();
     object1 = PyNumber_Long(object);
     if(object1 == NULL){
	 *result = INT_MIN;
	 if(info){
	      info->error_description = msg;
	      return PyGSL_set_error_string_for_callback(info);
	 }
	 DEBUG_MESS(2, "Not from call back treatment, normal error. info = %p", info);
	 PyGSL_ERROR(msg, GSL_EINVAL);
     }

     tmp = PyLong_AsLong(object1);
     if(tmp > INT_MAX)
	  PyGSL_ERROR("Number too big for int", GSL_EINVAL);
     else if (tmp < INT_MIN)
	  PyGSL_ERROR("Number too small for int", GSL_EINVAL);

     *result = (int) tmp;
     DEBUG_MESS(3, "found a int of %d\n", *result);
     Py_DECREF(object1);
     FUNC_MESS_END();
     return GSL_SUCCESS;
}


/*
 * Checks following conditions:
 *  For No Arguments: Got Py_None and No Error
 *  For 1  Argument:  Got an Object, No None  and No Error 
 *         (Is None a legal return for one object? I think so.) On the other hand its a
 *         callback and Conversions are waiting, so its good not to accept None. 
 * For 2  Arguments: Got a tuple of approbriate size
 */
#define PyGSL_CHECK_PYTHON_RETURN(object, nargs, info)                              \
  (                                                                                 \
        (  ( (nargs) == 0 ) && ( object ) && ( Py_None == (object) ) && ( !PyErr_Occurred() ) )   \
    ||  (  ( (nargs) == 1 ) && ( object ) && ( Py_None != (object) ) && ( !PyErr_Occurred() ) )   \
    ||  (  ( (nargs) >  1 ) && ( object ) && ( PyTuple_Check((object)) ) &&                       \
                 ( (nargs) == PyTuple_GET_SIZE((object)) ) )                        \
 )                                                                                  \
 ?                                                                                  \
    GSL_SUCCESS                                                                     \
 :                                                                                  \
   PyGSL_check_python_return((object), (nargs), (info))        

PyGSL_API_EXTERN int
PyGSL_check_python_return(PyObject *object, int nargs, PyGSL_error_info  *info)
{
     int tuple_size, flag=-1;
     char *msg;
     int status;
     PyObject *py_exception = NULL;
     
     FUNC_MESS_BEGIN();

     DEBUG_MESS(2, "Return object  = %p expected args = %d info = %p",
		object, nargs, info);

     if(info == NULL){
	     DEBUG_MESS(2, "info %p must not be NULL ", info);
	  FUNC_MESS_FAILED();
	  return GSL_EBADFUNC;
     }

     DEBUG_MESS(2, "info->message = '%s' info->info.error_description = '%s'",
		info->message, info->error_description);

     py_exception = PyErr_Occurred();
     DEBUG_MESS(2, "Py_Exception = %p", py_exception);

     if(object == NULL && py_exception){
	  /* 
	   * Error was apparently raised by the function, so lets just add a
	   * traceback frame .... 
	   */
	  info->error_description = "User function raised exception!";
	  PyGSL_add_traceback(NULL, "Unknown file", info->message, __LINE__);
	  FUNC_MESS_END();
	  return GSL_EBADFUNC;
     }
     if(py_exception){
	  info->error_description = "Function raised an exception.";
	  PyGSL_add_traceback(NULL, "Unknown file", info->message, __LINE__);
	  FUNC_MESS_END();
	  return GSL_EBADFUNC;
	  /* return PyGSL_set_error_string_for_callback(info); */
     }

     /* Expected No argumets */	
     if(nargs == 0){
	  if(object != Py_None){
	       info->error_description = "I expected 0 arguments, but I got an object different from None.";
	       FUNC_MESS_END();
	       status = PyGSL_set_error_string_for_callback(info);
	       FUNC_MESS_END();
	       return status;
	  } else {
	       FUNC_MESS_END();
	       return GSL_SUCCESS;
	  }
     }

     if(nargs == 1){
	  if(object == Py_None){
	       info->error_description = "Expected 1 argument, but None was returned. This value is not acceptable for" 
		    " the following arithmetic calculations.";
	       status = PyGSL_set_error_string_for_callback(info);
	       FUNC_MESS_END();
	       return status;
	  } else {
	       FUNC_MESS_END();
	       return GSL_SUCCESS;
	  } 
     }

     if(nargs > 1){
	  msg = (char *) malloc(256 * sizeof(char));

	  if(object == Py_None){
	       snprintf(msg, 256, "I expected %d arguments, but the function returned None!", nargs);
	       info->error_description = msg;
	       flag = PyGSL_set_error_string_for_callback(info);	       

	  } else if(!PyTuple_Check(object)){
	       snprintf(msg, 256, "Expected %d arguments, but I didn't get a tuple! "
			"Did you just return one argument?.", nargs);
	       info->error_description = msg;
	       flag = PyGSL_set_error_string_for_callback(info);	       

	  } else {
	       tuple_size = PyTuple_GET_SIZE(object);
	       if(tuple_size != nargs){
		    snprintf(msg, 256, "I expected %d arguments, but the function returned %d arguments! ",
			     nargs, tuple_size);
	       info->error_description = msg;
	       flag = PyGSL_set_error_string_for_callback(info);

	       } else {
		    flag = GSL_SUCCESS;
	       }
	  }
	  free(msg);
     }    
     FUNC_MESS_END();
     return flag;
}

PyGSL_API_EXTERN void
PyGSL_clear_name(char *name, int size)
{
     int j;
     for(j = 0; j<size; j++){
	  if(name[j] == '-')
	       name[j] = '_';
     }
}
