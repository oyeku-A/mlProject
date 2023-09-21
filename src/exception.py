import os
import sys

def error_message_detail(error, error_detail:sys):
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    file_name=os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_message="Error occured in python script name: {0}, Line number: '{1}' and Eror message: '{2}'".format(file_name, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self)->str:
        return self.error_message

