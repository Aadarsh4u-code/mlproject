import sys

def error_message_details(error, error_detail: sys):
    _,_,exc_traceback = error_detail.exc_info() # execuation info and it will give 3 info 
    error_message = "Error occured in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name = exc_traceback.tb_frame.f_code.co_filename,
        exc_traceback.tb.lineno,
        str(error)) 

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message



