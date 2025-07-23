import sys
from src.logger import logging
import inspect

def error_message_details(error, error_detail: sys):
    try:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            # Manually get file and line info from where the exception is raised
            frame = inspect.currentframe().f_back
            file_name = frame.f_code.co_filename
            line_number = frame.f_lineno

        error_message = (
            f"Error occured in the python script name [{file_name}] "
            f"line number [{line_number}] error message [{str(error)}]"
        )
        return error_message

    except Exception as e:
        return f"Exception formatting failed: {str(e)} | Original error: {str(error)}"


class CustomException(Exception):
      def __init__(self, error_message, error_detail:sys):
            super().__init__(error_message)
            self.error_message = error_message_details(error_message, error_detail = error_detail)

      def __str__(self):
            return self.error_message