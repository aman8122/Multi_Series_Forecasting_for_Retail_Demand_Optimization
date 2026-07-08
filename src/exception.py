# Import required modules
import sys
import logging


# Function to generate a detailed error message
# including the file name, line number, and original exception.
def error_message_detail(error, error_detail:sys):

    # Get exception information (type, value, traceback)
    _, _,exc_tb=error_detail.exc_info()

    # Extract the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a detailed error message
    error_message=(f"Error occurred in Python script:[{file_name}] "f"at line number:[{exc_tb.tb_lineno}] "
        f"Error message:[{str(error)}]"
    )

    return error_message


# Custom exception class to provide detailed debugging information
class CustomException(Exception):

    # Initialize the custom exception with detailed error information
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    # Return the formatted error message when the exception is printed
    def __str__(self):
        return self.error_message


# Example usage of the custom exception
if __name__ == "__main__":

    try:
        # Intentionally raising a ZeroDivisionError
        a = 1 / 0

    except Exception as e:

        # Log the error before raising the custom exception
        logging.info("Divide by zero error")

        # Raise the custom exception with detailed traceback information
        raise CustomException(e, sys)