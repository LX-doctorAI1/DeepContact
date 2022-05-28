import os
import sys
import time
import tempfile

################################################################################
# This function returns an instance of a class that decorates a stream.
# The decoration forces the stream to flush when it is written.
################################################################################
def make_unbuffered(stream):
    class UnbufferedStream:
        def __init__(self, chained_stream):
            self._chained_stream = chained_stream

        def write(self, message):
            self._chained_stream.write(message)
            self._chained_stream.flush()

        def flush(self):
            self._chained_stream.flush()
    return UnbufferedStream(stream)

################################################################################
# This class helps returning file names with a fixed pattern and time stamp.
################################################################################
class StdOutRedirectorHelpers:
    def __init__(self):
        # Get a unique time stamp for both files
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def get_filename(self, model_name):
        if len(model_name) > 0:
            model_name = model_name + "_"
        return "ad2trainer_log_" + model_name + self.timestamp + ".txt"

    def get_full_path(self, base_directory):
        if base_directory == "":
            return tempfile.gettempdir()
        else:
            return os.path.join(base_directory, "Log")

def redirect(base_directory="", model_name=""):
    # Create an instance so that all files have the same time stamp
    helper = StdOutRedirectorHelpers()
    # Create the logfile with an absolute path
    folder = helper.get_full_path(base_directory)
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Generate the absolute path including the log file name
    log_file = os.path.join(folder, helper.get_filename(model_name))
    # Redirect stdout and stderr to this file
    sys.stdout = make_unbuffered(open(log_file, "a+"))
    sys.stderr = make_unbuffered(open(log_file, "a+"))
    # return the absolute file path
    return log_file
