class LogFile:
    def __init__(self, output_folder):
        import os
        # checks if file_path exits. If not creates file_path
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.file = open(os.path.join(output_folder, "log_file.txt"), "w")
        return

    def error_message(self, message):
        r"""
        Error message for the log file

        Parameters
        ----------
        :param message: message that will be displayed
        """

        self.file.write("# Error # : " + message + "\n")
        return

    def info_message(self, message):
        r"""
        Warning message for the log file

        Parameters
        ----------
        :param message: message that will be displayed
        """

        self.file.write("# Info # : " + message + "\n")
        return

    def close(self):
        self.file.close()
        return
