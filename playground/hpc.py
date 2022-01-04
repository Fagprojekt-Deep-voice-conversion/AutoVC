from paramiko import SSHClient, AutoAddPolicy
from getpass import getpass
import time
import sys, os

class HPC:
    def __init__(self, hostname = 'login1.gbar.dtu.dk') -> None:
        # get credentials 
        self.hostname = hostname
        self.__username = input("Username: ")
        self.__pw = getpass("Password: ")

        # Connect
        self.client = SSHClient()
        # self.client.set_missing_host_key_policy(AutoAddPolicy())
        self.client.load_system_host_keys()
        self.client.connect(hostname, username=self.__username, password=self.__pw, port = '22')#, disabled_algorithms=dict(pubkeys=["rsa-sha2-512"]))
    

    def shell(self):
        # open channel
        channel = self.client.get_transport().open_session()
        channel.get_pty()
        channel.invoke_shell()

        # read welcome message
        time.sleep(1)
        while not channel.recv_ready():
            print("Working...")
            time.sleep(2)
        print(channel.recv(4096).decode("utf-8"))

        # interactive commands
        while True:
            # get command
            command = input('$ ')

            # exit
            if command == 'exit':
                break
            
            # send command
            channel.send(command + "\n")
            time.sleep(.1)

            # read output
            while not channel.recv_ready():
                print("Working...")
                time.sleep(2)
            out = channel.recv(1024).decode("utf-8")
            out = "\n".join(out.split("\n")[1:-2])
            print(out)
        
        # close connection
        self.close()

    def send_data(self, data_path, receive_folder, gbar = True):
        if gbar and "transfer.gbar.dtu.dk" not in receive_folder:
            receive_folder = self.__username + "@transfer.gbar.dtu.dk:" + receive_folder
  
        if "win" in sys.platform:
            command = f'pscp -pw "{self.__pw}" -r {data_path} {receive_folder}'
        else:
            command = f'rsync -r {data_path} {receive_folder}'

        os.system(command)

    def receive_data(self, data_path, receive_folder, gbar = True):
        os.makedirs(receive_folder, exist_ok=True)

        if gbar and "transfer.gbar.dtu.dk" not in data_path:
            data_path = self.__username + "@transfer.gbar.dtu.dk:" + data_path

        if "win" in sys.platform:
            command = f'pscp -pw "{self.__pw}" -scp {data_path} {receive_folder}'
        else:
            command = f'rsync -r {data_path} {receive_folder}'

        os.system(command)
        

    def close(self):
        self.client.close()


if __name__ == "__main__":
    HPC().shell()
    # hpc = HPC()
    # hpc.receive_data("Downloads/args.py", "playground/")
    # hpc.send_data("playground/yang.wav", "Downloads")

    # file = hpc.client.open_sftp()
    # file.get("s183920@transfer.gbar.dtu.dk:Downloads/args.py", "playground/")
    # file.close()

    # hpc.close()


# stdin, stdout, stderr = self.client.exec_command(command)
#         # stdin, stdout, stderr = self.client.exec_command('echo $PWD')

#         print(type(stdin))  # <class 'paramiko.channel.ChannelStdinFile'>
#         print(type(stdout))  # <class 'paramiko.channel.ChannelFile'>
#         print(type(stderr))  # <class 'paramiko.channel.ChannelStderrFile'>

#         # Optionally, send data via STDIN, and shutdown when done
#         # stdin.write('<?php echo "Hello!"; sleep(2); ?>')
#         # stdin.channel.shutdown_write()

#         # Print output of command. Will wait for command to finish.
#         print(f'STDOUT: {stdout.read().decode("utf8")}')
#         print(f'STDERR: {stderr.read().decode("utf8")}')

#         # Get return code from command (0 is default for success)
#         print(f'Return code: {stdout.channel.recv_exit_status()}')

#         # Because they are file objects, they need to be closed
#         stdin.close()
#         stdout.close()
#         stderr.close()

